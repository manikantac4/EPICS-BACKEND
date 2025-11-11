"""
Flask IoT backend (fixed daily averages calculation, CSV export removed)

Expect POST /api/sensordata JSON:
{
  "mq2_ppm": 17.75,
  "temperature": 30.4,
  "humidity": 79.1,
  "voc_ppm": 244.74,
  "h2s_ppm": 0.30,
  "co_ppm": 11.55,
  "air_quality_percent": 75.5
}

Env variables (create a .env):
MONGO_URI="your_mongo_uri"
FIREBASE_SERVICE_ACCOUNT="path/to/serviceAccountKey.json"
FCM_DEVICE_TOKEN="your_fcm_device_token"
OPENWEATHER_API_KEY="your_openweather_api_key"
"""

from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import threading
import time
import os
import requests
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

# Optional: firebase import guarded because sometimes environments don't have the SDK
firebase_available = False
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    firebase_available = True
except Exception:
    firebase_available = False

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
FIREBASE_SERVICE_ACCOUNT = os.getenv('FIREBASE_SERVICE_ACCOUNT', 'serviceAccountKey.json')
FCM_DEVICE_TOKEN = os.getenv('FCM_DEVICE_TOKEN', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Monitor thread cooldown (sec)
NOTIFY_COOLDOWN_SECONDS = int(os.getenv('NOTIFY_COOLDOWN_SECONDS', '60'))
MONITOR_LOOP_INTERVAL = float(os.getenv('MONITOR_LOOP_INTERVAL', '5'))

# -----------------------------
# ALERT THRESHOLDS (ppm / percent)
# Tweak these values to match calibration/requirements.
# -----------------------------
THRESHOLDS = {
    'mq2': { 'moderate': 50.0,  'extreme': 200.0 },   # LPG/Smoke ppm
    'voc': { 'moderate': 300.0, 'extreme': 1000.0 },  # VOC ppm
    'h2s': { 'moderate': 0.5,   'extreme': 1.5 },     # H2S ppm (very low thresholds)
    'co':  { 'moderate': 35.0,  'extreme': 100.0 },   # CO ppm (WHO short-term ~35 ppm concern)
    'air_quality_percent': { 'bad_threshold': 60.0 }  # percent threshold for "bad"
}

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('iot_backend')

# -----------------------------
# FLASK APP INITIALIZATION
# -----------------------------
app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app)
except Exception:
    pass

# -----------------------------
# MONGODB CONNECTION (ATLAS)
# -----------------------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')  # verify connection
    db = client.get_database()
    # Collections
    sensor_collection = db['sensor_snapshot']    # latest single doc (_id=1)
    avg_collection = db['daily_average_ppm']     # rolling 3-day averages (ppm)
    history_collection = db['sensor_history_ppm']# time-series history
    # Indices
    history_collection.create_index([('timestamp', ASCENDING)])
    avg_collection.create_index([('date', ASCENDING)])
    sensor_collection.create_index([('_id', ASCENDING)])
    logger.info('✅ Connected to MongoDB and indexes ensured')
except Exception as e:
    logger.exception('❌ Failed to connect to MongoDB')
    raise

# -----------------------------
# FIREBASE INITIALIZATION (optional)
# -----------------------------
firebase_initialized = False
if firebase_available:
    try:
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)
            firebase_admin.initialize_app(cred)
            firebase_initialized = True
            logger.info('✅ Firebase Admin initialized')
        else:
            logger.warning('⚠ Firebase service account file not found: %s', FIREBASE_SERVICE_ACCOUNT)
    except Exception:
        logger.exception('❌ Firebase initialization failed; continuing without FCM')
        firebase_initialized = False
else:
    logger.warning('⚠ firebase-admin not installed or import failed; notifications disabled')

# -----------------------------
# GLOBALS
# -----------------------------
latest_alert = {'alert': False, 'message': 'All clear 👍', 'level': 'Good'}
priority_order = ["mq2", "co", "h2s", "voc", "temperature", "humidity", "air_quality_percent"]
monitor_thread = None
monitor_thread_started = threading.Event()

# -----------------------------
# UTILITIES
# -----------------------------
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def _ensure_tz_aware(dt):
    """
    Ensure a datetime is timezone-aware. If naive, assume UTC and set tzinfo=timezone.utc.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _date_range_from_iso(date_iso):
    """
    Given a date string 'YYYY-MM-DD', return (start_dt, end_dt) timezone-aware in UTC
    representing [date 00:00:00, next_date 00:00:00)
    """
    d = datetime.strptime(date_iso, '%Y-%m-%d').date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end

# -----------------------------
# FIREBASE NOTIFICATION FUNCTION
# -----------------------------
# -----------------------------
# Gemini (Google Generative) helper
# -----------------------------
def call_gemini_for_suggestions(three_day_averages, model=GEMINI_MODEL, api_key=GEMINI_API_KEY, timeout=10):
    """
    three_day_averages: list of dicts like returned by /api/viewdailyaverage
    Returns: dict with raw Gemini JSON (or error)
    """
    if not api_key:
        return {'error': 'GEMINI_API_KEY not configured'}

    # Build a compact prompt: include JSON and ask for suggestions
    prompt_lines = []
    prompt_lines.append("You are an assistant that analyzes 3-day indoor air quality summaries and gives:\n"
                        "- short plain-language summary for each day\n"
                        "- overall trend (improving/worsening/stable)\n"
                        "- top 3 recommendations for the user (safety and actions)\n"
                        "- any sensors/values of concern and why\n")
    prompt_lines.append("Here are the 3-day daily averages (date, counts, and average ppm/%):")
    # attach data as a JSON block for clarity
    prompt_lines.append(json.dumps(three_day_averages, indent=2))
    prompt_lines.append("\nRespond with a short JSON object containing: summary_by_day (array of {date, summary}), overall_trend, recommendations (array), concerns (array). Also include a short human-readable 'text' field summarizing findings.")
    prompt_text = "\n\n".join(prompt_lines)

    # Build request body in Gemini generateContent format (simple text contents)
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ],
                # role may be omitted or set; docs accept simple content blocks
            }
        ],
        # You can tune model params here if desired
        "temperature": 0.2,
        "maxOutputTokens": 512
    }

    # Use the documented generateContent REST endpoint (v1beta or v1 depending on your access)
    # Example base: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        # Many examples pass the key as query param ?key=API_KEY or via x-goog-api-key header.
        # We'll send it as an x-goog-api-key header (works with API key auth).
        "x-goog-api-key": api_key
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.exception("❌ call_gemini_for_suggestions failed")
        return {"error": str(e), "status_code": getattr(e.response, "status_code", None),
                "text": getattr(e.response, "text", None)}

# -----------------------------
# API: Ask Gemini for suggestions based on last 3 days
# -----------------------------
@app.route('/api/gemini_suggestions', methods=['GET'])
def gemini_suggestions():
    """
    Returns suggestions from Gemini for the last 3 daily averages.
    Query params:
      model (optional) - override default model
    """
    try:
        model = request.args.get('model', GEMINI_MODEL)

        # Fetch last 3 day averages same as /api/viewdailyaverage logic
        rows_cursor = avg_collection.find({}, {'_id': 0}).sort('date', -1).limit(3)
        rows = list(rows_cursor)
        if not rows:
            return jsonify({'status': 'error', 'message': 'no daily averages available'}), 404

        # We will send the rows directly — keep keys consistent with view_daily_average output
        # Ensure ordering oldest -> newest might be easier for human reading; reverse if currently desc
        three_days = list(reversed(rows))  # oldest first

        # Call Gemini
        model_name = model or GEMINI_MODEL
        gemini_resp = call_gemini_for_suggestions(three_days, model=model_name)

        # Try to extract a readable text from recommended fields (response shape may differ by model version)
        # The Gemini REST response commonly includes a top-level 'candidates' or 'outputs' structure.
        # We'll attempt a safe extraction:
        human_text = None
        try:
            # some docs show top-level 'candidates' with 'content' or 'output' fields
            if isinstance(gemini_resp, dict):
                # new style: 'candidates' or 'output' or 'outputs'
                if 'candidates' in gemini_resp and len(gemini_resp['candidates']) > 0:
                    # candidate might contain 'content' with 'text' or 'output'
                    cand = gemini_resp['candidates'][0]
                    # attempt to find text fields
                    human_text = cand.get('content', {}).get('text') if isinstance(cand.get('content'), dict) else cand.get('content')
                    if not human_text:
                        human_text = cand.get('output') or cand.get('text')
                elif 'outputs' in gemini_resp and len(gemini_resp['outputs']) > 0:
                    # outputs often have 'content' -> list -> {'type':'output_text','text': '...'}
                    out = gemini_resp['outputs'][0]
                    if isinstance(out.get('content'), list):
                        for c in out['content']:
                            if c.get('type') in ('output_text','text'):
                                human_text = c.get('text')
                                break
                elif 'text' in gemini_resp:
                    human_text = gemini_resp.get('text')
        except Exception:
            human_text = None

        return jsonify({
            'status': 'success',
            'model_response': gemini_resp,
            'human_readable': human_text
        }), 200

    except Exception:
        logger.exception('❌ gemini_suggestions endpoint failed')
        return jsonify({'status': 'error', 'message': 'internal error'}), 500


def send_fcm_notification(title, body, level):
    if not firebase_initialized or not FCM_DEVICE_TOKEN:
        logger.info('ℹ Skipping FCM send (firebase not initialized or token missing)')
        return
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=FCM_DEVICE_TOKEN,
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    sound='default',
                    color='#FF0000' if level == 'Extreme' else '#FFA500'
                )
            ),
            data={'click_action': 'FLUTTER_NOTIFICATION_CLICK', 'alert_level': level}
        )
        response = messaging.send(message)
        logger.info('✅ FCM Notification sent: %s', response)
    except Exception:
        logger.exception('❌ Failed to send FCM Notification')

# -----------------------------
# ALERT EVALUATION (ppm-aware)
# -----------------------------
def evaluate_alerts(doc):
    """
    doc is expected to contain:
    mq2_ppm, temperature, humidity, voc_ppm, h2s_ppm, co_ppm, air_quality_percent
    """
    alerts = {}

    mq2 = safe_float(doc.get('mq2_ppm', 0.0))
    temperature = safe_float(doc.get('temperature', 0.0))
    humidity = safe_float(doc.get('humidity', 0.0))
    voc = safe_float(doc.get('voc_ppm', 0.0))
    h2s = safe_float(doc.get('h2s_ppm', 0.0))
    co = safe_float(doc.get('co_ppm', 0.0))
    aq_percent = safe_float(doc.get('air_quality_percent', 100.0))

    # MQ-2 (LPG/Smoke)
    if THRESHOLDS['mq2']['moderate'] < mq2 <= THRESHOLDS['mq2']['extreme']:
        alerts['mq2'] = ('Moderate', f'⚠ MODERATE: LPG/Smoke detected {mq2:.2f} ppm — ventilate.')
    elif mq2 > THRESHOLDS['mq2']['extreme']:
        alerts['mq2'] = ('Extreme', f'🚨 EXTREME: LPG/Smoke: {mq2:.2f} ppm — take immediate action!')

    # CO (MQ-7)
    if THRESHOLDS['co']['moderate'] < co <= THRESHOLDS['co']['extreme']:
        alerts['co'] = ('Moderate', f'⚠ MODERATE: CO rising {co:.2f} ppm — ventilate.')
    elif co > THRESHOLDS['co']['extreme']:
        alerts['co'] = ('Extreme', f'🚨 EXTREME: High CO {co:.2f} ppm — danger!')

    # H2S (MQ-136) - small ppm values are dangerous
    if THRESHOLDS['h2s']['moderate'] < h2s <= THRESHOLDS['h2s']['extreme']:
        alerts['h2s'] = ('Moderate', f'⚠ MODERATE: H2S detected {h2s:.3f} ppm.')
    elif h2s > THRESHOLDS['h2s']['extreme']:
        alerts['h2s'] = ('Extreme', f'🚨 EXTREME: High H2S {h2s:.3f} ppm — evacuate area!')

    # VOC
    if THRESHOLDS['voc']['moderate'] < voc <= THRESHOLDS['voc']['extreme']:
        alerts['voc'] = ('Moderate', f'⚠ MODERATE: VOCs {voc:.2f} ppm — ventilate.')
    elif voc > THRESHOLDS['voc']['extreme']:
        alerts['voc'] = ('Extreme', f'🚨 EXTREME: VOC level critical {voc:.2f} ppm!')

    # Temperature
    if 32.0 < temperature <= 35.0:
        alerts['temperature'] = ('Moderate', '🌡 MODERATE: Slightly high temperature.')
    elif temperature > 35.0:
        alerts['temperature'] = ('Extreme', '🔥 EXTREME: Temperature too high!')

    # Humidity
    if (70.0 < humidity <= 80.0) or (20.0 < humidity <= 30.0):
        alerts['humidity'] = ('Moderate', '💧 MODERATE: Humidity slightly off.')
    elif humidity > 90.0 or humidity < 20.0:
        alerts['humidity'] = ('Extreme', '💧 EXTREME: Humidity critical!')

    # Air Quality percent (user-facing)
    if aq_percent < THRESHOLDS['air_quality_percent']['bad_threshold']:
        alerts['air_quality_percent'] = ('Moderate', f'⚠ Air Quality degrading: {aq_percent:.1f}%')

    # Resolve by priority
    for key in priority_order:
        if key in alerts:
            level, message = alerts[key]
            return True, level, message
    return False, 'Good', '✅ Air Quality: GOOD'

# -----------------------------
# DATABASE INITIALIZATION
# -----------------------------
def initialize_db():
    try:
        if sensor_collection.count_documents({}) == 0:
            sensor_collection.insert_one({
                '_id': 1,
                'mq2_ppm': 0.0,
                'temperature': 0.0,
                'humidity': 0.0,
                'voc_ppm': 0.0,
                'h2s_ppm': 0.0,
                'co_ppm': 0.0,
                'air_quality_percent': 100.0,
                # store timezone-aware ISO timestamp (UTC)
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')
            })
            logger.info('✅ MongoDB ready with initial sensor document (ppm fields)')
    except Exception:
        logger.exception('❌ initialize_db failed')

# -----------------------------
# API: Receive Sensor Data
# -----------------------------
@app.route('/api/sensordata', methods=['POST'])
def receive_data():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    logger.info('📩 Received data: %s', data)

    # Parse and coerce numeric values safely
    mq2_ppm = safe_float(data.get('mq2_ppm', data.get('gas_value', 0.0)))
    temperature = safe_float(data.get('temperature', 0.0))
    humidity = safe_float(data.get('humidity', 0.0))
    voc_ppm = safe_float(data.get('voc_ppm', data.get('voc', 0.0)))
    h2s_ppm = safe_float(data.get('h2s_ppm', data.get('h2s', 0.0)))
    co_ppm = safe_float(data.get('co_ppm', data.get('co', 0.0)))
    air_quality_percent = safe_float(data.get('air_quality_percent', data.get('voc_percent', 0.0)))
    timestamp = datetime.now(timezone.utc)  # timezone-aware UTC

    # Update latest doc (atomic upsert)
    try:
        sensor_collection.update_one(
            {'_id': 1},
            {'$set': {
                'mq2_ppm': mq2_ppm,
                'temperature': temperature,
                'humidity': humidity,
                'voc_ppm': voc_ppm,
                'h2s_ppm': h2s_ppm,
                'co_ppm': co_ppm,
                'air_quality_percent': air_quality_percent,
                # save a human-readable UTC timestamp with offset
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %z')
            }},
            upsert=True
        )
    except Exception:
        logger.exception('❌ Failed to update latest sensor document')

    # Insert into history (store tz-aware datetime)
    try:
        history_doc = {
            'mq2_ppm': mq2_ppm,
            'temperature': temperature,
            'humidity': humidity,
            'voc_ppm': voc_ppm,
            'h2s_ppm': h2s_ppm,
            'co_ppm': co_ppm,
            'air_quality_percent': air_quality_percent,
            'timestamp': timestamp  # tz-aware datetime
        }
        history_collection.insert_one(history_doc)
    except Exception:
        logger.exception('❌ Failed to insert history doc')

    # Update daily averages
    try:
        update_daily_average({
            'mq2_ppm': mq2_ppm,
            'temperature': temperature,
            'humidity': humidity,
            'voc_ppm': voc_ppm,
            'h2s_ppm': h2s_ppm,
            'co_ppm': co_ppm,
            'air_quality_percent': air_quality_percent
        })
    except Exception:
        logger.exception('❌ update_daily_average failed')

    return jsonify({'status': 'success', 'message': 'Data updated successfully'}), 200

# -----------------------------
# API: Get Latest Sensor Data
# -----------------------------
@app.route('/api/viewdata', methods=['GET'])
def view_data():
    try:
        row = sensor_collection.find_one({'_id': 1}, {'_id': 0})
        return jsonify(row if row else {}), 200
    except Exception:
        logger.exception('❌ view_data failed')
        return jsonify({}), 500

# -----------------------------
# API: Get Latest Alert
# -----------------------------
@app.route('/api/alertstatus', methods=['GET'])
def get_alert():
    return jsonify(latest_alert), 200

# -----------------------------
# BACKGROUND MONITOR LOOP
# -----------------------------
def monitor_loop(notify_cooldown_seconds=NOTIFY_COOLDOWN_SECONDS, interval=MONITOR_LOOP_INTERVAL):
    global latest_alert
    last_sent_message = None
    # timezone-aware minimum time
    last_sent_time = datetime.min.replace(tzinfo=timezone.utc)

    logger.info('🔁 monitor_loop started (interval=%ss)', interval)

    while True:
        try:
            row = sensor_collection.find_one({'_id': 1})
        except Exception:
            logger.exception('⚠ monitor_loop DB read error')
            time.sleep(5)
            continue

        if row:
            alert_flag, level, msg = evaluate_alerts(row)
            latest_alert = {'alert': alert_flag, 'level': level, 'message': msg}
            logger.info('🔔 [Monitor] %s', msg)

            if alert_flag:
                now = datetime.now(timezone.utc)
                cooldown_ok = (now - last_sent_time) >= timedelta(seconds=notify_cooldown_seconds)
                if (msg != last_sent_message) or cooldown_ok:
                    try:
                        send_fcm_notification('Air Quality Alert', msg, level)
                        last_sent_message = msg
                        last_sent_time = now
                        logger.info('✅ Notification sent (message=%s)', msg[:80])
                    except Exception:
                        logger.exception('❌ Failed to send notification in monitor_loop')
        else:
            logger.info('ℹ monitor_loop: no latest sensor doc found yet')

        time.sleep(interval)

# -----------------------------
# WEATHER API
# -----------------------------
@app.route('/get-data-weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', '16.4821')
    lon = request.args.get('lon', '80.6914')
    api_key = OPENWEATHER_API_KEY
    if not api_key:
        return jsonify({'error': 'OpenWeather API key not configured'}), 400

    url = 'http://api.openweathermap.org/data/2.5/air_pollution'
    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        print('🌤 Weather API response:', response.json())
        return jsonify(response.json()), 200
    except Exception:
        logger.exception('❌ get_weather failed')
        return jsonify({'error': 'Failed to fetch weather/air pollution data'}), 500

# -----------------------------
# UPDATE DAILY AVERAGE VALUES (ppm-aware)
# Keeps rolling latest 3 days
# Now stores computed averages directly in MongoDB
# -----------------------------
def update_daily_average(data):
    # use timezone-aware current date (UTC)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # ensure numeric values
    mq2 = safe_float(data.get('mq2_ppm', 0.0))
    temperature = safe_float(data.get('temperature', 0.0))
    humidity = safe_float(data.get('humidity', 0.0))
    voc = safe_float(data.get('voc_ppm', 0.0))
    h2s = safe_float(data.get('h2s_ppm', 0.0))
    co = safe_float(data.get('co_ppm', 0.0))
    aq = safe_float(data.get('air_quality_percent', 0.0))

    # Atomic upsert: increment sums and count, set date on insert
    try:
        result = avg_collection.update_one(
            {'date': today},
            {
                '$inc': {
                    'mq2_sum': mq2,
                    'temperature_sum': temperature,
                    'humidity_sum': humidity,
                    'voc_sum': voc,
                    'h2s_sum': h2s,
                    'co_sum': co,
                    'air_quality_sum': aq,
                    'count': 1
                },
                '$setOnInsert': {
                    'date': today
                }
            },
            upsert=True
        )
        
        # After updating sums, compute and store the actual averages
        doc = avg_collection.find_one({'date': today})
        if doc and doc.get('count', 0) > 0:
            count = doc['count']
            avg_collection.update_one(
                {'date': today},
                {'$set': {
                    'avg_mq2_ppm': round(doc.get('mq2_sum', 0.0) / count, 3),
                    'avg_temperature': round(doc.get('temperature_sum', 0.0) / count, 2),
                    'avg_humidity': round(doc.get('humidity_sum', 0.0) / count, 2),
                    'avg_voc_ppm': round(doc.get('voc_sum', 0.0) / count, 3),
                    'avg_h2s_ppm': round(doc.get('h2s_sum', 0.0) / count, 4),
                    'avg_co_ppm': round(doc.get('co_sum', 0.0) / count, 3),
                    'avg_air_quality_percent': round(doc.get('air_quality_sum', 0.0) / count, 2)
                }}
            )
    except Exception:
        logger.exception('❌ update_daily_average: failed to upsert')

    # Cleanup keep only latest 3 days (do this safely)
    try:
        all_docs = list(avg_collection.find().sort('date', -1))
        if len(all_docs) > 3:
            for doc in all_docs[3:]:
                try:
                    avg_collection.delete_one({'_id': doc['_id']})
                except Exception:
                    logger.exception('⚠ Failed to delete old average doc %s', doc.get('_id'))
    except Exception:
        logger.exception('⚠ Failed to cleanup old averages')

# -----------------------------
# API: Get Last 3 Days Average Data (FIXED: Returns actual averages, not sums)
# -----------------------------
@app.route('/api/viewdailyaverage', methods=['GET'])
def view_daily_average():
    """
    Return last 3 days of daily averages.
    Computes averages by dividing stored sums by count.
    If count is missing/zero for a date, fallback to compute from history_collection.
    """
    try:
        rows_cursor = avg_collection.find({}, {'_id': 0}).sort('date', -1).limit(3)
        rows = list(rows_cursor)
        averages = []

        for row in rows:
            date_str = row.get('date')
            count = int(row.get('count', 0))

            if count > 0:
                # FIXED: Compute averages by dividing sums by count
                averages.append({
                    'date': date_str,
                    'avg_mq2_ppm': round(row.get('mq2_sum', 0.0) / count, 3),
                    'avg_temperature': round(row.get('temperature_sum', 0.0) / count, 2),
                    'avg_humidity': round(row.get('humidity_sum', 0.0) / count, 2),
                    'avg_voc_ppm': round(row.get('voc_sum', 0.0) / count, 3),
                    'avg_h2s_ppm': round(row.get('h2s_sum', 0.0) / count, 4),
                    'avg_co_ppm': round(row.get('co_sum', 0.0) / count, 3),
                    'avg_air_quality_percent': round(row.get('air_quality_sum', 0.0) / count, 2),
                    'count': count
                })
            else:
                # fallback to compute the averages from history (safe)
                if not date_str:
                    continue
                start_dt, end_dt = _date_range_from_iso(date_str)
                # aggregation to compute averages and count
                pipeline = [
                    {'$match': {'timestamp': {'$gte': start_dt, '$lt': end_dt}}},
                    {'$group': {
                        '_id': None,
                        'avg_mq2_ppm': {'$avg': '$mq2_ppm'},
                        'avg_temperature': {'$avg': '$temperature'},
                        'avg_humidity': {'$avg': '$humidity'},
                        'avg_voc_ppm': {'$avg': '$voc_ppm'},
                        'avg_h2s_ppm': {'$avg': '$h2s_ppm'},
                        'avg_co_ppm': {'$avg': '$co_ppm'},
                        'avg_air_quality_percent': {'$avg': '$air_quality_percent'},
                        'count': {'$sum': 1}
                    }}
                ]
                agg = list(history_collection.aggregate(pipeline))
                if agg:
                    a = agg[0]
                    c = int(a.get('count', 0)) or 0
                    averages.append({
                        'date': date_str,
                        'avg_mq2_ppm': round(a.get('avg_mq2_ppm', 0.0) or 0.0, 3),
                        'avg_temperature': round(a.get('avg_temperature', 0.0) or 0.0, 2),
                        'avg_humidity': round(a.get('avg_humidity', 0.0) or 0.0, 2),
                        'avg_voc_ppm': round(a.get('avg_voc_ppm', 0.0) or 0.0, 3),
                        'avg_h2s_ppm': round(a.get('avg_h2s_ppm', 0.0) or 0.0, 4),
                        'avg_co_ppm': round(a.get('avg_co_ppm', 0.0) or 0.0, 3),
                        'avg_air_quality_percent': round(a.get('avg_air_quality_percent', 0.0) or 0.0, 2),
                        'count': c
                    })
                else:
                    # no history found for that date
                    averages.append({
                        'date': date_str,
                        'avg_mq2_ppm': 0.0,
                        'avg_temperature': 0.0,
                        'avg_humidity': 0.0,
                        'avg_voc_ppm': 0.0,
                        'avg_h2s_ppm': 0.0,
                        'avg_co_ppm': 0.0,
                        'avg_air_quality_percent': 0.0,
                        'count': 0
                    })

        return jsonify(averages), 200
    except Exception:
        logger.exception('❌ view_daily_average failed')
        return jsonify([]), 500

# -----------------------------
# View history (last N records)
# -----------------------------
@app.route('/api/viewhistory', methods=['GET'])
def view_history():
    try:
        limit = int(request.args.get('limit', 100))
    except Exception:
        limit = 100
    try:
        cursor = history_collection.find().sort('timestamp', -1).limit(limit)
        data_out = []
        for doc in cursor:
            ts = doc.get('timestamp')
            ts = _ensure_tz_aware(ts)
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S %z') if ts else ''
            data_out.append({
                'mq2_ppm': doc.get('mq2_ppm', 0.0),
                'temperature': doc.get('temperature', 0.0),
                'humidity': doc.get('humidity', 0.0),
                'voc_ppm': doc.get('voc_ppm', 0.0),
                'h2s_ppm': doc.get('h2s_ppm', 0.0),
                'co_ppm': doc.get('co_ppm', 0.0),
                'air_quality_percent': doc.get('air_quality_percent', 0.0),
                'timestamp': ts_str
            })
        return jsonify(data_out), 200
    except Exception:
        logger.exception('❌ view_history failed')
        return jsonify([]), 500

# -----------------------------
# Maintenance endpoint: recompute daily averages from history
# Useful to repair counts/sums if previous logic was incorrect
# -----------------------------
@app.route('/api/recompute_daily_averages', methods=['POST'])
def recompute_daily_averages():
    """
    Recompute and overwrite daily_average_ppm documents for the last N days using history_collection.
    Body: {"days": 7}  (optional, default 7)
    """
    try:
        body = request.get_json(silent=True) or {}
        days = int(body.get('days', 7))
    except Exception:
        days = 7

    try:
        now = datetime.now(timezone.utc)
        for i in range(days):
            day = now - timedelta(days=i)
            date_str = day.strftime('%Y-%m-%d')
            start_dt, end_dt = _date_range_from_iso(date_str)

            pipeline = [
                {'$match': {'timestamp': {'$gte': start_dt, '$lt': end_dt}}},
                {'$group': {
                    '_id': None,
                    'mq2_sum': {'$sum': '$mq2_ppm'},
                    'temperature_sum': {'$sum': '$temperature'},
                    'humidity_sum': {'$sum': '$humidity'},
                    'voc_sum': {'$sum': '$voc_ppm'},
                    'h2s_sum': {'$sum': '$h2s_ppm'},
                    'co_sum': {'$sum': '$co_ppm'},
                    'air_quality_sum': {'$sum': '$air_quality_percent'},
                    'count': {'$sum': 1}
                }}
            ]
            agg = list(history_collection.aggregate(pipeline))
            if agg and agg[0].get('count', 0) > 0:
                doc = agg[0]
                # upsert the sums and count
                avg_collection.update_one(
                    {'date': date_str},
                    {'$set': {
                        'mq2_sum': float(doc.get('mq2_sum', 0.0)),
                        'temperature_sum': float(doc.get('temperature_sum', 0.0)),
                        'humidity_sum': float(doc.get('humidity_sum', 0.0)),
                        'voc_sum': float(doc.get('voc_sum', 0.0)),
                        'h2s_sum': float(doc.get('h2s_sum', 0.0)),
                        'co_sum': float(doc.get('co_sum', 0.0)),
                        'air_quality_sum': float(doc.get('air_quality_sum', 0.0)),
                        'count': int(doc.get('count', 0))
                    }},
                    upsert=True
                )
            else:
                # remove any existing doc if no history for that date
                avg_collection.delete_one({'date': date_str})

        return jsonify({'status': 'success', 'message': f'recomputed last {days} days'}), 200
    except Exception:
        logger.exception('❌ recompute_daily_averages failed')
        return jsonify({'status': 'error', 'message': 'recompute failed'}), 500
# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    initialize_db()

    # Start monitor thread only once
    if not monitor_thread_started.is_set():
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        monitor_thread_started.set()
        logger.info('✅ monitor thread started')

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True)
