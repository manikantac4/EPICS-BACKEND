"""
Flask IoT backend with Scheduled Notifications and Daily Routines

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
GEMINI_API_KEY="your_gemini_api_key"
TIMEZONE="Asia/Kolkata"
"""
import json
import traceback

from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone, time as dt_time
import threading
import time
import math
import joblib
import os
import requests
import logging
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd 
from google import generativeai as genai
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from ml.co_predictor import predict_co_api
from ml.mq2_predictor import predict_mq2_api
from ml.voc_predictor import predict_voc_api
from ml.h2s_predictor import predict_h2s_api
from ml.humidity_predictor import predict_humidity_api
from ml.temperature import predict_temperature_api
# APScheduler for scheduled notifications
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# Optional: firebase import guarded
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

# Timezone for scheduled notifications
TIMEZONE = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Kolkata'))

# Monitor thread cooldown (sec)
NOTIFY_COOLDOWN_SECONDS = int(os.getenv('NOTIFY_COOLDOWN_SECONDS', '60'))
MONITOR_LOOP_INTERVAL = float(os.getenv('MONITOR_LOOP_INTERVAL', '5'))

# -----------------------------
# ALERT THRESHOLDS (ppm / percent)
# -----------------------------
THRESHOLDS = {
    'mq2': { 'moderate': 50.0,  'extreme': 200.0 },
    'voc': { 'moderate': 300.0, 'extreme': 1000.0 },
    'h2s': { 'moderate': 0.5,   'extreme': 1.5 },
    'co':  { 'moderate': 35.0,  'extreme': 100.0 },
    'air_quality_percent': { 'bad_threshold': 60.0 }
}

# -----------------------------
# DAILY ROUTINE CONFIGURATIONS
# -----------------------------
DAILY_ROUTINES = {
    'morning_fresh_air': {
        'time': (6, 0),
        'title': '🌅 Morning Fresh-Air Routine',
        'message': 'Open windows for 15-20 minutes. Let fresh outdoor air circulate before cooking begins.',
        'tips': [
            'Sunlight kills airborne microbes and reduces VOC buildup',
            'Ideal temperature: 22-26°C, humidity 40-60%',
            'If outdoor AQI < 100, open doors/windows fully'
        ],
        'level': 'Good'
    },
    'cooking_safety': {
        'time': (8, 0),
        'title': '🍳 Cooking Safety Check',
        'message': 'Ensure kitchen ventilation and exhaust fan are ON during cooking.',
        'tips': [
            'Cooking increases LPG, CO₂, and VOCs',
            'Use chimney or open balcony door',
            'Avoid staying close to stove if readings rise'
        ],
        'level': 'Good'
    },
    'midday_refresh': {
        'time': (11, 0),
        'title': '☀️ Midday Air Refresh',
        'message': 'Short ventilation break (5-10 minutes). Open windows briefly.',
        'tips': [
            'CO₂ can build up indoors even after morning airing',
            'Run air purifier on Auto mode',
            'Keep humidity between 40-65%'
        ],
        'level': 'Good'
    },
    'humidity_check': {
        'time': (16, 0),
        'title': '🌇 Humidity & Dust Check',
        'message': 'Inspect humidity and dust levels in your home.',
        'tips': [
            'High humidity encourages mold growth',
            'If humidity > 70%, use dehumidifier',
            'Wipe surfaces and clear corners'
        ],
        'level': 'Good'
    },
    'evening_boost': {
        'time': (19, 0),
        'title': '🌆 Indoor Clean-Air Boost',
        'message': 'Run ventilation or air purifier for 20 minutes.',
        'tips': [
            'Indoor VOCs rise after traffic hours',
            'Avoid incense or strong fresheners',
            'Use natural candles (soy, beeswax) for fragrance'
        ],
        'level': 'Good'
    },
    'night_comfort': {
        'time': (21, 30),
        'title': '🌙 Night Comfort Check',
        'message': 'Before bed, ensure sensors read safe levels.',
        'tips': [
            'Ideal sleep: CO₂ < 800 ppm, Temp ~25°C',
            'Slightly open window if CO₂ is high',
            'Unplug chargers to reduce ozone/VOC drift'
        ],
        'level': 'Good'
    }
}

WEEKLY_ROUTINES = {
    'monday': {
        'day': 0,
        'time': (9, 0),
        'title': '🧹 Monday Maintenance',
        'message': 'Wipe ceiling fans and A/C filters.',
        'tips': ['Clean filters improve air circulation', 'Reduces dust and allergens']
    },
    'wednesday': {
        'day': 2,
        'time': (10, 0),
        'title': '🪴 Wednesday Plant Check',
        'message': 'Check indoor plants\' soil - avoid over-watering.',
        'tips': ['Over-watering increases VOC/mold risk', 'Ensure proper drainage']
    },
    'friday': {
        'day': 4,
        'time': (10, 0),
        'title': '🧼 Friday Deep Clean',
        'message': 'Deep-clean kitchen and bathroom vents.',
        'tips': ['Prevents grease and dust buildup', 'Improves ventilation efficiency']
    },
    'sunday': {
        'day': 6,
        'time': (10, 0),
        'title': '☀️ Sunday Airing Day',
        'message': 'Balcony/window airing for 30 min. Sun-dry bedsheets or cushions.',
        'tips': ['UV rays kill bacteria and dust mites', 'Fresh air removes indoor odors']
    }
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
# MONGODB CONNECTION
# -----------------------------
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client.get_database()
    sensor_collection = db['sensor_snapshot']
    avg_collection = db['daily_average_ppm']
    history_collection = db['sensor_history_ppm']
    history_collection.create_index([('timestamp', ASCENDING)])
    avg_collection.create_index([('date', ASCENDING)])
    sensor_collection.create_index([('_id', ASCENDING)])
    logger.info('✅ Connected to MongoDB and indexes ensured')
except Exception as e:
    logger.exception('❌ Failed to connect to MongoDB')
    raise

# -----------------------------
# FIREBASE INITIALIZATION
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
    logger.warning('⚠ firebase-admin not installed; notifications disabled')

# -----------------------------
# GEMINI INITIALIZATION
# -----------------------------
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized successfully")
    except Exception:
        logger.exception("❌ Failed to initialize Gemini client")
else:
    logger.warning("⚠ GEMINI_API_KEY not found; Gemini suggestions disabled")

# -----------------------------
# APSCHEDULER INITIALIZATION
# -----------------------------
scheduler = BackgroundScheduler(timezone=TIMEZONE)

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
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _date_range_from_iso(date_iso):
    d = datetime.strptime(date_iso, '%Y-%m-%d').date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end

# The main API endpoint
@app.route("/api/predict/co", methods=["GET"])
def predict_co():
    return predict_co_api(history_collection)

@app.route("/api/predict/mq2", methods=["GET"])
def predict_mq2():
    return predict_mq2_api(history_collection)

@app.route("/api/predict/voc", methods=["GET"])
def predict_voc():
    return predict_voc_api(history_collection)

@app.route("/api/predict/h2s", methods=["GET"])
def predict_h2s():
    return predict_h2s_api(history_collection)

@app.route("/api/predict/humidity", methods=["GET"])
def predict_humidity():
    return predict_humidity_api(history_collection)

@app.route("/api/predict/temperature", methods=["GET"])
def predict_temperature():
    return predict_temperature_api(history_collection)


# FCM NOTIFICATION FUNCTION
# -----------------------------
def send_fcm_notification(title, body, level):
    """Send FCM notification to the configured device token"""
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
                    color='#FF0000' if level == 'Extreme' else '#FFA500' if level == 'Moderate' else '#4CAF50'
                )
            ),
            data={'click_action': 'FLUTTER_NOTIFICATION_CLICK', 'alert_level': level}
        )
        response = messaging.send(message)
        logger.info('✅ FCM Notification sent: %s', response)
    except Exception:
        logger.exception('❌ Failed to send FCM Notification')

# -----------------------------
# ROUTINE NOTIFICATION FUNCTIONS
# -----------------------------
def send_routine_notification(routine_key, routine_data):
    """Send notification for a routine reminder with sensor context"""
    try:
        title = routine_data['title']
        message = routine_data['message']
        level = routine_data.get('level', 'Good')
        
        # Add sensor context if available
        row = sensor_collection.find_one({'_id': 1})
        if row:
            temp = safe_float(row.get('temperature', 0))
            humidity = safe_float(row.get('humidity', 0))
            co_ppm = safe_float(row.get('co_ppm', 0))
            
            # Add contextual info
            context = f"\n\nCurrent: Temp {temp:.1f}°C, Humidity {humidity:.1f}%"
            
            # Special handling for specific routines
            if routine_key == 'morning_fresh_air' and humidity > 70:
                message += " ⚠️ High humidity detected - extra ventilation recommended!"
            elif routine_key == 'night_comfort' and co_ppm > 80:
                message += " ⚠️ CO levels elevated - open a window slightly!"
            elif routine_key == 'humidity_check':
                if humidity > 70:
                    message += f" ⚠️ Current humidity {humidity:.1f}% - use dehumidifier!"
                elif humidity < 40:
                    message += f" ⚠️ Low humidity {humidity:.1f}% - consider humidifier!"
                else:
                    message += f" ✅ Humidity is good at {humidity:.1f}%"
            
            message += context
        
        # Send FCM notification
        send_fcm_notification(title, message, level)
        logger.info(f"✅ Sent routine reminder: {routine_key}")
        
    except Exception:
        logger.exception(f"❌ Failed to send routine notification: {routine_key}")

# Define routine functions
def morning_fresh_air():
    send_routine_notification('morning_fresh_air', DAILY_ROUTINES['morning_fresh_air'])

def cooking_safety():
    send_routine_notification('cooking_safety', DAILY_ROUTINES['cooking_safety'])

def midday_refresh():
    send_routine_notification('midday_refresh', DAILY_ROUTINES['midday_refresh'])

def humidity_check():
    send_routine_notification('humidity_check', DAILY_ROUTINES['humidity_check'])

def evening_boost():
    send_routine_notification('evening_boost', DAILY_ROUTINES['evening_boost'])

def night_comfort():
    send_routine_notification('night_comfort', DAILY_ROUTINES['night_comfort'])

def monday_maintenance():
    send_routine_notification('monday', WEEKLY_ROUTINES['monday'])

def wednesday_plant_check():
    send_routine_notification('wednesday', WEEKLY_ROUTINES['wednesday'])

def friday_deep_clean():
    send_routine_notification('friday', WEEKLY_ROUTINES['friday'])

def sunday_airing():
    send_routine_notification('sunday', WEEKLY_ROUTINES['sunday'])

# -----------------------------
# SCHEDULE ROUTINES
# -----------------------------
def schedule_daily_routines():
    """Schedule all daily air quality routines"""
    try:
        # Daily routines
        func_map = {
            'morning_fresh_air': morning_fresh_air,
            'cooking_safety': cooking_safety,
            'midday_refresh': midday_refresh,
            'humidity_check': humidity_check,
            'evening_boost': evening_boost,
            'night_comfort': night_comfort
        }
        
        for routine_key, routine_data in DAILY_ROUTINES.items():
            hour, minute = routine_data['time']
            scheduler.add_job(
                func_map[routine_key],
                trigger='cron',
                hour=hour,
                minute=minute,
                timezone=TIMEZONE,
                id=f'routine_{routine_key}'
            )
            logger.info(f"✅ Scheduled {routine_key} at {hour:02d}:{minute:02d}")
        
        # Weekly routines
        scheduler.add_job(monday_maintenance, trigger='cron', day_of_week='mon', hour=9, minute=0, timezone=TIMEZONE, id='routine_monday')
        scheduler.add_job(wednesday_plant_check, trigger='cron', day_of_week='wed', hour=10, minute=0, timezone=TIMEZONE, id='routine_wednesday')
        scheduler.add_job(friday_deep_clean, trigger='cron', day_of_week='fri', hour=10, minute=0, timezone=TIMEZONE, id='routine_friday')
        scheduler.add_job(sunday_airing, trigger='cron', day_of_week='sun', hour=10, minute=0, timezone=TIMEZONE, id='routine_sunday')
        
        logger.info("✅ All daily routines scheduled")
    except Exception:
        logger.exception("❌ Failed to schedule routines")

# -----------------------------
# GEMINI HELPER
# -----------------------------
def call_gemini_for_suggestions(three_day_averages, model=GEMINI_MODEL):
    if not gemini_client:
        return {"error": "Gemini client not initialized or missing API key"}
    try:
        prompt = (
            "You are an assistant analyzing indoor air quality for the past 3 days. "
            "Summarize the overall air quality trends, highlight any risks, "
            "and provide 3 simple safety recommendations for the user.\n\n"
            "Here are the 3 days of averaged sensor data:\n"
            f"{json.dumps(three_day_averages, indent=2)}\n\n"
            "Return a short and clear explanation."
        )
        response = gemini_client.models.generate_content(model=model, contents=prompt)
        text_out = getattr(response, "text", None)
        if not text_out and hasattr(response, "candidates"):
            text_out = response.candidates[0].content.parts[0].text
        return {"text": text_out or "No output text", "raw": str(response)}
    except Exception as e:
        logger.exception("❌ Gemini generation failed")
        return {"error": str(e)}

# -----------------------------
# ALERT EVALUATION
# -----------------------------
def evaluate_alerts(doc):
    alerts = {}
    mq2 = safe_float(doc.get('mq2_ppm', 0.0))
    temperature = safe_float(doc.get('temperature', 0.0))
    humidity = safe_float(doc.get('humidity', 0.0))
    voc = safe_float(doc.get('voc_ppm', 0.0))
    h2s = safe_float(doc.get('h2s_ppm', 0.0))
    co = safe_float(doc.get('co_ppm', 0.0))
    aq_percent = safe_float(doc.get('air_quality_percent', 100.0))

    if THRESHOLDS['mq2']['moderate'] < mq2 <= THRESHOLDS['mq2']['extreme']:
        alerts['mq2'] = ('Moderate', f'⚠ MODERATE: LPG/Smoke detected {mq2:.2f} ppm — ventilate.')
    elif mq2 > THRESHOLDS['mq2']['extreme']:
        alerts['mq2'] = ('Extreme', f'🚨 EXTREME: LPG/Smoke: {mq2:.2f} ppm — take immediate action!')

    if THRESHOLDS['co']['moderate'] < co <= THRESHOLDS['co']['extreme']:
        alerts['co'] = ('Moderate', f'⚠ MODERATE: CO rising {co:.2f} ppm — ventilate.')
    elif co > THRESHOLDS['co']['extreme']:
        alerts['co'] = ('Extreme', f'🚨 EXTREME: High CO {co:.2f} ppm — danger!')

    if THRESHOLDS['h2s']['moderate'] < h2s <= THRESHOLDS['h2s']['extreme']:
        alerts['h2s'] = ('Moderate', f'⚠ MODERATE: H2S detected {h2s:.3f} ppm.')
    elif h2s > THRESHOLDS['h2s']['extreme']:
        alerts['h2s'] = ('Extreme', f'🚨 EXTREME: High H2S {h2s:.3f} ppm — evacuate area!')

    if THRESHOLDS['voc']['moderate'] < voc <= THRESHOLDS['voc']['extreme']:
        alerts['voc'] = ('Moderate', f'⚠ MODERATE: VOCs {voc:.2f} ppm — ventilate.')
    elif voc > THRESHOLDS['voc']['extreme']:
        alerts['voc'] = ('Extreme', f'🚨 EXTREME: VOC level critical {voc:.2f} ppm!')

    if 32.0 < temperature <= 35.0:
        alerts['temperature'] = ('Moderate', '🌡 MODERATE: Slightly high temperature.')
    elif temperature > 35.0:
        alerts['temperature'] = ('Extreme', '🔥 EXTREME: Temperature too high!')

    if (70.0 < humidity <= 80.0) or (20.0 < humidity <= 30.0):
        alerts['humidity'] = ('Moderate', '💧 MODERATE: Humidity slightly off.')
    elif humidity > 90.0 or humidity < 20.0:
        alerts['humidity'] = ('Extreme', '💧 EXTREME: Humidity critical!')

    if aq_percent < THRESHOLDS['air_quality_percent']['bad_threshold']:
        alerts['air_quality_percent'] = ('Moderate', f'⚠ Air Quality degrading: {aq_percent:.1f}%')

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
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')
            })
            logger.info('✅ MongoDB ready with initial sensor document')
    except Exception:
        logger.exception('❌ initialize_db failed')

# -----------------------------
# DAILY AVERAGE UPDATE
# -----------------------------
def update_daily_average(data):
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    mq2 = safe_float(data.get('mq2_ppm', 0.0))
    temperature = safe_float(data.get('temperature', 0.0))
    humidity = safe_float(data.get('humidity', 0.0))
    voc = safe_float(data.get('voc_ppm', 0.0))
    h2s = safe_float(data.get('h2s_ppm', 0.0))
    co = safe_float(data.get('co_ppm', 0.0))
    aq = safe_float(data.get('air_quality_percent', 0.0))

    try:
        avg_collection.update_one(
            {'date': today},
            {
                '$inc': {
                    'mq2_sum': mq2, 'temperature_sum': temperature, 'humidity_sum': humidity,
                    'voc_sum': voc, 'h2s_sum': h2s, 'co_sum': co, 'air_quality_sum': aq, 'count': 1
                },
                '$setOnInsert': {'date': today}
            },
            upsert=True
        )
        
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
        logger.exception('❌ update_daily_average failed')

    # Cleanup
    try:
        all_docs = list(avg_collection.find().sort('date', -1))
        if len(all_docs) > 3:
            for doc in all_docs[3:]:
                avg_collection.delete_one({'_id': doc['_id']})
    except Exception:
        logger.exception('⚠ Failed to cleanup old averages')

# -----------------------------
# BACKGROUND MONITOR LOOP
# -----------------------------
def monitor_loop(notify_cooldown_seconds=NOTIFY_COOLDOWN_SECONDS, interval=MONITOR_LOOP_INTERVAL):
    global latest_alert
    last_sent_message = None
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

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/sensordata', methods=['POST'])
def receive_data():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    logger.info('📩 Received data: %s', data)
    
    mq2_ppm = safe_float(data.get('mq2_ppm', data.get('gas_value', 0.0)))
    temperature = safe_float(data.get('temperature', 0.0))
    humidity = safe_float(data.get('humidity', 0.0))
    voc_ppm = safe_float(data.get('voc_ppm', data.get('voc', 0.0)))
    h2s_ppm = safe_float(data.get('h2s_ppm', data.get('h2s', 0.0)))
    co_ppm = safe_float(data.get('co_ppm', data.get('co', 0.0)))
    air_quality_percent = safe_float(data.get('air_quality_percent', data.get('voc_percent', 0.0)))
    timestamp = datetime.now(timezone.utc)

    try:
        sensor_collection.update_one(
            {'_id': 1},
            {'$set': {
                'mq2_ppm': mq2_ppm, 'temperature': temperature, 'humidity': humidity,
                'voc_ppm': voc_ppm, 'h2s_ppm': h2s_ppm, 'co_ppm': co_ppm,
                'air_quality_percent': air_quality_percent,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %z')
            }},
            upsert=True
        )
    except Exception:
        logger.exception('❌ Failed to update latest sensor document')

    try:
        history_doc = {
            'mq2_ppm': mq2_ppm, 'temperature': temperature, 'humidity': humidity,
            'voc_ppm': voc_ppm, 'h2s_ppm': h2s_ppm, 'co_ppm': co_ppm,
            'air_quality_percent': air_quality_percent, 'timestamp': timestamp
        }
        history_collection.insert_one(history_doc)
    except Exception:
        logger.exception('❌ Failed to insert history doc')

    try:
        update_daily_average({
            'mq2_ppm': mq2_ppm, 'temperature': temperature, 'humidity': humidity,
            'voc_ppm': voc_ppm, 'h2s_ppm': h2s_ppm, 'co_ppm': co_ppm,
            'air_quality_percent': air_quality_percent
        })
    except Exception:
        logger.exception('❌ update_daily_average failed')

    return jsonify({'status': 'success', 'message': 'Data updated successfully'}), 200

@app.route('/api/viewdata', methods=['GET'])
def view_data():
    try:
        row = sensor_collection.find_one({'_id': 1}, {'_id': 0})
        return jsonify(row if row else {}), 200
    except Exception:
        logger.exception('❌ view_data failed')
        return jsonify({}), 500

@app.route('/api/alertstatus', methods=['GET'])
def get_alert():
    return jsonify(latest_alert), 200

@app.route('/monitor', methods=['GET'])
def monitor_endpoint():
    """Flutter app polling endpoint - evaluates alerts in real-time"""
    try:
        row = sensor_collection.find_one({'_id': 1})
        if not row:
            return jsonify({'alert': False, 'level': 'Good', 'message': 'No sensor data available yet'}), 200
        
        alert_flag, level, message = evaluate_alerts(row)
        logger.info('📱 /monitor checked: alert=%s, level=%s', alert_flag, level)
        
        return jsonify({'alert': alert_flag, 'level': level, 'message': message}), 200
    except Exception:
        logger.exception('❌ /monitor endpoint failed')
        return jsonify({'alert': False, 'level': 'Error', 'message': 'Failed to check sensor status'}), 500

@app.route('/monitor_cached', methods=['GET'])
def monitor_cached():
    """Returns cached alert status (faster)"""
    return jsonify(latest_alert), 200

@app.route('/api/viewdailyaverage', methods=['GET'])
def view_daily_average():
    try:
        rows_cursor = avg_collection.find({}, {'_id': 0}).sort('date', -1).limit(3)
        rows = list(rows_cursor)
        averages = []

        for row in rows:
            date_str = row.get('date')
            count = int(row.get('count', 0))
            if count > 0:
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
        return jsonify(averages), 200
    except Exception:
        logger.exception('❌ view_daily_average failed')
        return jsonify([]), 500

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
            ts = _ensure_tz_aware(doc.get('timestamp'))
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

@app.route('/get-data-weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', '16.4821')
    lon = request.args.get('lon', '80.6914')
    if not OPENWEATHER_API_KEY:
        return jsonify({'error': 'OpenWeather API key not configured'}), 400
    url = 'http://api.openweathermap.org/data/2.5/air_pollution'
    params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHER_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return jsonify(response.json()), 200
    except Exception:
        logger.exception('❌ get_weather failed')
        return jsonify({'error': 'Failed to fetch weather data'}), 500

@app.route('/api/gemini_suggestions', methods=['GET'])
def gemini_suggestions():
    try:
        model = request.args.get('model', GEMINI_MODEL)
        rows_cursor = avg_collection.find({}, {'_id': 0}).sort('date', -1).limit(3)
        rows = list(rows_cursor)
        if not rows:
            return jsonify({'status': 'error', 'message': 'no daily averages available'}), 404
        
        three_days = list(reversed(rows))
        gemini_resp = call_gemini_for_suggestions(three_days, model=model)
        
        human_text = None
        if isinstance(gemini_resp, dict):
            if 'text' in gemini_resp:
                human_text = gemini_resp.get('text')
        
        return jsonify({
            'status': 'success',
            'model_response': gemini_resp,
            'human_readable': human_text
        }), 200
    except Exception:
        logger.exception('❌ gemini_suggestions endpoint failed')
        return jsonify({'status': 'error', 'message': 'internal error'}), 500

@app.route('/api/recompute_daily_averages', methods=['POST'])
def recompute_daily_averages():
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
                avg_collection.delete_one({'date': date_str})
        
        return jsonify({'status': 'success', 'message': f'recomputed last {days} days'}), 200
    except Exception:
        logger.exception('❌ recompute_daily_averages failed')
        return jsonify({'status': 'error', 'message': 'recompute failed'}), 500

# =============================================================================
# ROUTINE API ENDPOINTS
# =============================================================================

@app.route('/api/routines/daily', methods=['GET'])
def get_daily_routines():
    """Get all configured daily routines"""
    return jsonify({'status': 'success', 'routines': DAILY_ROUTINES}), 200

@app.route('/api/routines/weekly', methods=['GET'])
def get_weekly_routines():
    """Get all configured weekly routines"""
    return jsonify({'status': 'success', 'routines': WEEKLY_ROUTINES}), 200

@app.route('/api/routines/next', methods=['GET'])
def get_next_routine():
    """Get the next scheduled routine"""
    try:
        now = datetime.now(TIMEZONE)
        current_time = now.time()
        
        next_routine = None
        min_diff = None
        
        for key, data in DAILY_ROUTINES.items():
            hour, minute = data['time']
            routine_time = dt_time(hour, minute)
            
            routine_datetime = datetime.combine(now.date(), routine_time)
            routine_datetime = TIMEZONE.localize(routine_datetime)
            
            if routine_time < current_time:
                routine_datetime = routine_datetime + timedelta(days=1)
            
            diff = (routine_datetime - now).total_seconds()
            
            if min_diff is None or diff < min_diff:
                min_diff = diff
                next_routine = {
                    'key': key,
                    'data': data,
                    'time_until': int(diff / 60),
                    'scheduled_time': f"{hour:02d}:{minute:02d}"
                }
        
        return jsonify({'status': 'success', 'next_routine': next_routine}), 200
    except Exception:
        logger.exception('❌ Failed to get next routine')
        return jsonify({'status': 'error'}), 500

@app.route('/api/routines/trigger', methods=['POST'])
def trigger_routine_manually():
    """Manually trigger a specific routine"""
    try:
        data = request.get_json() or {}
        routine_type = data.get('type', 'daily')
        routine_key = data.get('key')
        
        if not routine_key:
            return jsonify({'status': 'error', 'message': 'routine key required'}), 400
        
        if routine_type == 'daily':
            if routine_key not in DAILY_ROUTINES:
                return jsonify({'status': 'error', 'message': 'invalid routine key'}), 400
            send_routine_notification(routine_key, DAILY_ROUTINES[routine_key])
        elif routine_type == 'weekly':
            if routine_key not in WEEKLY_ROUTINES:
                return jsonify({'status': 'error', 'message': 'invalid routine key'}), 400
            send_routine_notification(routine_key, WEEKLY_ROUTINES[routine_key])
        else:
            return jsonify({'status': 'error', 'message': 'invalid routine type'}), 400
        
        return jsonify({'status': 'success', 'message': f'Triggered {routine_key}'}), 200
    except Exception:
        logger.exception('❌ Failed to trigger routine')
        return jsonify({'status': 'error'}), 500

@app.route('/api/routines/history', methods=['GET'])
def get_routine_history():
    """Get history of scheduled routines"""
    try:
        jobs = []
        for job in scheduler.get_jobs():
            if job.id and job.id.startswith('routine_'):
                jobs.append({
                    'id': job.id,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'name': job.name
                })
        return jsonify({'status': 'success', 'scheduled_jobs': jobs}), 200
    except Exception:
        logger.exception('❌ Failed to get routine history')
        return jsonify({'status': 'error'}), 500

@app.route('/api/send_test_notification', methods=['POST'])
def send_test_notification():
    """Send a test notification"""
    try:
        data = request.get_json() or {}
        title = data.get('title', 'Test Notification')
        body = data.get('body', 'This is a test notification from your IoT system.')
        level = data.get('level', 'Good')
        
        send_fcm_notification(title, body, level)
        return jsonify({'status': 'success', 'message': 'Test notification sent'}), 200
    except Exception:
        logger.exception('❌ Failed to send test notification')
        return jsonify({'status': 'error', 'message': 'Failed to send notification'}), 500

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    initialize_db()

    # Start monitor thread
    if not monitor_thread_started.is_set():
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        monitor_thread_started.set()
        logger.info('✅ Monitor thread started')

    # Start scheduler and schedule routines
    try:
        schedule_daily_routines()
        scheduler.start()
        logger.info('✅ Scheduler started with daily routines')
    except Exception:
        logger.exception("⚠️ Failed to start scheduler")

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True)