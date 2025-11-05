from flask import Flask, request, jsonify, Response
from datetime import datetime, timedelta
import threading
import time
import os
import requests
import firebase_admin
from firebase_admin import credentials, messaging
from pymongo import MongoClient, ASCENDING

# -----------------------------
# FLASK APP INITIALIZATION
# -----------------------------
app = Flask(__name__)

# -----------------------------
# MONGODB CONNECTION (ATLAS)
# -----------------------------
MONGO_URI = "mongodb+srv://panduranga1797:17971797@cluster0.cqvn1hp.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)  # consider adding serverSelectionTimeoutMS=5000 in production
db = client['air_quality_db']          # Database name
sensor_collection = db['sensor_data']  # For latest readings
avg_collection = db['daily_average']   # For 3-day averages
history_collection = db['sensor_history']  # NEW: time-series history

# Ensure index for faster queries on timestamp (create once)
history_collection.create_index([("timestamp", ASCENDING)])
# Optional: index by any other field you query frequently, e.g. {"gas_value":1}

# -----------------------------
# FIREBASE INITIALIZATION
# -----------------------------
cred = credentials.Certificate("serviceAccountkey.json")
firebase_admin.initialize_app(cred)

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
latest_alert = {"alert": False, "message": "All clear 👍"}
priority_order = ["lpg", "co2", "h2s", "voc", "temperature", "humidity"]

# Replace with your actual device token
FCM_DEVICE_TOKEN = "d9opAp5QRkG0gFcOD_kbck:APA91bEUXkhYE1_sxIGH_uOeuw5bMasaGwTUUxrCv19GOTCp427ByQLtsSNRVPfiosnxaMbIAtA76ZVoel0fla7glJqnXt295gjW4W7KbdFXrv0j_ADfpdY"

# -----------------------------
# FIREBASE NOTIFICATION FUNCTION
# -----------------------------
def send_fcm_notification(title, body, level):
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=FCM_DEVICE_TOKEN,
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    sound="default",
                    color="#FF0000" if level == "Extreme" else "#FFA500",
                ),
            ),
            data={"click_action": "FLUTTER_NOTIFICATION_CLICK", "alert_level": level}
        )
        response = messaging.send(message)
        print(f"✅ FCM Notification sent successfully! ID: {response}")
    except Exception as e:
        print("❌ Failed to send FCM Notification:", e)

# -----------------------------
# INITIALIZE DATABASE DOCUMENTS
# -----------------------------
def initialize_db():
    if sensor_collection.count_documents({}) == 0:
        sensor_collection.insert_one({
            "_id": 1,
            "gas_value": 0, "temperature": 0, "humidity": 0,
            "voc": 0, "h2s": 0, "co2": 0, "voc_percent": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print("✅ MongoDB ready with seven-sensor structure!")

# -----------------------------
# API: Receive Sensor Data
# -----------------------------
@app.route('/api/sensordata', methods=['POST'])
def receive_data():
    data = request.get_json()
    print("📩 Received Data:", data)

    # Extract with defaults
    gas_value = float(data.get('gas_value', 0))
    temperature = float(data.get('temperature', 0))
    humidity = float(data.get('humidity', 0))
    voc = float(data.get('voc', 0))
    h2s = float(data.get('h2s', 0))
    co2 = float(data.get('co2', 0))
    voc_percent = float(data.get('voc_percent', 0))
    timestamp = datetime.utcnow()  # store as datetime for better queries

    # -----------------------------
    # Update Latest Sensor Data (single doc)
    # -----------------------------
    sensor_collection.update_one(
        {"_id": 1},
        {"$set": {
            "gas_value": gas_value, "temperature": temperature, "humidity": humidity,
            "voc": voc, "h2s": h2s, "co2": co2, "voc_percent": voc_percent,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }},
        upsert=True
    )

    # -----------------------------
    # INSERT into HISTORY collection (every reading)
    # -----------------------------
    try:
        history_doc = {
            "gas_value": gas_value,
            "temperature": temperature,
            "humidity": humidity,
            "voc": voc,
            "h2s": h2s,
            "co2": co2,
            "voc_percent": voc_percent,
            "timestamp": timestamp  # datetime object
        }
        history_collection.insert_one(history_doc)
    except Exception as e:
        print("❌ Failed to insert into history_collection:", e)

    # -----------------------------
    # Update 3-Day Daily Average Table
    # -----------------------------
    update_daily_average({
        "gas_value": gas_value,
        "temperature": temperature,
        "humidity": humidity,
        "voc": voc,
        "h2s": h2s,
        "co2": co2,
        "voc_percent": voc_percent
    })

    print(f"✅ Updated (latest + history): MQ2={gas_value}, Temp={temperature}, Hum={humidity}, VOC={voc}, H2S={h2s}, CO2={co2}")
    return jsonify({"status": "success", "message": "Data updated successfully"}), 200

# -----------------------------
# API: Get Latest Sensor Data
# -----------------------------
@app.route('/api/viewdata', methods=['GET'])
def view_data():
    row = sensor_collection.find_one({"_id": 1}, {"_id": 0})
    return jsonify(row if row else {})

# -----------------------------
# API: Get Latest Alert
# -----------------------------
@app.route('/api/alertstatus', methods=['GET'])
def get_alert():
    return jsonify(latest_alert)

# -----------------------------
# PRIORITY ALERT FUNCTION
# -----------------------------
def evaluate_alerts(row):
    alerts = {}
    gas_value = row.get('gas_value', 0)
    temperature = row.get('temperature', 0)
    humidity = row.get('humidity', 0)
    voc = row.get('voc', 0)
    h2s = row.get('h2s', 0)
    co2 = row.get('co2', 0)

    # LPG
    if 1000 < gas_value <= 1500:
        alerts["lpg"] = ("Moderate", "⚠️ MODERATE: LPG presence detected, ensure ventilation.")
    elif gas_value > 1500:
        alerts["lpg"] = ("Extreme", "🚨 EXTREME: LPG leak detected, act immediately!")

    # CO2
    if 1000 < co2 <= 1500:
        alerts["co2"] = ("Moderate", "⚠️ MODERATE: CO₂ rising, open windows.")
    elif co2 > 1500:
        alerts["co2"] = ("Extreme", "🚨 EXTREME: Dangerous CO₂ level! Ventilate now!")

    # H2S
    if 1800 < h2s <= 2400:
        alerts["h2s"] = ("Moderate", "⚠️ MODERATE: Low H₂S detected.")
    elif h2s > 2400:
        alerts["h2s"] = ("Extreme", "🚨 EXTREME: High H₂S detected! Possible leak!")

    # VOC
    if 2500 < voc <= 2900:
        alerts["voc"] = ("Moderate", "⚠️ MODERATE: VOCs increasing, ventilate area.")
    elif voc > 2900:
        alerts["voc"] = ("Extreme", "🚨 EXTREME: VOC level critical!")

    # Temperature
    if 32 < temperature <= 35:
        alerts["temperature"] = ("Moderate", "🌡️ MODERATE: Slightly high room temperature.")
    elif temperature > 35:
        alerts["temperature"] = ("Extreme", "🔥 EXTREME: Temperature too high!")

    # Humidity
    if (70 < humidity <= 80) or (20 < humidity <= 30):
        alerts["humidity"] = ("Moderate", "💧 MODERATE: Humidity slightly off.")
    elif humidity > 90 or humidity < 20:
        alerts["humidity"] = ("Extreme", "💧 EXTREME: Humidity critical!")

    for key in priority_order:
        if key in alerts:
            return True, alerts[key][0], alerts[key][1]
    return False, "Good", "✅ Air Quality: GOOD"

# -----------------------------
# BACKGROUND MONITOR LOOP
# -----------------------------
def monitor_loop(notify_cooldown_seconds=60):
    """
    Monitor latest document and send Firebase notifications when alert changes.
    notify_cooldown_seconds: minimum seconds between two FCM sends for the same message.
    """
    global latest_alert
    last_sent_message = None
    last_sent_time = datetime.min

    while True:
        try:
            row = sensor_collection.find_one({"_id": 1})
        except Exception as e:
            # Don't let DB/network errors crash the thread
            print("⚠️ monitor_loop DB error (will retry):", e)
            time.sleep(5)
            continue

        if row:
            alert_flag, level, msg = evaluate_alerts(row)
            latest_alert = {"alert": alert_flag, "level": level, "message": msg}
            print(f"🔔 [Monitor] {msg}")

            # Only attempt notification if alert is True
            if alert_flag:
                now = datetime.utcnow()
                # send if message changed OR cooldown expired
                cooldown_ok = (now - last_sent_time) >= timedelta(seconds=notify_cooldown_seconds)
                if (msg != last_sent_message) or cooldown_ok:
                    try:
                        send_fcm_notification("Air Quality Alert", msg, level)
                        last_sent_message = msg
                        last_sent_time = now
                        print(f"✅ Notification sent (message='{msg[:60]}') at {now}")
                    except Exception as e:
                        print("❌ Failed to send FCM inside monitor_loop:", e)
        else:
            # No row found: keep trying
            print("ℹ️ monitor_loop: no latest sensor document found; retrying...")

        time.sleep(5)

# -----------------------------
# WEATHER API
# -----------------------------
@app.route('/get-data-weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', '16.4821')
    lon = request.args.get('lon', '80.6914')
    api_key = "c6b3ba16b0dd5b70beed998a1ec6b3c8"
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    response = requests.get(url, params=params)
    return jsonify(response.json()) if response.status_code == 200 else jsonify({"error": "Failed to fetch data"}), response.status_code

# -----------------------------
# UPDATE DAILY AVERAGE VALUES
# -----------------------------
def update_daily_average(data):
    today = datetime.now().strftime("%Y-%m-%d")
    existing = avg_collection.find_one({"date": today})

    if existing:
        avg_collection.update_one(
            {"date": today},
            {"$inc": {
                "gas_value_sum": data['gas_value'],
                "temperature_sum": data['temperature'],
                "humidity_sum": data['humidity'],
                "voc_sum": data['voc'],
                "h2s_sum": data['h2s'],
                "co2_sum": data['co2'],
                "voc_percent_sum": data['voc_percent'],
                "count": 1
            }}
        )
    else:
        avg_collection.insert_one({
            "date": today,
            "gas_value_sum": data['gas_value'],
            "temperature_sum": data['temperature'],
            "humidity_sum": data['humidity'],
            "voc_sum": data['voc'],
            "h2s_sum": data['h2s'],
            "co2_sum": data['co2'],
            "voc_percent_sum": data['voc_percent'],
            "count": 1
        })

    # Keep only latest 3 days
    all_docs = list(avg_collection.find().sort("date", -1))
    if len(all_docs) > 3:
        for doc in all_docs[3:]:
            avg_collection.delete_one({"_id": doc["_id"]})

# -----------------------------
# API: Get Last 3 Days Average Data
# -----------------------------
@app.route('/api/viewdailyaverage', methods=['GET'])
def view_daily_average():
    rows = list(avg_collection.find({}, {"_id": 0}).sort("date", -1))
    averages = []
    for row in rows:
        count = row.get('count', 1)
        averages.append({
            "date": row['date'],
            "avg_gas_value": round(row['gas_value_sum'] / count, 2),
            "avg_temperature": round(row['temperature_sum'] / count, 2),
            "avg_humidity": round(row['humidity_sum'] / count, 2),
            "avg_voc": round(row['voc_sum'] / count, 2),
            "avg_h2s": round(row['h2s_sum'] / count, 2),
            "avg_co2": round(row['co2_sum'] / count, 2),
            "avg_voc_percent": round(row['voc_percent_sum'] / count, 2)
        })
    return jsonify(averages)

# -----------------------------
# NEW: View history (last N records)
# -----------------------------
@app.route('/api/viewhistory', methods=['GET'])
def view_history():
    try:
        limit = int(request.args.get('limit', 100))
    except:
        limit = 100
    cursor = history_collection.find().sort("timestamp", -1).limit(limit)
    data = []
    for doc in cursor:
        data.append({
            "gas_value": doc.get("gas_value", 0),
            "temperature": doc.get("temperature", 0),
            "humidity": doc.get("humidity", 0),
            "voc": doc.get("voc", 0),
            "h2s": doc.get("h2s", 0),
            "co2": doc.get("co2", 0),
            "voc_percent": doc.get("voc_percent", 0),
            "timestamp": doc.get("timestamp").strftime("%Y-%m-%d %H:%M:%S") if doc.get("timestamp") else ""
        })
    return jsonify(data)

# -----------------------------
# NEW: Export CSV for training
# Example: /api/exportcsv?days=3&limit=10000
# -----------------------------
@app.route('/api/exportcsv', methods=['GET'])
def export_csv():
    try:
        days = int(request.args.get('days', 3))
    except:
        days = 3
    try:
        limit = int(request.args.get('limit', 10000))
    except:
        limit = 10000

    start_time = datetime.utcnow() - timedelta(days=days)
    cursor = history_collection.find({"timestamp": {"$gte": start_time}}).sort("timestamp", ASCENDING).limit(limit)

    # Build CSV in memory
    from io import StringIO
    import csv
    output = StringIO()
    writer = csv.writer(output)
    # header (choose fields you need for training)
    writer.writerow(["timestamp", "gas_value", "temperature", "humidity", "voc", "h2s", "co2", "voc_percent"])

    rows = 0
    for doc in cursor:
        ts = doc.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        writer.writerow([
            ts_str,
            doc.get("gas_value", ""),
            doc.get("temperature", ""),
            doc.get("humidity", ""),
            doc.get("voc", ""),
            doc.get("h2s", ""),
            doc.get("co2", ""),
            doc.get("voc_percent", "")
        ])
        rows += 1

    output.seek(0)
    csv_data = output.getvalue()
    output.close()

    filename = f"sensor_history_{days}d_{rows}rows.csv"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }
    return Response(csv_data, mimetype="text/csv", headers=headers)

# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == '__main__':
    # Initialize DB etc.
    initialize_db()

    # When Flask debug=True, the reloader runs the script twice (parent + child).
    # The actual app runs in the child process with WERKZEUG_RUN_MAIN == "true".
    # We start the monitor thread only in that child (or when reloader is disabled).
    should_start_monitor = True
    # If the dev server reloader is enabled, only start monitor in child process
    if os.environ.get("WERKZEUG_RUN_MAIN") is not None:
        # child process when reloader is active
        should_start_monitor = True
    else:
        # when not using reloader, still start monitor
        should_start_monitor = True

    # Alternative: to avoid any reloader-related doubled processes, you can set use_reloader=False below.
    if should_start_monitor:
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("✅ monitor thread started.")

    # Run Flask. If you still get WinError 10038, try turning off reloader:
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    app.run(host='0.0.0.0', port=5000, debug=True)