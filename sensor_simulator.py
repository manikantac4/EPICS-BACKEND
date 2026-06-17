import requests
import random
import time
from datetime import datetime, timezone

API_URL = "https://epics-backend.onrender.com/api/sensordata"

id_counter = 1

def get_time_multipliers():
    hour = datetime.now().hour + datetime.now().minute / 60.0

    # Dead hours 00–06
    if hour < 6:
        return {
            "mq2": (30, 60), "temp": (22, 26), "hum": (50, 65),
            "voc": (80, 120), "h2s": (0.05, 0.3), "co": (0.2, 0.8), "aq": (92, 99)
        }
    # Morning startup 06–08
    elif hour < 8:
        return {
            "mq2": (55, 90), "temp": (24, 28), "hum": (55, 70),
            "voc": (110, 160), "h2s": (0.1, 0.5), "co": (0.5, 1.5), "aq": (85, 94)
        }
    # Breakfast cooking 08–09:30
    elif hour < 9.5:
        return {
            "mq2": (120, 220), "temp": (27, 33), "hum": (60, 78),
            "voc": (200, 350), "h2s": (0.4, 1.2), "co": (2.0, 5.5), "aq": (65, 80)
        }
    # Mid-morning 09:30–12
    elif hour < 12:
        return {
            "mq2": (40, 80), "temp": (26, 30), "hum": (45, 65),
            "voc": (90, 150), "h2s": (0.1, 0.4), "co": (0.3, 1.0), "aq": (88, 97)
        }
    # Lunch cooking 12–13:30
    elif hour < 13.5:
        return {
            "mq2": (150, 280), "temp": (29, 36), "hum": (65, 82),
            "voc": (280, 450), "h2s": (0.6, 1.8), "co": (3.0, 7.0), "aq": (55, 72)
        }
    # Afternoon heat 13:30–16
    elif hour < 16:
        return {
            "mq2": (45, 85), "temp": (31, 37), "hum": (55, 72),
            "voc": (100, 170), "h2s": (0.1, 0.5), "co": (0.4, 1.2), "aq": (82, 93)
        }
    # Evening ventilation 16–18
    elif hour < 18:
        return {
            "mq2": (30, 65), "temp": (28, 33), "hum": (48, 65),
            "voc": (80, 130), "h2s": (0.05, 0.3), "co": (0.2, 0.8), "aq": (90, 98)
        }
    # Dinner cooking 18–20 (peak)
    elif hour < 20:
        return {
            "mq2": (200, 380), "temp": (32, 42), "hum": (70, 90),
            "voc": (380, 600), "h2s": (1.0, 3.0), "co": (5.0, 12.0), "aq": (40, 62)
        }
    # Post-dinner 20–22
    elif hour < 22:
        return {
            "mq2": (60, 110), "temp": (28, 33), "hum": (60, 78),
            "voc": (140, 220), "h2s": (0.2, 0.7), "co": (0.8, 2.5), "aq": (75, 88)
        }
    # Late night 22–00
    else:
        return {
            "mq2": (35, 70), "temp": (25, 29), "hum": (52, 68),
            "voc": (90, 140), "h2s": (0.05, 0.35), "co": (0.3, 1.0), "aq": (88, 96)
        }

def apply_noise(val, pct=0.04):
    return val * (1 + random.gauss(0, pct))

def generate():
    global id_counter
    ranges = get_time_multipliers()
    roll = random.random()

    # 2% chance: gas leak event
    if roll < 0.02:
        r = ranges
        data = {
            "values_id": id_counter,
            "mq2_ppm":              round(apply_noise(random.uniform(800, 2000)), 2),
            "temperature":          round(apply_noise(random.uniform(r["temp"][0], r["temp"][1]), 0.02), 1),
            "humidity":             round(apply_noise(random.uniform(r["hum"][0], r["hum"][1]), 0.02), 1),
            "voc_ppm":              round(apply_noise(random.uniform(700, 1500)), 2),
            "h2s_ppm":              round(apply_noise(random.uniform(5.0, 15.0)), 3),
            "co_ppm":               round(apply_noise(random.uniform(20.0, 50.0)), 2),
            "air_quality_percent":  round(max(5, apply_noise(random.uniform(5, 25), 0.05)), 1),
            "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        }
    # 3% chance: fire risk (high temp + CO spike)
    elif roll < 0.05:
        r = ranges
        data = {
            "values_id": id_counter,
            "mq2_ppm":              round(apply_noise(random.uniform(400, 900)), 2),
            "temperature":          round(apply_noise(random.uniform(42, 65), 0.02), 1),
            "humidity":             round(apply_noise(random.uniform(10, 28), 0.02), 1),
            "voc_ppm":              round(apply_noise(random.uniform(500, 1000)), 2),
            "h2s_ppm":              round(apply_noise(random.uniform(2.0, 6.0)), 3),
            "co_ppm":               round(apply_noise(random.uniform(30.0, 80.0)), 2),
            "air_quality_percent":  round(max(3, apply_noise(random.uniform(3, 18), 0.05)), 1),
            "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        }
    # 3% chance: poor ventilation (CO₂/VOC buildup)
    elif roll < 0.08:
        r = ranges
        data = {
            "values_id": id_counter,
            "mq2_ppm":              round(apply_noise(random.uniform(r["mq2"][0], r["mq2"][1])), 2),
            "temperature":          round(apply_noise(random.uniform(r["temp"][0], r["temp"][1]), 0.02), 1),
            "humidity":             round(apply_noise(random.uniform(85, 99), 0.02), 1),
            "voc_ppm":              round(apply_noise(random.uniform(600, 1200)), 2),
            "h2s_ppm":              round(apply_noise(random.uniform(1.5, 4.0)), 3),
            "co_ppm":               round(apply_noise(random.uniform(8.0, 20.0)), 2),
            "air_quality_percent":  round(max(20, apply_noise(random.uniform(20, 45), 0.05)), 1),
            "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        }
    # 2% chance: sensor glitch (one channel maxes, one drops to 0)
    elif roll < 0.10:
        r = ranges
        mq2  = apply_noise(random.uniform(r["mq2"][0], r["mq2"][1]))
        voc  = apply_noise(random.uniform(r["voc"][0],  r["voc"][1]))
        h2s  = apply_noise(random.uniform(r["h2s"][0],  r["h2s"][1]))
        co   = apply_noise(random.uniform(r["co"][0],   r["co"][1]))
        # pick a glitch victim
        glitch = random.choice(["mq2", "voc", "h2s", "co"])
        if glitch == "mq2":  mq2  = random.choice([0.0, random.uniform(4500, 5000)])
        elif glitch == "voc": voc  = random.choice([0.0, random.uniform(4500, 5000)])
        elif glitch == "h2s": h2s  = random.choice([0.0, random.uniform(48, 60)])
        elif glitch == "co":  co   = random.choice([0.0, random.uniform(90, 120)])
        data = {
            "values_id": id_counter,
            "mq2_ppm":              round(mq2, 2),
            "temperature":          round(apply_noise(random.uniform(r["temp"][0], r["temp"][1]), 0.02), 1),
            "humidity":             round(apply_noise(random.uniform(r["hum"][0],  r["hum"][1]),  0.02), 1),
            "voc_ppm":              round(voc, 2),
            "h2s_ppm":              round(h2s, 3),
            "co_ppm":               round(co, 2),
            "air_quality_percent":  round(apply_noise(random.uniform(r["aq"][0], r["aq"][1]), 0.03), 1),
            "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        }
    # 82% chance: normal time-based reading
    else:
        r = ranges
        data = {
            "values_id": id_counter,
            "mq2_ppm":              round(apply_noise(random.uniform(r["mq2"][0], r["mq2"][1])), 2),
            "temperature":          round(apply_noise(random.uniform(r["temp"][0], r["temp"][1]), 0.02), 1),
            "humidity":             round(apply_noise(random.uniform(r["hum"][0],  r["hum"][1]),  0.02), 1),
            "voc_ppm":              round(apply_noise(random.uniform(r["voc"][0],  r["voc"][1])), 2),
            "h2s_ppm":              round(apply_noise(random.uniform(r["h2s"][0],  r["h2s"][1])), 3),
            "co_ppm":               round(apply_noise(random.uniform(r["co"][0],   r["co"][1])), 2),
            "air_quality_percent":  round(min(100, apply_noise(random.uniform(r["aq"][0], r["aq"][1]), 0.03)), 1),
            "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
        }

    # Hard clamp — sensors have physical limits
    data["mq2_ppm"]             = round(max(0,   min(5000,  data["mq2_ppm"])), 2)
    data["temperature"]         = round(max(-10,  min(80,    data["temperature"])), 1)
    data["humidity"]            = round(max(0,    min(100,   data["humidity"])), 1)
    data["voc_ppm"]             = round(max(0,    min(5000,  data["voc_ppm"])), 2)
    data["h2s_ppm"]             = round(max(0,    min(100,   data["h2s_ppm"])), 3)
    data["co_ppm"]              = round(max(0,    min(150,   data["co_ppm"])), 2)
    data["air_quality_percent"] = round(max(0,    min(100,   data["air_quality_percent"])), 1)

    id_counter += 1
    return data

def simulator_loop():
    while True:
        data = generate()

        try:
            response = requests.post(API_URL, json=data, timeout=30)

            print("=" * 50)
            print("Sent:", data)
            print("Status:", response.status_code)

            if response.status_code != 200:
                print("Response:", response.text)

        except Exception as e:
            print("ERROR:", e)

        time.sleep(30)


if __name__ == "__main__":
    simulator_loop()