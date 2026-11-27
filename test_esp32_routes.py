import unittest
import json
import os
import sys

# Mock MongoDB for offline/sandboxed test execution
class MockCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction=1):
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

class MockCollection:
    def __init__(self, name):
        self.name = name
        self.data = []

    def create_index(self, *args, **kwargs):
        pass

    def count_documents(self, filter_query):
        return len(self.data)

    def insert_one(self, doc):
        d = dict(doc)
        if '_id' not in d:
            d['_id'] = len(self.data) + 1
        self.data.append(d)
        return type('InsertResult', (), {'inserted_id': d['_id']})()

    def update_one(self, filter_query, update_query, upsert=False):
        set_vals = update_query.get('$set', {})
        for doc in self.data:
            match = True
            for k, v in filter_query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                doc.update(set_vals)
                return type('UpdateResult', (), {'matched_count': 1, 'modified_count': 1})()
        if upsert:
            new_doc = dict(filter_query)
            new_doc.update(set_vals)
            self.data.append(new_doc)
            return type('UpdateResult', (), {'matched_count': 0, 'upserted_id': new_doc.get('_id')})()
        return type('UpdateResult', (), {'matched_count': 0, 'modified_count': 0})()

    def find_one(self, filter_query, projection=None):
        for doc in reversed(self.data):
            match = True
            for k, v in filter_query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                res = dict(doc)
                if projection and projection.get('_id') == 0:
                    res.pop('_id', None)
                return res
        return None

    def find(self, filter_query=None, projection=None):
        out = []
        for doc in reversed(self.data):
            res = dict(doc)
            if projection and projection.get('_id') == 0:
                res.pop('_id', None)
            out.append(res)
        return MockCursor(out)

class MockDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]

    def command(self, cmd):
        return {'ok': 1}

class MockMongoClient:
    def __init__(self, *args, **kwargs):
        self.db = MockDatabase()
        self.admin = self.db

    def get_database(self, *args, **kwargs):
        return self.db

# Patch pymongo.MongoClient before importing app
import pymongo
pymongo.MongoClient = MockMongoClient

# Add backend root to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app

class TestEsp32Routes(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_01_post_esp32_sensordata(self):
        payload = {
            "device_id": "ESP32_OMX44",
            "timestamp": "2026-11-27T09:00:00Z",
            "mq2_ppm": 18.75,
            "temperature": 28.40,
            "humidity": 55.20,
            "air_quality_percent": None,
            "mq2_raw": 420,
            "led_status": "OFF",
            "buzzer_status": "OFF",
            "sensor_fault": False
        }
        response = self.client.post(
            '/api/esp32/sensordata',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data['data']['device_id'], 'ESP32_OMX44')
        self.assertEqual(data['data']['mq2_raw'], 420)
        self.assertEqual(data['data']['mq2_ppm'], 18.75)
        self.assertEqual(data['data']['temperature'], 28.40)
        self.assertEqual(data['data']['led_status'], 'OFF')
        self.assertEqual(data['data']['buzzer_status'], 'OFF')
        print("✅ test_01_post_esp32_sensordata passed!")

    def test_02_get_esp32_latest(self):
        response = self.client.get('/api/esp32/latest')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('data', data)
        self.assertEqual(data['data']['device_id'], 'ESP32_OMX44')
        self.assertEqual(data['data']['mq2_raw'], 420)
        self.assertEqual(data['data']['led_status'], 'OFF')
        print("✅ test_02_get_esp32_latest passed!")

    def test_03_get_esp32_data_alias(self):
        response = self.client.get('/api/esp32/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data['data']['device_id'], 'ESP32_OMX44')
        print("✅ test_03_get_esp32_data_alias passed!")

    def test_04_danger_actuators_on(self):
        # Payload when gas or temp exceeds threshold on ESP32
        payload = {
            "device_id": "ESP32_OMX44",
            "timestamp": "2026-11-27T09:05:00Z",
            "mq2_ppm": 125.40,
            "temperature": 42.10,
            "humidity": 75.00,
            "mq2_raw": 1350,
            "led_status": "ON",
            "buzzer_status": "ON",
            "sensor_fault": False
        }
        response = self.client.post(
            '/api/esp32/sensordata',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify get latest reflects actuators ON
        get_resp = self.client.get('/api/esp32/latest')
        self.assertEqual(get_resp.status_code, 200)
        latest_data = json.loads(get_resp.data)['data']
        self.assertEqual(latest_data['led_status'], 'ON')
        self.assertEqual(latest_data['buzzer_status'], 'ON')
        self.assertEqual(latest_data['mq2_raw'], 1350)
        self.assertEqual(latest_data['temperature'], 42.10)
        print("✅ test_04_danger_actuators_on passed!")

    def test_05_legacy_route_esp32_compatibility(self):
        # When ESP32 posts to /api/sensordata with led_status and buzzer_status
        payload = {
            "device_id": "ESP32_OMX44",
            "timestamp": "2026-11-27T09:10:00Z",
            "mq2_ppm": 22.10,
            "temperature": 26.50,
            "humidity": 50.00,
            "mq2_raw": 390,
            "led_status": "OFF",
            "buzzer_status": "OFF",
            "sensor_fault": False
        }
        response = self.client.post(
            '/api/sensordata',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify esp32 endpoint captured it
        get_resp = self.client.get('/api/esp32/latest')
        self.assertEqual(get_resp.status_code, 200)
        latest_data = json.loads(get_resp.data)['data']
        self.assertEqual(latest_data['mq2_raw'], 390)
        self.assertEqual(latest_data['led_status'], 'OFF')
        print("✅ test_05_legacy_route_esp32_compatibility passed!")

    def test_06_get_esp32_history(self):
        response = self.client.get('/api/esp32/history?limit=10')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertGreater(data.get('count', 0), 0)
        self.assertIsInstance(data.get('history'), list)
        print("✅ test_06_get_esp32_history passed!")

if __name__ == '__main__':
    unittest.main()
