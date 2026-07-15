import unittest

from fastapi.testclient import TestClient
from challenge import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_should_get_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

    def test_should_get_predict(self):
        data = {
            "products": [
                {
                    "gtin": "7804587213648",
                    "fecha": "2024-06-15"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predict", response.json())

    def test_should_failed_unknown_product(self):
        data = {
            "products": [
                {
                    "gtin": "9999999999999",
                    "fecha": "2024-06-15"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_invalid_date(self):
        data = {
            "products": [
                {
                    "gtin": "7804587213648",
                    "fecha": "invalid-date"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)