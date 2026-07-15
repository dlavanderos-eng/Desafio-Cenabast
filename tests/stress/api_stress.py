from locust import HttpUser, task


class StressUser(HttpUser):

    @task
    def predict_metformina(self):
        self.client.post(
            "/predict",
            json={
                "products": [
                    {
                        "gtin": "7804587213648",
                        "fecha": "2024-06-15"
                    }
                ]
            }
        )

    @task
    def predict_losartan(self):
        self.client.post(
            "/predict",
            json={
                "products": [
                    {
                        "gtin": "7809235178465",
                        "fecha": "2024-06-15"
                    }
                ]
            }
        )

    @task
    def predict_amlodipino(self):
        self.client.post(
            "/predict",
            json={
                "products": [
                    {
                        "gtin": "7801748632954",
                        "fecha": "2024-06-15"
                    }
                ]
            }
        )