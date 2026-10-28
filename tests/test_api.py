from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    payload = {"Pclass": 1, "Sex": 1, "Age": 29, "sibsp": 0,
               "Parch": 0, "Fare": 80, "Embarked": 2}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["survived"] in (0, 1)