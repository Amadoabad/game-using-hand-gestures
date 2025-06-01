from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample valid input: a list of 63 float values
valid_input = {
    "landmarks": [0.0 for _ in range(63)]  
}

# Sample invalid input: wrong size
invalid_input_short = {
    "landmarks": [0.0 for _ in range(50)]  # Too short
}

def test_predict_valid_input():
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert "gesture" in response.json()

def test_predict_invalid_input_length():
    response = client.post("/predict", json=invalid_input_short)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_missing_key():
    response = client.post("/predict", json={"wrong_key": valid_input["landmarks"]})
    assert response.status_code == 422

def test_predict_wrong_type():
    response = client.post("/predict", json={"landmarks": "not a list"})
    assert response.status_code == 422
