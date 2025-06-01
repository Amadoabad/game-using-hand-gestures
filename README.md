# game-using-hand-gestures

## Overview
This repository provides an API for real-time hand gesture classification, designed to be integrated into interactive applications such as games. The core of the system is a machine learning model served using FastAPI, enabling efficient and scalable inference.

## Features
- **Model Serving:** The hand gesture classification model is served via a FastAPI application for low-latency predictions.
- **Unit Testing:** Initial unit tests are provided (see `tests/test_api.py`) to ensure API reliability and correct error handling.
- **Containerization:** The application is containerized using Docker for consistent deployment across environments.
- **Monitoring:** Three key metrics are monitored to ensure model and server health:
  - **Model Metric:** Histogram of prediction confidence probabilities. This helps track the model's certainty in its predictions, allowing us to detect potential issues such as overconfidence or uncertainty in new data.
  - **Data Metric:** Data drift monitoring by comparing the mean of incoming data to the training data mean. This helps identify when the input data distribution changes, which could indicate that the model may need retraining.
  - **Server Metric:** Request latency logging. Monitoring server response times helps ensure the API remains performant and helps identify bottlenecks or infrastructure issues.

## Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd game-using-hand-gestures
   ```
2. **Install dependencies:**
   - For production (minimal):
     ```bash
     pip install -r requirements.txt
     ```
   - For development (includes testing tools):
     ```bash
     pip install -r requirements-dev.txt
     ```
3. **Run the API:**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Run tests:**
   ```bash
   pytest tests/
   ```
5. **Build and run with Docker:**
   ```bash
   docker build -t hand-gesture-api .
   docker run -p 8000:8000 hand-gesture-api
   ```

## Requirements Files
- `requirements.txt`: Contains only the dependencies needed to run the application and is used in the Docker container.
- `requirements-dev.txt`: Contains additional packages for development (e.g., `pytest`, `fastapi[test]`, etc.). Both files are maintained in the development branch to support both production and development workflows.

## API Endpoints
- `GET /` — Health check and welcome message.
- `POST /predict` — Accepts a JSON payload with hand landmarks and returns the predicted gesture and confidence.

## Testing
Unit tests are located in `tests/test_api.py` and cover scenarios such as valid/invalid input, missing keys, and type errors to ensure robust API behavior.
