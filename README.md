# game-using-hand-gestures

## Overview
This repository provides an API for real-time hand gesture classification, designed to be integrated into interactive applications such as games. The core of the system is a machine learning model served using FastAPI, enabling efficient and scalable inference.

## Features
- **Model Serving:** The hand gesture classification model is served via a FastAPI application for low-latency predictions.
- **Containerization:** The application is containerized using Docker for consistent deployment across environments.
- **Docker Compose:** A `docker-compose.yml` file is provided for orchestrating the API, monitoring stack, and supporting services with a single command.
- **Monitoring & Observability:**
  - **Prometheus Metrics Exposure:** The API exposes Prometheus-compatible metrics at `/metrics` for real-time monitoring of model, data, and server health.
  - **Grafana Dashboard:** A sample `dashboard.json` is included for visualizing key metrics in Grafana.
  - **Three Key Metrics:**
    - **Model Metric:** Histogram of prediction confidence probabilities to track model certainty and detect anomalies.
    - **Data Metric:** Data drift monitoring by comparing the mean of incoming data to the training data mean, alerting when retraining may be needed.
    - **Server Metric:** Request latency logging to ensure API performance and identify bottlenecks.

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

3. **Run the API:**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Build and run with Docker:**
   ```bash
   docker build -t hand-gesture-api .
   docker run -p 8000:8000 hand-gesture-api
   ```
5. **Run with Docker Compose (recommended for full stack):**
   ```bash
   docker-compose up --build
   ```
   This will start the API, Prometheus, and Grafana (pre-configured with the provided dashboard).

## Requirements Files
- `requirements.txt`: Contains only the dependencies needed to run the application and is used in the Docker container.
- `requirements-dev.txt`: Contains additional packages for development (e.g., `pytest`, `fastapi[test]`, etc.). Both files are maintained in the development branch to support both production and development workflows.

## API Endpoints
- `GET /` — Health check and welcome message.
- `POST /predict` — Accepts a JSON payload with hand landmarks and returns the predicted gesture and confidence.
- `GET /metrics` — Exposes Prometheus metrics for monitoring.

## Monitoring & Observability
- **Prometheus Metrics:** Access at `http://localhost:8000/metrics` when running locally or via Docker Compose.
- **Grafana Dashboard:** Import the provided `dashboard.json` into Grafana to visualize model confidence, data drift, and latency metrics.
- **docker-compose.yml:** Orchestrates the API, Prometheus, and Grafana for a complete monitoring stack.
