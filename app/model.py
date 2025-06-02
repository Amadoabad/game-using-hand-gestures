import joblib
from prometheus_client import Histogram, Gauge
from app.preprocessing import get_tst_points, normalize_hand
import numpy as np


confidence_histogram = Histogram(
    "model_prediction_confidence",
    "Confidence of model predictions",
    buckets=[1/10.0 for i in range(11)]
)

# Add drift monitoring gauge
drift_gauge = Gauge(
    "model_data_drift",
    "Data drift from training distribution (L2 norm of mean difference)"
)

class GestureClassifier:
    """A class to classify hand gestures using a pre-trained model"""

    def __init__(self):
        self.model = joblib.load("model/rforest.pkl")
        self.encoder = joblib.load("model/encoder.pkl")
        self.training_mean = np.load("model/training_mean.npy")
        
        
    def predict(self, hand_landmarks):
        """Predicts the hand gesture from the landmarks of the hand

        Arguments:
            hand_landmarks -- landmarks of the hand

        Returns:
            predicted gesture and confidence
        """
        # Preprocessing
        hand_landmarks = get_tst_points(hand_landmarks, order=True)
        hand_landmarks = normalize_hand(hand_landmarks)
        hand_landmarks = hand_landmarks.to_frame().T
        
        # Make single prediction and get probabilities
        probabilities = self.model.predict_proba(hand_landmarks)
        pred_index = np.argmax(probabilities)
        confidence = float(probabilities[0, pred_index])
        confidence_histogram.observe(confidence)
        
        # Monitoring: Calculate drift and expose to Prometheus
        current_mean = hand_landmarks.mean(axis=0)
        drift = np.linalg.norm(current_mean - self.training_mean)
        drift_gauge.set(float(drift))
        print(f'Drift from training mean: {drift:.4f}')
        
        # Get prediction from index and map labels
        pred = self.encoder.inverse_transform([pred_index])[0]
        
        if pred == 'one':
            pred = 'up'
        elif pred == 'dislike':
            pred = 'down'
        elif pred == 'call':
            pred = 'left'
        elif pred == 'rock':
            pred = 'right'
        
        return pred, confidence