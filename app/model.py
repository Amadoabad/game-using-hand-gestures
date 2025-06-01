import joblib
from prometheus_client import Histogram
import numpy as np


confidence_histogram = Histogram(
    "model_prediction_confidence",
    "Confidence of model predictions",
    buckets=[1/10.0 for i in range(11)]
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
            predicted gesture
        """
        # Monitoring: Model prediction confidence histogram
        probabilities = self.model.predict_proba(hand_landmarks)
        confidence = float(np.max(probabilities))
        confidence_histogram.observe(confidence)
        
        # Monitoring: Calculate drift
        current_mean = hand_landmarks.mean(axis=1)
        drift = np.linalg.norm(current_mean - self.training_mean)
        print(f'Drift from training mean: {drift:.4f}')
        
        
        pred = self.model.predict(hand_landmarks)
        pred = self.encoder.inverse_transform(pred)

        return pred[0], confidence
    
    
    