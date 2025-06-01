import joblib

class GestureClassifier:
    """A class to classify hand gestures using a pre-trained model"""

    def __init__(self):
        self.model = joblib.load("model/hand_gesture_final_model.pkl")
        self.encoder = joblib.load("model/encoder.pkl")

    def predict(self, hand_landmarks):
        """Predicts the hand gesture from the landmarks of the hand

        Arguments:
            hand_landmarks -- landmarks of the hand

        Returns:
            predicted gesture
        """
        pred = self.model.predict(hand_landmarks)
        pred = self.encoder.inverse_transform(pred)

        return pred[0]
    