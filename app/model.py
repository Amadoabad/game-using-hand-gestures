import joblib
from app.preprocessing import get_tst_points, normalize_hand

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
        tst = get_tst_points(hand_landmarks.landmark, order=True)
        tst = normalize_hand(tst)

        pred = self.model.predict(tst.values.reshape((-1, 63)))
        pred = self.encoder.inverse_transform(pred)

        return pred[0]
    