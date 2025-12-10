import joblib
from pathlib import Path
from src.preprocessor import PersianPreprocessor


class TextClassifier:
    # Wrapper class for loading the trained model + vectorizer,
    # and running prediction on new text.

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = PersianPreprocessor()

    def predict(self, text: str):
        # Predict label of a single text input.
        clean_text = self.preprocessor.preprocess(text)
        vector = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(vector)[0]
        return prediction
