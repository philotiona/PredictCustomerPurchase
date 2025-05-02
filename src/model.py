import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  

class PurchasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def prepare_features(self, df):
        for col in ['customer_id', 'item_id']:
            if col not in self.label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        features = ['customer_id', 'day_of_week', 'is_weekend', 'week_of_month']
        X = df[features]
        y = df['item_id']
        return X, y

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    def save_model(self, file_path):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'encoders': self.label_encoders}, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        data = joblib.load(file_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['encoders']
        print(f"Model loaded from {file_path}")
