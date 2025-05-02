import pandas as pd
from model import PurchasePredictor

def main():
    df = pd.read_csv('data/processed/processed_purchases.csv')

    predictor = PurchasePredictor()

    X, y = predictor.prepare_features(df)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor.train(X_train, y_train)

    predictor.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()