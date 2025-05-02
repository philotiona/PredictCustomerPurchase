import pandas as pd
from model import PurchasePredictor
from utils import setup_logger

def main():
    logger = setup_logger()
    logger.info("Starting prediction script...")

    predictor = PurchasePredictor()
    predictor.load_model('models/purchase_predictor.pkl')

    new_data = pd.read_csv('data/raw/new_purchases.csv')
    logger.info(f"Loaded new data with shape: {new_data.shape}")

    X, _ = predictor.prepare_features(new_data)

    predictions = predictor.predict(X)
    new_data['predicted_item_id'] = predictions

    new_data.to_csv('data/processed/predictions.csv', index=False)
    logger.info("Predictions saved to data/processed/predictions.csv")

if __name__ == "__main__":
    main()