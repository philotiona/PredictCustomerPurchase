import pandas as pd
from datetime import datetime
import numpy as np

def load_data():
    """Load data from raw data files"""
    # Reading purchase history and trends data
    purchases_df = pd.read_csv('data/raw/large_purchases.csv')
    return purchases_df

def preprocess_purchases(df):
    """Preprocess the purchases data"""
    # Basic datetime conversion
    df['date'] = pd.to_datetime(df['date'])
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['week_of_month'] = df['date'].dt.day // 7 + 1
    
    # Purchase patterns
    customer_items = df.groupby('customer_id')['item_id'].agg(list).reset_index()
    customer_items['unique_items'] = customer_items['item_id'].apply(lambda x: len(set(x)))
    
    # Purchase frequency by customer
    purchase_freq = df.groupby(['customer_id', 'item_id']).size().reset_index(name='buy_count')
    
    # Last purchase date for each customer-item pair
    last_purchase = df.groupby(['customer_id', 'item_id'])['date'].max().reset_index()
    last_purchase.rename(columns={'date': 'last_bought'}, inplace=True)
    
    return df, purchase_freq, last_purchase

def main():
    # Load raw data
    purchases_df = load_data()
    
    # Process and create features
    processed_df, frequency_df, last_purchase_df = preprocess_purchases(purchases_df)
    
    # Save processed datasets
    processed_df.to_csv('data/processed/processed_purchases.csv', index=False)
    frequency_df.to_csv('data/processed/purchase_frequencies.csv', index=False)
    last_purchase_df.to_csv('data/processed/last_purchases.csv', index=False)

if __name__ == "__main__":
    main()
load_data()