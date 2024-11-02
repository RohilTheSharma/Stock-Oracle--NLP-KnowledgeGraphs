import pandas as pd
import numpy as np
import json

def load_data():
    """Load and preprocess raw data files"""
    print("Loading data...")

    # Load news and stock data
    news_df = pd.read_csv("News_Train.csv", encoding = 'utf-8')
    stock_df = pd.read_csv("Nifty_50_data.csv", encoding = 'utf-8')

    # Load saved JSON files
    with open('combined_strengths.json', 'r', encoding='utf-8') as f:
        combined_strengths = json.load(f)

    with open('clusters.json', 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    # Convert dates
    news_df['datePublished'] = pd.to_datetime(news_df['datePublished'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Basic preprocessing
    news_df = news_df.sort_values('datePublished')
    stock_df = stock_df.sort_values('Date')

    # Create feature names list from clusters
    feature_names = list(clusters.keys())

    return news_df, stock_df, combined_strengths, clusters, feature_names


def combine_news_and_stock_data(news_df, stock_df):
    """Combine news_df with stock_df columns based on date, filling missing data with forward/backward fill and averages."""
    print("Combining news and stock data...")

    # Convert 'datePublished' in news_df and 'Date' in stock_df to just dates (no time or timezone)
    news_df['datePublished'] = news_df['datePublished'].dt.date
    stock_df['Date'] = stock_df['Date'].dt.date

    # Add empty columns for stock data in news_df
    for col in stock_df.columns:
        if col != 'Date':  # Avoid duplicating the Date column
            news_df[col] = np.nan

    # Set indices to facilitate date-by-date comparison
    news_df.set_index('datePublished', inplace=True)
    stock_df.set_index('Date', inplace=True)

    # Fill in matching dates and NaNs for missing dates
    for date in news_df.index:
        if date in stock_df.index:
            # Copy stock row values into the matching news_df row
            news_df.loc[date, stock_df.columns] = stock_df.loc[date].values
        else:
            # Fill with NaNs to be replaced later with averages
            news_df.loc[date, stock_df.columns] = np.nan

    # Reset index for a clean DataFrame
    news_df.reset_index(inplace=True)

    # Step 1: Forward and Backward Fill to handle close gaps
    news_df.fillna(method='ffill', inplace=True)
    news_df.fillna(method='bfill', inplace=True)

    # Step 2: Fill remaining NaNs with column averages
    averages = stock_df.mean()
    news_df.fillna(value=averages, inplace=True)

    print("Combined DataFrame shape:", news_df.shape)
    return news_df


# Load data and combine
news_df, stock_df, _, _, _ = load_data()
d = combine_news_and_stock_data(news_df, stock_df)
d = d.drop(['Unnamed: 0', 'author', 'url'], axis=1)
d.to_csv('News+Stock data.csv', index=False, encoding='utf-8')