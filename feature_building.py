import pandas as pd
import numpy as np
import json
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer
import ta
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class StockTrendPredictor:
    def __init__(self, news_stock_path, clusters_path, strengths_path):
        """
        Initialize the predictor with paths to required data files
        """
        self.news_stock_path = news_stock_path
        self.clusters_path = clusters_path
        self.strengths_path = strengths_path
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.scaler = MinMaxScaler()

    def get_sentiment_score(self, text):
        """
        Calculate combined sentiment score using TextBlob and FinBERT,
        handling long texts and edge cases properly
        """
        # Basic sentiment using TextBlob
        blob_sentiment = TextBlob(str(text)).sentiment.polarity

        try:
            # Convert text to string and clean it
            text = str(text).strip()
            if not text:  # Handle empty text
                return 0.0

            # Process with FinBERT
            try:
                # Truncate text to maximum token length (512)
                words = text.split()[:512]  # Simple truncation
                truncated_text = ' '.join(words)

                fin_result = self.finbert(truncated_text)[0]

                # Standardize label case and handle variations
                label = fin_result['label'].upper()
                fin_sentiment = {
                                    'POSITIVE': 1.0,
                                    'NEUTRAL': 0.0,
                                    'NEGATIVE': -1.0,
                                    'LABEL_0': -1.0,  # Some models use numerical labels
                                    'LABEL_1': 0.0,
                                    'LABEL_2': 1.0
                                }.get(label, 0.0) * fin_result['score']

                # Combine TextBlob and FinBERT sentiments
                combined_sentiment = 0.3 * blob_sentiment + 0.7 * fin_sentiment
                return combined_sentiment

            except Exception as e:
                print(f"FinBERT processing error: {e}")
                return blob_sentiment

        except Exception as e:
            print(f"General sentiment error: {e}")
            return 0.0  # Return neutral sentiment in case of major errors

    def load_data(self):
        """
        Load and initialize all required data
        """
        # Load main dataset
        self.df = pd.read_csv(self.news_stock_path)
        self.df['date'] = pd.to_datetime(self.df['datePublished'])

        # Load clusters and strengths
        with open(self.clusters_path, 'r', encoding = 'utf-8') as f:
            self.clusters = json.load(f)
        with open(self.strengths_path, 'r', encoding = 'utf-8') as f:
            self.causal_strengths = json.load(f)

        # Extract trend nodes and their relationships
        self.trend_nodes = ['Trend_Up', 'Trend_Neutral', 'Trend_Down']
        self.cluster_trend_strengths = {
            cluster: {trend: 0.0 for trend in self.trend_nodes}
            for cluster in self.clusters.keys()
        }

        # Populate cluster-trend relationships from causal strengths
        for relation, strength in self.causal_strengths.items():
            source, target = relation.split(' -> ')
            if target in self.trend_nodes:
                self.cluster_trend_strengths[source][target] = strength

    def assign_clusters(self, tags):
        """
        Assign clusters based on tags with weighted scoring
        """
        cluster_scores = {cluster: 0 for cluster in self.clusters.keys()}
        tags_list = tags.split(', ')

        for cluster, cluster_tags in self.clusters.items():
            matches = sum(1 for tag in tags_list if tag in cluster_tags)
            if matches > 0:
                cluster_scores[cluster] = matches / len(tags_list)

        # Get primary and secondary clusters
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        primary_cluster = sorted_clusters[0][0]
        secondary_cluster = sorted_clusters[1][0] if len(sorted_clusters) > 1 and sorted_clusters[1][1] > 0 else None

        return primary_cluster, secondary_cluster

    def calculate_technical_indicators(self, window_short=12, window_long=26):
        """
        Calculate technical indicators for the stock data
        """
        df_tech = self.df.copy()

        # Basic technical indicators
        df_tech['RSI'] = ta.momentum.RSIIndicator(close=df_tech['Close'], window=14).rsi()

        macd = ta.trend.MACD(
            close=df_tech['Close'],
            window_slow=window_long,
            window_fast=window_short
        )
        df_tech['MACD'] = macd.macd()
        df_tech['MACD_signal'] = macd.macd_signal()

        # Add price momentum
        df_tech['Returns'] = df_tech['Close'].pct_change()
        df_tech['Returns_5d'] = df_tech['Close'].pct_change(periods=5)

        # Moving averages
        df_tech['SMA_20'] = df_tech['Close'].rolling(window=20).mean()
        df_tech['SMA_50'] = df_tech['Close'].rolling(window=50).mean()

        return df_tech

    def calculate_cluster_influence(self, row):
        """
        Calculate cluster influence on trends based on causal strengths
        """
        primary_cluster = row['primary_cluster']
        secondary_cluster = row['secondary_cluster']

        trend_influences = {trend: 0.0 for trend in self.trend_nodes}

        # Primary cluster influence
        for trend in self.trend_nodes:
            trend_influences[trend] += self.cluster_trend_strengths[primary_cluster][trend]

        # Secondary cluster influence (if exists, with lower weight)
        if secondary_cluster:
            for trend in self.trend_nodes:
                trend_influences[trend] += 0.5 * self.cluster_trend_strengths[secondary_cluster][trend]

        return pd.Series(trend_influences)

    def batch_sentiment_analysis(self, texts, batch_size=32):
        """
        Process sentiments in batches for better efficiency
        """
        sentiments = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_sentiments = []

            for text in batch:
                sentiment = self.get_sentiment_score(text)
                batch_sentiments.append(sentiment)

            sentiments.extend(batch_sentiments)

        return sentiments

    def engineer_features(self):
        """
        Complete feature engineering pipeline integrating all components
        """
        # Assign clusters
        self.df['primary_cluster'], self.df['secondary_cluster'] = zip(
            *self.df['tags'].apply(self.assign_clusters)
        )

        # Calculate cluster influences on trends first
        print("Calculating cluster influences...")
        trend_influences = self.df.apply(self.calculate_cluster_influence, axis=1)

        # Add all trend influences to DataFrame
        for trend in self.trend_nodes:  # This includes all three trends: Up, Neutral, Down
            self.df[f'influence_{trend}'] = trend_influences[trend]

        # Calculate sentiment scores
        print("Calculating sentiment scores...")
        articles = self.df['articleBody'].fillna('').values
        sentiments = self.batch_sentiment_analysis(articles)
        self.df['sentiment_score'] = sentiments

        # Add technical indicators
        print("Calculating technical indicators...")
        df_tech = self.calculate_technical_indicators()
        technical_columns = ['RSI', 'MACD', 'MACD_signal', 'Returns', 'Returns_5d', 'SMA_20', 'SMA_50']
        self.df = pd.concat([self.df, df_tech[technical_columns]], axis=1)

        # Create interaction features with all trends
        print("Creating interaction features...")
        self.df['sentiment_trend_up_interaction'] = (
                self.df['sentiment_score'] * self.df['influence_Trend_Up']
        )
        self.df['sentiment_trend_neutral_interaction'] = (
                self.df['sentiment_score'] * self.df['influence_Trend_Neutral']
        )
        self.df['sentiment_trend_down_interaction'] = (
                self.df['sentiment_score'] * self.df['influence_Trend_Down']
        )

        # Calculate combined trend influence
        self.df['combined_trend_influence'] = (
                self.df['influence_Trend_Up'] -
                self.df['influence_Trend_Down'] +
                0.5 * self.df['influence_Trend_Neutral']  # Neutral has half weight
        )

        # Add time-based features
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['month'] = self.df['date'].dt.month

        # Calculate rolling statistics for all trends
        print("Calculating rolling statistics...")
        for trend in self.trend_nodes:
            self.df[f'rolling_influence_{trend}'] = (
                self.df.groupby('symbol')[f'influence_{trend}']
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        # Scale numerical features
        print("Scaling features...")
        numerical_features = [
                                 'sentiment_score',
                                 'RSI',
                                 'MACD',
                                 'Returns_5d',
                                 'combined_trend_influence',
                                 'sentiment_trend_up_interaction',
                                 'sentiment_trend_neutral_interaction',
                                 'sentiment_trend_down_interaction'
                             ] + [f'influence_{trend}' for trend in self.trend_nodes]

        self.df[numerical_features] = self.scaler.fit_transform(
            self.df[numerical_features].fillna(0)
        )

        return self.df

    def prepare_final_features(self):
        """
        Prepare final feature set for modeling with all trend components
        """
        feature_cols = [
                           'sentiment_score',
                           'RSI',
                           'MACD',
                           'Returns_5d',
                           'combined_trend_influence',
                           'sentiment_trend_up_interaction',
                           'sentiment_trend_neutral_interaction',
                           'sentiment_trend_down_interaction',
                           'day_of_week',
                           'month'
                       ] + [f'influence_{trend}' for trend in self.trend_nodes]

        return self.df[feature_cols]

    # Add print statements in main() for better visibility
def main():
        print("Initializing predictor...")
        predictor = StockTrendPredictor(
            news_stock_path='News+Stock data.csv',
            clusters_path='clusters.json',
            strengths_path='combined_strengths.json'
        )

        print("Loading data...")
        predictor.load_data()

        print("Engineering features...")
        processed_df = predictor.engineer_features()

        print("Preparing final features...")
        X = predictor.prepare_final_features()

        # Preview results
        print("\nProcessed data shape:", processed_df.shape)
        print("\nFeature columns:", list(X.columns))
        print("\nSample of trend influences:")
        influence_cols = ['primary_cluster', 'secondary_cluster'] + [f'influence_{trend}' for trend in
                                                                     predictor.trend_nodes]
        print(processed_df[influence_cols].head())

        # Print some basic statistics
        print("\nFeature statistics:")
        print(X.describe())

        return processed_df, X

if __name__ == "__main__":
        processed_df, X = main()
        processed_df.to_csv('Market_features.csv', index = False, encoding = 'utf-8')
        X.to_csv("final_features.csv", index=False, encoding = 'utf-8')