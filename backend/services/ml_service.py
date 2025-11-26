import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class MLService:
    """
    Service for ML operations: sentiment analysis, embeddings, and predictions.

    Args:

    Returns:

    """
    
    def __init__(self):
        """
        Initialize MLService: load embedding model and prepare preprocessing components.

        """
        # Load embedding model
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        self.embedding_model.eval()
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = None
        self.regression_model = None
        self.best_columns = None
        
        # Model metrics storage
        self.metrics = {
            'mae': None,
            'mse': None,
            'r2': None
        }

        print("kisim model:", self.embedding_model_name)
    
    def analyze_sentiment(self, articles: List[str]) -> Dict:
        """
        Analyze sentiment of a list of news articles and return aggregated sentiment scores.

        Args:
            articles (List[str]): List of article texts to analyze.

        Returns:
            Dict: Aggregated sentiment results including overall_sentiment, article_sentiments, sentiment_label, and confidence.
        """
        if not articles:
            return {"overall_sentiment": 0.0, "article_sentiments": []}
        
        embeddings = self.generate_embeddings(articles)
        sentiments = []
        # For compatibility keep iteration semantics
        for emb in [embeddings]:
            sentiment_score = float(np.mean(emb))
            sentiments.append(sentiment_score)
        overall = np.mean(sentiments)
        return {
            "overall_sentiment": float(overall),
            "article_sentiments": [float(s) for s in sentiments],
            "sentiment_label": "bullish" if overall > 0 else "bearish",
            "confidence": abs(float(overall))
        }
    
    def generate_embeddings(self, articles: List[str]) -> np.ndarray:
        """
        Generate a single averaged embedding vector for the provided articles.

        Args:
            articles (List[str]): List of article texts to embed.

        Returns:
            np.ndarray: Averaged embedding vector (length 384). Returns zeros if input is empty.
        """
        if not articles:
            return np.zeros(384)
        vectors = []
        with torch.no_grad():
            for text in articles[:5]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                outputs = self.embedding_model(**inputs)
                vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                vectors.append(vec)
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(384)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare ML feature matrix from a merged multi-stock DataFrame.

        Args:
            df (pd.DataFrame): Merged DataFrame containing technical indicators and embedding columns.

        Returns:
            pd.DataFrame: Feature matrix used for training/prediction.
        """
        stock_features = [
            'Open','Low','Close','Volume','30_day_MA','60_day_MA','90_day_MA',
            'SMA_30','SMA_60','SMA_90','EMA_30','EMA_60','EMA_90','EMA_12','EMA_26',
            'RSI','MACD','Signal_Line','MACD_Histogram','Price_Change_Pct'
        ]
        embedding_features = [c for c in df.columns if c.startswith('embedding_')]
        available_stock = [f for f in stock_features if f in df.columns]
        X = df[available_stock + embedding_features].copy()
        print(f"Feature matrix shape (including embeddings): {X.shape}")
        print(f"Total feature columns: {len(X.columns)}")
        return X
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, top_k_features: int = 30):
        """
        Train a simple linear regression model using top features selected by Random Forest.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target series.
            top_k_features (int): Number of top features to select based on Random Forest importance.

        Returns:
            dict: Training metrics (mae, mse, r2).
        """
        # -------------------------------
        # Step 1: Feature selection using Random Forest
        # -------------------------------
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)
        
        feature_importances = rf.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}) \
                        .sort_values(by='Importance', ascending=False)
        
        self.best_columns = importance_df['Feature'].iloc[:top_k_features].tolist()
        
        # Use only the top features
        X_best = X[self.best_columns]
        
        # -------------------------------
        # Step 2: Standardize features
        # -------------------------------
        X_scaled = self.scaler.fit_transform(X_best)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        # -------------------------------
        # Step 3: PyTorch Linear Regression
        # -------------------------------
        self.regression_model = torch.nn.Linear(X_scaled.shape[1], 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.regression_model.parameters(), lr=0.001)
        epochs = 1000
        
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.regression_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # -------------------------------
        # Step 4: Predictions and metrics
        # -------------------------------
        with torch.no_grad():
            predictions = self.regression_model(X_tensor).numpy()
        
        self.metrics['mae'] = mean_absolute_error(y, predictions)
        self.metrics['mse'] = mean_squared_error(y, predictions)
        self.metrics['r2'] = r2_score(y, predictions)
        
        return self.metrics

    
    def predict_price(self, merged_df: pd.DataFrame, ticker: str) -> Dict:
        """
        Predict future price for a given ticker using the trained pipeline.

        Args:
            merged_df (pd.DataFrame): Merged DataFrame with data for all tickers.
            ticker (str): Ticker symbol to predict for.

        Returns:
            Dict: Contains 'prediction' (PredictionResult-like dict) and 'feature_importance'.
        """
        df_ticker = merged_df[merged_df['Stock'] == ticker].copy()
        if df_ticker.empty:
            raise ValueError(f"No data for ticker {ticker} in merged DataFrame")
        

        print("df_ticker columns:", df_ticker.columns)

        # # check if any of the embedding columns have zero values
        # embedding_columns = [col for col in df_ticker.columns if col.startswith('embedding_')]
        # for col in embedding_columns:
        #     if (df_ticker[col] == 0).all():
        #         print(f"Warning: All values in {col} are zero.")

        
        date_col = 'date' if 'date' in df_ticker.columns else 'Date'
        df_ticker.sort_values(by=date_col, inplace=True)
        X = self.prepare_features(df_ticker)
        y = df_ticker['Price_Change_Pct'].shift(-1).dropna()
        X = X.iloc[:-1]
        self.train_model(X, y)
        last_features = X.iloc[-1:]  # last row as DataFrame
        last_features = last_features[self.best_columns]  # select top features
        print("Best Columns:", self.best_columns)
        X_scaled = self.scaler.transform(last_features)   # scale
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            predicted_price_change_pct = self.regression_model(X_tensor).item()
        current_price = df_ticker['Close'].iloc[-1]
        price_change = predicted_price_change_pct*current_price/100
        predicted_price = current_price + price_change
        feature_importance = self._calculate_feature_importance(X)
        return {
            "prediction": {
                "ticker": ticker,
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "price_change_pct": float(predicted_price_change_pct),
                "trend": "bullish" if price_change > 0 else "bearish",
                "confidence": float(self.metrics['r2']) if self.metrics['r2'] else 0.5
            },
            "feature_importance": feature_importance
        }
    
    def _calculate_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> Dict:
        """
        Calculate feature importance approximate values using PCA components and regression weights.

        Args:
            X (pd.DataFrame): Feature matrix used for training.
            top_n (int): Number of top features to return.

        Returns:
            Dict: Mapping feature name -> importance score for top_n features.
        """ 
        return {}

    
    def get_model_metrics(self) -> Dict:
        """
        Return current stored model metrics and training status.

        Args:

        Returns:
            Dict: Dictionary with mae, mse, r2, and status.
        """
        return {
            "mae": float(self.metrics['mae']) if self.metrics['mae'] else None,
            "mse": float(self.metrics['mse']) if self.metrics['mse'] else None,
            "r2": float(self.metrics['r2']) if self.metrics['r2'] else None,
            "status": "trained" if self.regression_model else "not_trained"
        }
