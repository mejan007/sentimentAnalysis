"""
Pydantic schemas used by the API for requests and responses.

Args:

Returns:

"""

from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.

    Args:
        tickers (List[str]): List of ticker symbols to analyze.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        prediction_days (Optional[int]): Number of days ahead to predict.

    Returns:
        PredictionRequest: Validated request payload.
    """
    tickers: List[str] = Field(..., example=["AAPL", "MSFT", "GOOGL"])
    start_date: str = Field(..., example="2023-01-01")
    end_date: str = Field(..., example="2024-01-01")


class StockData(BaseModel):
    """
    Single stock data point schema.

    Args:
        date (str): Date string (YYYY-MM-DD).
        open (float): Opening price.
        high (float): High price.
        low (float): Low price.
        close (float): Closing price.
        volume (int): Traded volume.

    Returns:
        StockData: Stock data record.
    """
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PredictionResult(BaseModel):
    """
    Single prediction result for a ticker.

    Args:
        ticker (str): Ticker symbol.
        current_price (float): Current/latest closing price.
        predicted_price (float): Predicted future price.
        price_change_pct (float): Percentage change between predicted and current.
        trend (str): 'bullish' or 'bearish'.
        confidence (float): Confidence score for the prediction.

    Returns:
        PredictionResult: Prediction details for a ticker.
    """
    ticker: str
    current_price: float
    predicted_price: float
    price_change_pct: float
    trend: str  # "bullish" or "bearish"
    confidence: float


class AnalysisResponse(BaseModel):
    """
    Response model returned by the prediction endpoint.

    Args:
        stocks_data (dict): Mapping ticker -> list of StockData.
        predictions (List[PredictionResult]): List of prediction results.
        sentiment_scores (dict): Sentiment analysis results.
        feature_importance (dict): Feature importance values.
        model_metrics (dict): Stored model metrics (mae/mse/r2).
        date_range (dict): Input date range and prediction_days.

    Returns:
        AnalysisResponse: Aggregated analysis payload.
    """
    stocks_data: dict  # ticker -> list of StockData
    predictions: List[PredictionResult]
    sentiment_scores: dict
    feature_importance: dict
    model_metrics: dict
    date_range: dict
    articles: List[str]
