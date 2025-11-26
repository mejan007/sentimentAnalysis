"""
API routes for stock sentiment analysis.

Args:

Returns:

"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import pandas as pd

from ..models.schemas import (
    PredictionRequest,
    AnalysisResponse
)
import importlib

router = APIRouter()


@router.get("/", summary="Root")
async def root():
    """
    Root endpoint returning API metadata.

    Args:

    Returns:
        dict: Basic API information and available endpoints.
    """
    return {
        "message": "Stock Sentiment Analysis API",
        "status": "running",
        "endpoints": [
            "/predict",
            "/stocks/tickers",
            "/stocks/historical/{ticker}",
            "/health"
        ]
    }


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    Args:

    Returns:
        dict: Health status and current timestamp.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.post("/predict", response_model=AnalysisResponse)
async def predict_stock_prices(request: PredictionRequest):
    """
    Predict stock prices using ML and sentiment data.

    Args:
        request (PredictionRequest): Request parameters including tickers and date range.

    Returns:
        AnalysisResponse: Aggregated analysis including predictions, sentiments, and metrics.
    """
    try:
        # Initialize services
        try:
            StockService = getattr(importlib.import_module("backend.services.stock_service"), "StockService")
            MLService = getattr(importlib.import_module("backend.services.ml_service"), "MLService")
            NewsService = getattr(importlib.import_module("backend.services.news_service"), "NewsService")
            
            stock_service = StockService()
            ml_service = MLService()
            news_service = NewsService()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Service initialization failed: {str(e)}")

        # Fetch stock data
        try:
            merged_df = stock_service.get_multiple_stocks(
                request.tickers, request.start_date, request.end_date
            )
            
            if merged_df.empty:
                raise HTTPException(status_code=404, detail="No stock data found for the given tickers and date range")
            
            print(f"✅ Fetched stock data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")

        # Format stock data for response
        try:
            stocks_data = {}
            for ticker in request.tickers:
                df_t = merged_df[merged_df['Stock'] == ticker].copy()
                
                if df_t.empty:
                    print(f"⚠️ No data found for ticker: {ticker}")
                    stocks_data[ticker] = []
                    continue
                
                stocks_data[ticker] = [
                    {
                        "date": pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume'])
                    }
                    for _, row in df_t.iterrows()
                ]
                print(f"✅ Formatted {len(stocks_data[ticker])} records for {ticker}")
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Missing required column in stock data: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to format stock data: {str(e)}")

        # Fetch news and analyze sentiment
        try:
            ticker_embeddings = {}
            ticker_sentiments = {}
            
            for ticker in request.tickers:
                try:
                    articles = news_service.fetch_recent_news(ticker)
                    
                    if not articles:
                        print(f"⚠️ No articles found for {ticker}, using default sentiment")
                        ticker_sentiments[ticker] = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
                        ticker_embeddings[ticker] = [0.0] * 384  # Default embedding dimension
                        continue
                    
                    ticker_sentiments[ticker] = ml_service.analyze_sentiment(articles)
                    emb = ml_service.generate_embeddings(articles)
                    ticker_embeddings[ticker] = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    
                    print(f"✅ {ticker}: {len(articles)} articles, sentiment analyzed")
                except Exception as e:
                    print(f"⚠️ Error processing news for {ticker}: {str(e)}")
                    # Use defaults on error
                    ticker_sentiments[ticker] = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
                    ticker_embeddings[ticker] = [0.0] * 384
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"News/sentiment analysis failed: {str(e)}")

        # Add embeddings to dataframe
        try:
            emb_len = len(next(iter(ticker_embeddings.values()))) if ticker_embeddings else 384
            
            # Initialize embedding columns
            for i in range(emb_len):
                merged_df[f"embedding_{i}"] = 0.0
            
            # Populate embeddings per ticker
            for ticker, emb in ticker_embeddings.items():
                mask = merged_df['Stock'] == ticker
                if not mask.any():
                    print(f"⚠️ No rows found for {ticker} when adding embeddings")
                    continue
                    
                for i, val in enumerate(emb):
                    merged_df.loc[mask, f"embedding_{i}"] = float(val)
            
            print(f"✅ Added {emb_len} embedding features to dataframe")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add embeddings: {str(e)}")

        # Generate predictions
        try:
            predictions = []
            feature_importance = {}
            
            for ticker in request.tickers:
                try:
                    result = ml_service.predict_price(
                        merged_df,
                        ticker,
                    )
                    predictions.append(result['prediction'])
                    feature_importance[ticker] = result['feature_importance']
                    print(f"✅ Prediction generated for {ticker}")
                except Exception as e:
                    print(f"❌ Prediction failed for {ticker}: {str(e)}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Prediction failed for {ticker}: {str(e)}"
                    )
                    
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction process failed: {str(e)}")

        # Get model metrics and return response
        try:
            model_metrics = ml_service.get_model_metrics()
            
            return AnalysisResponse(
                stocks_data=stocks_data,
                predictions=predictions,
                sentiment_scores=ticker_sentiments,
                feature_importance=feature_importance,
                model_metrics=model_metrics,
                date_range={
                    "start": request.start_date,
                    "end": request.end_date,
                },
                articles=articles
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build response: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as exc:
        print(f"❌ Unexpected error: {str(exc)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}")


@router.get("/stocks/tickers")
async def get_available_tickers():
    """
    Return a list of available tickers supported by the API.

    Args:

    Returns:
        dict: Dictionary with a 'tickers' key mapping to a list of ticker info.
    """
    tickers = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    ]
    return {"tickers": tickers}


@router.get("/stocks/historical/{ticker}")
async def get_historical_data(ticker: str, start_date: str, end_date: str):
    """
    Get historical stock data for a specific ticker.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        dict: Contains the ticker, data list, and count of records.
    """
    try:
        StockService = getattr(importlib.import_module("api.app.services.stock_service"), "StockService")
        stock_service = StockService()
        data = stock_service.get_historical_data(ticker, start_date, end_date)
        return {"ticker": ticker, "data": data, "count": len(data)}
    except Exception:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found or invalid date range")
