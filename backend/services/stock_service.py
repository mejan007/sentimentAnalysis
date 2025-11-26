import yfinance as yf
import pandas as pd
from typing import List, Dict

class StockService:
    """
    Service for fetching and processing stock data using yfinance.

    """
    
    def __init__(self):
        """
        Initialize StockService with an in-memory cache.
        """
        self.cache = {}  # Simple in-memory cache
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch historical stock data from yfinance for a single ticker and return a list of dicts.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            List[Dict]: List of historical records with date, open, high, low, close, volume.
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Download data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Convert to list of dicts for API response
            data = []
            for _, row in df.iterrows():
                data.append({
                    "date": row['Date'].strftime('%Y-%m-%d'),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })
            
            # Cache the result
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def get_dataframe_for_ml(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Return a DataFrame for a single ticker with technical indicators applied.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame containing historical data and technical indicators.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        df.reset_index(inplace=True)
        df['Stock'] = ticker
        df.rename(columns={'Date': 'date'}, inplace=True)

        df = self._add_technical_indicators(df)
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators (SMA, EMA, RSI, MACD, etc.) to a merged DataFrame grouped by 'Stock'.

        Args:
            df (pd.DataFrame): Input DataFrame with required columns including 'Stock'.

        Returns:
            pd.DataFrame: DataFrame with added technical indicator columns.
        """
        if 'Stock' not in df.columns:
            raise ValueError("DataFrame must contain 'Stock' column for grouped indicators")

        # Simple Moving Averages (SMA)
        df['SMA_30'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=30).mean())
        df['SMA_60'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=60).mean())
        df['SMA_90'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=90).mean())

        # Exponential Moving Averages (EMA)
        df['EMA_30'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
        df['EMA_60'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=60, adjust=False).mean())
        df['EMA_90'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=90, adjust=False).mean())

        # Standard Moving Averages (explicit windows)
        df['30_day_MA'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=30).mean())
        df['60_day_MA'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=60).mean())
        df['90_day_MA'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=90).mean())

        # RSI Calculation via transform
        def calculate_rsi(x, periods=14):
            """
            Helper to calculate RSI over a pandas Series for grouped transform.

            Args:
                x (pd.Series): Series of prices.
                periods (int): RSI lookback.

            Returns:
                pd.Series: RSI values.
            """
            delta = x.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['RSI'] = df.groupby('Stock')['Close'].transform(calculate_rsi)

        # MACD Calculation
        df['EMA_12'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df['EMA_26'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df.groupby('Stock')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        # Price change percentage
        df['Price_Change_Pct'] = df.groupby('Stock')['Close'].transform(lambda x: x.pct_change() * 100)

        # Drop initial NaNs from rolling computations conservatively (keep if at least one indicator present)
        df.dropna(inplace=True)

        # Debug prints
        print(f"After technical indicators: {df.shape}")
        print(df.columns.tolist())
        print(df.head())

        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for a price series.

        Args:
            prices (pd.Series): Series of closing prices.
            period (int): Lookback period for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """
        Calculate MACD, signal line, and histogram for a price series.

        Args:
            prices (pd.Series): Series of closing prices.
            fast (int): Fast EMA span.
            slow (int): Slow EMA span.
            signal (int): Signal line EMA span.

        Returns:
            Dict: Dictionary with 'macd', 'signal', and 'histogram' Series.
        """
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def get_multiple_stocks(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and merge data for multiple tickers into a single DataFrame with indicators.

        Args:
            tickers (List[str]): List of ticker symbols.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Merged DataFrame containing data and technical indicators for all tickers.
        """
        frames = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue
                df.reset_index(inplace=True)
                df['Stock'] = ticker
                df.rename(columns={'Date': 'date'}, inplace=True)
                print(f"Fetched {len(df)} rows for {ticker}")
                print(df.head())
                frames.append(df)
            except Exception as e:
                print(f"Warning: Could not fetch data for {ticker}: {str(e)}")
                continue

        if not frames:
            raise ValueError("No stock data fetched for provided tickers")

        merged_df = pd.concat(frames, ignore_index=True)
        merged_df = self._add_technical_indicators(merged_df)
        return merged_df
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate whether a ticker symbol exists by checking yfinance metadata.

        Args:
            ticker (str): Ticker symbol to validate.

        Returns:
            bool: True if ticker appears valid; False otherwise.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info or 'shortName' in info
        except Exception as e:
            print(f"Error validating ticker {ticker}: {str(e)}")
            return False
