import re
import feedparser
from datetime import datetime, timedelta
from typing import List
from urllib.parse import quote_plus
import html  # Add this import


class NewsService:
    """
    Service to fetch recent news articles from Google News RSS using feedparser.
    """

    COMPANY_MAP = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "NVDA": "Nvidia",
        "TSLA": "Tesla",
    }

    def __init__(self):
        pass

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities like &nbsp;
        text = html.unescape(text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fetch_recent_news(self, ticker: str, days: int = 7, max_articles: int = 5) -> List[str]:
        """Fetch recent news articles for a given ticker symbol."""
        try:
            company = self.COMPANY_MAP.get(ticker.upper(), ticker)
            query = f"{company} stock"
            
            # URL-encode the query to handle spaces and special characters
            encoded_query = quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}"
            
            print(f"\n{'='*60}")
            print(f"🔍 Fetching news for: {ticker} ({company})")
            print(f"📡 RSS URL: {rss_url}")
            print(f"📅 Looking for articles from last {days} days")
            print(f"🎯 Max articles: {max_articles}")
            print(f"{'='*60}\n")

            # Parse feed with error handling
            try:
                feed = feedparser.parse(rss_url)
                print("✅ Feed parsed successfully")
                print(f"   Total entries: {len(feed.entries)}")
                
                if hasattr(feed, 'bozo_exception') and feed.bozo:
                    print(f"   ⚠️ Feed exception: {feed.bozo_exception}")
                    
            except Exception as parse_error:
                print(f"❌ Error parsing feed: {parse_error}")
                return []

            # Check for entries
            num_entries = len(feed.entries)
            if num_entries == 0:
                print(f"⚠️ No RSS entries found for {ticker} ({company})")
                return []

            cutoff = datetime.utcnow() - timedelta(days=days)
            print(f"🕒 Cutoff date: {cutoff.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            
            articles = []
            pattern = re.compile(company, re.IGNORECASE)

            # Process entries
            for idx, entry in enumerate(feed.entries, 1):
                try:
                    # Extract title
                    title = getattr(entry, "title", "")
                    title_clean = self._clean_html(title)
                    
                    # Extract and parse published date
                    published_dt = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            published_dt = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass

                    # Check date filter
                    if published_dt and published_dt < cutoff:
                        continue

                    # Extract summary/description
                    summary = (
                        getattr(entry, "summary", None)
                        or getattr(entry, "description", None)
                        or ""
                    )
                    summary_clean = self._clean_html(summary)

                    # Combine text - use only title for cleaner output
                    # The summary from Google News RSS is mostly just links
                    text = title_clean

                    combined_text = f"{title_clean} {summary_clean}".strip()
                    
                    print(f"[{idx}] {text[:100]}...")

                    # Check pattern match
                    if not pattern.search(text):
                        print(f"    ❌ Skipped: '{company}' not found\n")
                        continue

                    # Add to results
                    articles.append(combined_text)
                    print(f"    ✅ Added ({len(articles)}/{max_articles})\n")

                    if len(articles) >= max_articles:
                        break

                except Exception as entry_error:
                    print(f"   ❌ Error processing entry {idx}: {entry_error}\n")
                    continue

            print(f"{'='*60}")
            print(f"✅ Collected {len(articles)} articles for {ticker}")
            print(f"{'='*60}\n")
            
            return articles

        except Exception as outer_error:
            print(f"\n❌ CRITICAL ERROR: {outer_error}")
            import traceback
            print(traceback.format_exc())
            return []