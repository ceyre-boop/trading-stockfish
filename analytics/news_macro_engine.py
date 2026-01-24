"""
NewsMaproEngine: AI/NLP-powered macroeconomic and news feature extraction.

Converts high-impact news, macro events, and textual information into quantifiable,
time-aligned features for the trading engine and ELO tournament.

Key Principles:
- STRICTLY time-causal: no future data leakage
- AI/NLP used ONLY for parsing and classification, never prediction
- All outputs are numeric features usable by state_builder and evaluator
- Real-world data only (CSV/JSON inputs with strict timestamps)
- Production-grade quality with hard-fail semantics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MacroEventCategory(Enum):
    """Macroeconomic event categories."""
    INFLATION = "inflation"  # CPI, PPI, PCE
    EMPLOYMENT = "employment"  # NFP, jobless claims
    RATE_DECISION = "rate_decision"  # Fed, ECB, BoE, BoJ
    GDP = "gdp"  # GDP growth
    TRADE = "trade"  # Trade balance, exports
    SENTIMENT = "sentiment"  # ISM, PMI, consumer confidence
    GEOPOLITICAL = "geopolitical"  # Conflicts, sanctions
    EARNINGS = "earnings"  # Corporate earnings
    OTHER = "other"


class EventImpactLevel(Enum):
    """Impact level of macro events."""
    LOW = 0  # Calendar event, minor
    MEDIUM = 1  # Important event
    HIGH = 2  # Major event, moves markets
    CRITICAL = 3  # Game-changing event


class SentimentPolarity(Enum):
    """Sentiment classification for news/events."""
    STRONGLY_DOVISH = -1.0  # Strongly risk-off
    MILDLY_DOVISH = -0.5
    NEUTRAL = 0.0
    MILDLY_HAWKISH = 0.5
    STRONGLY_HAWKISH = 1.0  # Strongly risk-on


class RiskSentiment(Enum):
    """Overall risk sentiment from macro/news."""
    STRONG_RISK_OFF = -1.0
    MILD_RISK_OFF = -0.5
    NEUTRAL = 0.0
    MILD_RISK_ON = 0.5
    STRONG_RISK_ON = 1.0


@dataclass
class MacroEvent:
    """Structured macro event with timestamp and metadata."""
    timestamp: datetime
    symbol: str  # Currency or broad category (e.g., 'USD', 'EUR', 'GLOBAL')
    category: MacroEventCategory
    title: str
    description: str
    impact_level: EventImpactLevel
    actual: Optional[float] = None  # Actual value released
    forecast: Optional[float] = None  # Consensus forecast
    previous: Optional[float] = None  # Previous value
    
    def __post_init__(self):
        """Validate event data."""
        if self.timestamp is None:
            raise ValueError("[NEWS_MACRO] Event must have timestamp")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("[NEWS_MACRO] Event timestamp must be datetime")


@dataclass
class NewsArticle:
    """Structured news article with timestamp and metadata."""
    timestamp: datetime
    symbol: str  # Related currency or 'GLOBAL'
    headline: str
    summary: str
    source: str
    url: Optional[str] = None
    
    def __post_init__(self):
        """Validate article data."""
        if self.timestamp is None:
            raise ValueError("[NEWS_MACRO] Article must have timestamp")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("[NEWS_MACRO] Article timestamp must be datetime")


@dataclass
class MacroFeatures:
    """Aggregated macro/news features for a timestamp."""
    timestamp: datetime
    symbol: str
    
    # Surprise scores: [-1, 1]
    # -1: much worse than forecast, +1: much better
    surprise_score: float = 0.0
    
    # Hawkishness: [-1, 1]
    # -1: strongly dovish (easing), +1: strongly hawkish (tightening)
    hawkishness_score: float = 0.0
    
    # Risk sentiment: [-1, 1]
    # -1: strong risk-off, +1: strong risk-on
    risk_sentiment_score: float = 0.0
    
    # Event importance: 0-3
    # 0: no events, 3: critical event
    event_importance: int = 0
    
    # Time since last major event (hours)
    hours_since_last_event: float = float('inf')
    
    # Number of macro events in lookback window
    macro_event_count: int = 0
    
    # Number of news articles in lookback window
    news_article_count: int = 0
    
    # Categories of events present (for feature richness)
    event_categories: List[str] = field(default_factory=list)
    
    # Overall macro state
    macro_news_state: str = "NEUTRAL"  # STRONG_RISK_ON, MILD_RISK_ON, NEUTRAL, MILD_RISK_OFF, STRONG_RISK_OFF


# =============================================================================
# SIMPLE NLP/SENTIMENT ANALYSIS
# =============================================================================

class SimpleNLPClassifier:
    """
    Simple, rule-based NLP classifier for macro sentiment and surprise.
    
    Uses keyword matching and patterns to classify:
    - Sentiment (hawkish/dovish)
    - Surprise (beat/miss/in-line)
    - Risk sentiment (risk-on/risk-off)
    
    NO machine learning, NO external models, deterministic and fast.
    """
    
    # Hawkish keywords (tightening, rate increases, inflation concerns)
    HAWKISH_KEYWORDS = {
        'hawkish', 'tightening', 'rate hike', 'rate increase', 'inflation',
        'higher rates', 'restrictive', 'hold', 'pause hikes', 'inflation sticky',
        'price pressures', 'wage growth', 'upside inflation risk', 'stronger',
        'robust', 'resilient', 'strong demand', 'tight labor'
    }
    
    # Dovish keywords (easing, rate cuts, deflation concerns)
    DOVISH_KEYWORDS = {
        'dovish', 'easing', 'rate cut', 'lower rates', 'deflation',
        'lower for longer', 'accommodative', 'pause', 'hold', 'soft landing',
        'below target', 'below-trend', 'weaker', 'slow', 'cooling',
        'unemployment risk', 'labor market weakness', 'downside risk'
    }
    
    # Risk-on keywords (positive sentiment, growth, confidence)
    RISK_ON_KEYWORDS = {
        'strong', 'beat', 'better', 'growth', 'expansion', 'confidence',
        'optimistic', 'rally', 'surge', 'boom', 'robust', 'upside', 'momentum',
        'recovery', 'positive', 'gains', 'higher', 'upgrade'
    }
    
    # Risk-off keywords (negative sentiment, contraction, fear)
    RISK_OFF_KEYWORDS = {
        'weak', 'miss', 'worse', 'contraction', 'recession', 'crisis',
        'pessimistic', 'decline', 'crash', 'slump', 'fragile', 'downside',
        'weakness', 'loss', 'lower', 'downgrade', 'shock', 'conflict',
        'geopolitical', 'sanctions', 'fear', 'uncertainty'
    }
    
    @staticmethod
    def classify_sentiment(text: str) -> float:
        """
        Classify sentiment from text.
        
        Returns: float in [-1, 1]
        -1: strongly dovish
        +1: strongly hawkish
        0: neutral
        """
        text_lower = text.lower()
        
        hawkish_count = sum(1 for kw in SimpleNLPClassifier.HAWKISH_KEYWORDS if kw in text_lower)
        dovish_count = sum(1 for kw in SimpleNLPClassifier.DOVISH_KEYWORDS if kw in text_lower)
        
        if hawkish_count == 0 and dovish_count == 0:
            return 0.0
        
        if hawkish_count == 0:
            return min(-1.0, -dovish_count / 3)
        if dovish_count == 0:
            return min(1.0, hawkish_count / 3)
        
        # Both present: net sentiment
        net = (hawkish_count - dovish_count) / max(hawkish_count + dovish_count, 1)
        return np.clip(net, -1.0, 1.0)
    
    @staticmethod
    def classify_risk_sentiment(text: str) -> float:
        """
        Classify overall risk sentiment (risk-on vs risk-off).
        
        Returns: float in [-1, 1]
        -1: strong risk-off
        +1: strong risk-on
        0: neutral
        """
        text_lower = text.lower()
        
        risk_on_count = sum(1 for kw in SimpleNLPClassifier.RISK_ON_KEYWORDS if kw in text_lower)
        risk_off_count = sum(1 for kw in SimpleNLPClassifier.RISK_OFF_KEYWORDS if kw in text_lower)
        
        if risk_on_count == 0 and risk_off_count == 0:
            return 0.0
        
        if risk_on_count == 0:
            return min(-1.0, -risk_off_count / 3)
        if risk_off_count == 0:
            return min(1.0, risk_on_count / 3)
        
        net = (risk_on_count - risk_off_count) / max(risk_on_count + risk_off_count, 1)
        return np.clip(net, -1.0, 1.0)
    
    @staticmethod
    def classify_surprise(actual: float, forecast: float) -> float:
        """
        Classify economic surprise.
        
        Returns: float in [-1, 1]
        -1: much worse than forecast
        +1: much better than forecast
        0: in-line with forecast
        """
        if forecast is None or forecast == 0:
            return 0.0
        
        surprise_pct = (actual - forecast) / abs(forecast)
        
        # Normalize to [-1, 1]: ±50% = ±1.0
        return np.clip(surprise_pct / 0.5, -1.0, 1.0)


# =============================================================================
# NEWS & MACRO ENGINE
# =============================================================================

class NewsMacroEngine:
    """
    AI/NLP-powered macro and news feature extraction engine.
    
    - Loads real-world macro events and news articles
    - Parses text using simple NLP (no ML models)
    - Generates time-aligned, causal numeric features
    - Integrates with MarketStateBuilder and evaluator
    
    STRICTLY enforces time-causality: no future data ever.
    """
    
    def __init__(self, symbol: str = 'USD', lookback_hours: int = 24, verbose: bool = False):
        """
        Initialize NewsMacroEngine.
        
        Args:
            symbol: Primary symbol to track (e.g., 'USD', 'EUR', 'GLOBAL')
            lookback_hours: Hours to lookback when aggregating features
            verbose: If True, log detailed information
        """
        self.symbol = symbol
        self.lookback_hours = lookback_hours
        self.verbose = verbose
        
        self.macro_events: List[MacroEvent] = []
        self.news_articles: List[NewsArticle] = []
        self.nlp_classifier = SimpleNLPClassifier()
        
        logger.info(f"[NEWS_MACRO] Engine initialized for {symbol}, lookback={lookback_hours}h")
    
    def load_event_calendar(self, csv_path: str) -> int:
        """
        Load macro events from CSV file.
        
        Expected CSV columns:
        - timestamp (YYYY-MM-DD HH:MM:SS)
        - symbol (USD, EUR, GBP, JPY, GLOBAL, etc.)
        - category (inflation, employment, rate_decision, gdp, trade, sentiment, geopolitical, earnings, other)
        - title (Event name)
        - description (Event details)
        - impact_level (0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL)
        - actual (optional, actual released value)
        - forecast (optional, consensus forecast)
        - previous (optional, previous value)
        
        Returns:
            Number of events loaded
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"[NEWS_MACRO] Event calendar not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = ['timestamp', 'symbol', 'category', 'title', 'description', 'impact_level']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"[NEWS_MACRO] Missing columns in event calendar: {missing}")
        
        # Parse timestamps and convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp (enforce ordering)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Verify no future timestamps (hard check)
        now = datetime.now()
        if (df['timestamp'] > now).any():
            logger.warning("[NEWS_MACRO] Event calendar contains future timestamps - using current time as cutoff")
            df = df[df['timestamp'] <= now]
        
        # Convert to MacroEvent objects
        for _, row in df.iterrows():
            try:
                category = MacroEventCategory[row['category'].upper()]
                impact = EventImpactLevel(int(row['impact_level']))
                
                event = MacroEvent(
                    timestamp=row['timestamp'],
                    symbol=str(row['symbol']).upper(),
                    category=category,
                    title=str(row['title']),
                    description=str(row['description']),
                    impact_level=impact,
                    actual=float(row['actual']) if pd.notna(row.get('actual')) else None,
                    forecast=float(row['forecast']) if pd.notna(row.get('forecast')) else None,
                    previous=float(row['previous']) if pd.notna(row.get('previous')) else None,
                )
                self.macro_events.append(event)
            except (ValueError, KeyError) as e:
                logger.warning(f"[NEWS_MACRO] Skipping invalid event: {row.to_dict()} - {e}")
                continue
        
        logger.info(f"[NEWS_MACRO] Loaded {len(self.macro_events)} macro events from {csv_path}")
        return len(self.macro_events)
    
    def load_news_articles(self, csv_path: str) -> int:
        """
        Load news articles from CSV file.
        
        Expected CSV columns:
        - timestamp (YYYY-MM-DD HH:MM:SS)
        - symbol (USD, EUR, GLOBAL, etc.)
        - headline (Article headline)
        - summary (Article summary/body)
        - source (News source)
        - url (optional, article URL)
        
        Returns:
            Number of articles loaded
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"[NEWS_MACRO] News file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = ['timestamp', 'symbol', 'headline', 'summary', 'source']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"[NEWS_MACRO] Missing columns in news file: {missing}")
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Verify no future timestamps
        now = datetime.now()
        if (df['timestamp'] > now).any():
            logger.warning("[NEWS_MACRO] News articles contain future timestamps - using current time as cutoff")
            df = df[df['timestamp'] <= now]
        
        # Convert to NewsArticle objects
        for _, row in df.iterrows():
            try:
                article = NewsArticle(
                    timestamp=row['timestamp'],
                    symbol=str(row['symbol']).upper(),
                    headline=str(row['headline']),
                    summary=str(row['summary']),
                    source=str(row['source']),
                    url=str(row['url']) if pd.notna(row.get('url')) else None,
                )
                self.news_articles.append(article)
            except (ValueError, KeyError) as e:
                logger.warning(f"[NEWS_MACRO] Skipping invalid article: {row.to_dict()} - {e}")
                continue
        
        logger.info(f"[NEWS_MACRO] Loaded {len(self.news_articles)} news articles from {csv_path}")
        return len(self.news_articles)
    
    def get_features_for_timestamp(
        self,
        timestamp: datetime,
        official_mode: bool = False
    ) -> MacroFeatures:
        """
        Get aggregated macro/news features for a given timestamp.
        
        STRICTLY time-causal: only uses events/news with timestamp <= target timestamp.
        
        Args:
            timestamp: Target timestamp
            official_mode: If True, enforce hard errors on any future data leakage
        
        Returns:
            MacroFeatures object with aggregated scores
        
        Raises:
            ValueError: If official_mode=True and any future data detected
        """
        # HARD CHECK: No future data ever
        if timestamp > datetime.now():
            if official_mode:
                raise ValueError(
                    f"[NEWS_MACRO] Official mode: Cannot get features for future timestamp {timestamp}"
                )
            logger.warning(f"[NEWS_MACRO] Requested timestamp {timestamp} is in future, using now()")
            timestamp = datetime.now()
        
        # Define lookback window
        lookback_start = timestamp - timedelta(hours=self.lookback_hours)
        
        # Get events in window (strict causality: <= timestamp)
        events_in_window = [
            e for e in self.macro_events
            if lookback_start <= e.timestamp <= timestamp
        ]
        
        # Get news in window
        news_in_window = [
            a for a in self.news_articles
            if lookback_start <= a.timestamp <= timestamp
        ]
        
        # Compute surprise score (if we have actual vs forecast events)
        surprise_scores = []
        for event in events_in_window:
            if event.actual is not None and event.forecast is not None:
                surprise = self.nlp_classifier.classify_surprise(event.actual, event.forecast)
                surprise_scores.append(surprise)
        
        surprise_score = np.mean(surprise_scores) if surprise_scores else 0.0
        
        # Compute hawkishness from event descriptions and news
        hawkish_scores = []
        for event in events_in_window:
            # Rate decisions have explicit sentiment
            if event.category == MacroEventCategory.RATE_DECISION:
                if 'hawk' in event.description.lower():
                    hawkish_scores.append(0.8)
                elif 'dove' in event.description.lower():
                    hawkish_scores.append(-0.8)
                else:
                    # Default: neutral
                    hawkish_scores.append(0.0)
            else:
                # For other events, classify description
                sentiment = self.nlp_classifier.classify_sentiment(event.description)
                hawkish_scores.append(sentiment)
        
        # Add news sentiment to hawkishness
        for article in news_in_window:
            sentiment = self.nlp_classifier.classify_sentiment(
                article.headline + " " + article.summary
            )
            hawkish_scores.append(sentiment)
        
        hawkishness_score = np.mean(hawkish_scores) if hawkish_scores else 0.0
        
        # Compute risk sentiment
        risk_scores = []
        for event in events_in_window:
            risk = self.nlp_classifier.classify_risk_sentiment(event.description)
            risk_scores.append(risk)
        
        for article in news_in_window:
            risk = self.nlp_classifier.classify_risk_sentiment(
                article.headline + " " + article.summary
            )
            risk_scores.append(risk)
        
        risk_sentiment_score = np.mean(risk_scores) if risk_scores else 0.0
        
        # Get maximum impact level in window
        event_importance = max(
            [e.impact_level.value for e in events_in_window],
            default=0
        )
        
        # Time since last major event
        major_events = [e for e in events_in_window if e.impact_level.value >= 2]
        if major_events:
            last_major = max(major_events, key=lambda e: e.timestamp)
            hours_since = (timestamp - last_major.timestamp).total_seconds() / 3600
        else:
            hours_since = float('inf')
        
        # Event categories present
        categories = list(set(e.category.value for e in events_in_window))
        
        # Determine macro news state
        macro_state = self._compute_macro_state(
            risk_sentiment_score,
            hawkishness_score,
            event_importance,
            hours_since
        )
        
        features = MacroFeatures(
            timestamp=timestamp,
            symbol=self.symbol,
            surprise_score=float(np.clip(surprise_score, -1.0, 1.0)),
            hawkishness_score=float(np.clip(hawkishness_score, -1.0, 1.0)),
            risk_sentiment_score=float(np.clip(risk_sentiment_score, -1.0, 1.0)),
            event_importance=int(event_importance),
            hours_since_last_event=float(hours_since),
            macro_event_count=len(events_in_window),
            news_article_count=len(news_in_window),
            event_categories=categories,
            macro_news_state=macro_state,
        )
        
        if self.verbose:
            logger.info(
                f"[NEWS_MACRO] Features @ {timestamp}: "
                f"risk={risk_sentiment_score:.2f}, "
                f"hawk={hawkishness_score:.2f}, "
                f"surprise={surprise_score:.2f}, "
                f"state={macro_state}"
            )
        
        return features
    
    def _compute_macro_state(
        self,
        risk_sentiment: float,
        hawkishness: float,
        event_importance: int,
        hours_since_last_event: float
    ) -> str:
        """
        Compute overall macro news state.
        
        Returns: One of:
        - STRONG_RISK_ON
        - MILD_RISK_ON
        - NEUTRAL
        - MILD_RISK_OFF
        - STRONG_RISK_OFF
        """
        # Decay: recent events have more weight
        event_recency_weight = max(0.0, 1.0 - hours_since_last_event / (24 * self.lookback_hours))
        
        # Weighted sentiment
        weighted_risk = risk_sentiment * (0.6 + 0.4 * event_recency_weight)
        
        if weighted_risk >= 0.6:
            return "STRONG_RISK_ON"
        elif weighted_risk >= 0.2:
            return "MILD_RISK_ON"
        elif weighted_risk <= -0.6:
            return "STRONG_RISK_OFF"
        elif weighted_risk <= -0.2:
            return "MILD_RISK_OFF"
        else:
            return "NEUTRAL"
    
    def build_macro_features(self) -> pd.DataFrame:
        """
        Build a time-series of macro features across all timestamped events.
        
        Returns:
            DataFrame with columns:
            - timestamp
            - symbol
            - surprise_score
            - hawkishness_score
            - risk_sentiment_score
            - event_importance
            - hours_since_last_event
            - macro_event_count
            - news_article_count
            - macro_news_state
        """
        all_timestamps = set()
        for event in self.macro_events:
            all_timestamps.add(event.timestamp)
        for article in self.news_articles:
            all_timestamps.add(article.timestamp)
        
        if not all_timestamps:
            logger.warning("[NEWS_MACRO] No events or articles loaded - returning empty DataFrame")
            return pd.DataFrame()
        
        # Sort and create feature row for each timestamp
        sorted_timestamps = sorted(all_timestamps)
        rows = []
        
        for ts in sorted_timestamps:
            features = self.get_features_for_timestamp(ts)
            rows.append({
                'timestamp': features.timestamp,
                'symbol': features.symbol,
                'surprise_score': features.surprise_score,
                'hawkishness_score': features.hawkishness_score,
                'risk_sentiment_score': features.risk_sentiment_score,
                'event_importance': features.event_importance,
                'hours_since_last_event': features.hours_since_last_event,
                'macro_event_count': features.macro_event_count,
                'news_article_count': features.news_article_count,
                'macro_news_state': features.macro_news_state,
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"[NEWS_MACRO] Built feature timeseries with {len(df)} rows")
        return df
    
    def validate_time_causality(self) -> Tuple[bool, List[str]]:
        """
        Validate that all events and articles are properly time-ordered and causally consistent.
        
        Returns:
            (is_valid, warnings_list)
        """
        warnings = []
        
        # Check events ordered
        for i in range(len(self.macro_events) - 1):
            if self.macro_events[i].timestamp > self.macro_events[i + 1].timestamp:
                warnings.append(
                    f"Macro events not strictly ordered: "
                    f"{self.macro_events[i].timestamp} > {self.macro_events[i+1].timestamp}"
                )
        
        # Check articles ordered
        for i in range(len(self.news_articles) - 1):
            if self.news_articles[i].timestamp > self.news_articles[i + 1].timestamp:
                warnings.append(
                    f"News articles not strictly ordered: "
                    f"{self.news_articles[i].timestamp} > {self.news_articles[i+1].timestamp}"
                )
        
        # Check no duplicates
        event_ts = [e.timestamp for e in self.macro_events]
        if len(event_ts) != len(set(event_ts)):
            warnings.append("Duplicate macro event timestamps detected")
        
        article_ts = [a.timestamp for a in self.news_articles]
        if len(article_ts) != len(set(article_ts)):
            warnings.append("Duplicate news article timestamps detected")
        
        # Check no future data
        now = datetime.now()
        future_events = [e for e in self.macro_events if e.timestamp > now]
        if future_events:
            warnings.append(f"Future macro events detected: {len(future_events)}")
        
        future_articles = [a for a in self.news_articles if a.timestamp > now]
        if future_articles:
            warnings.append(f"Future news articles detected: {len(future_articles)}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def export_features_to_json(self, output_path: str) -> None:
        """Export all macro features to JSON file."""
        features_df = self.build_macro_features()
        
        # Convert to JSON-serializable format
        records = []
        for _, row in features_df.iterrows():
            record = {
                'timestamp': row['timestamp'].isoformat(),
                'symbol': row['symbol'],
                'surprise_score': float(row['surprise_score']),
                'hawkishness_score': float(row['hawkishness_score']),
                'risk_sentiment_score': float(row['risk_sentiment_score']),
                'event_importance': int(row['event_importance']),
                'hours_since_last_event': float(row['hours_since_last_event']),
                'macro_event_count': int(row['macro_event_count']),
                'news_article_count': int(row['news_article_count']),
                'macro_news_state': row['macro_news_state'],
            }
            records.append(record)
        
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"[NEWS_MACRO] Exported {len(records)} feature records to {output_path}")


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_macro_features_into_state(
    macro_engine: NewsMacroEngine,
    timestamp: datetime,
    existing_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integrate macro features into an existing market state dictionary.
    
    Args:
        macro_engine: NewsMacroEngine instance
        timestamp: Current timestamp
        existing_state: Existing state dictionary
    
    Returns:
        Updated state dictionary with macro features
    """
    macro_features = macro_engine.get_features_for_timestamp(timestamp)
    
    # Add macro features to state
    enhanced_state = existing_state.copy()
    enhanced_state['macro_news_features'] = {
        'surprise_score': macro_features.surprise_score,
        'hawkishness_score': macro_features.hawkishness_score,
        'risk_sentiment_score': macro_features.risk_sentiment_score,
        'event_importance': macro_features.event_importance,
        'hours_since_last_event': macro_features.hours_since_last_event,
        'macro_event_count': macro_features.macro_event_count,
        'news_article_count': macro_features.news_article_count,
        'macro_news_state': macro_features.macro_news_state,
    }
    
    return enhanced_state


# =============================================================================
# EXAMPLE USAGE (for documentation)
# =============================================================================

if __name__ == '__main__':
    """
    Example: Create and use NewsMacroEngine.
    
    python analytics/news_macro_engine.py
    """
    
    # Create engine
    engine = NewsMacroEngine(symbol='USD', lookback_hours=24, verbose=True)
    
    print("[EXAMPLE] NewsMacroEngine initialized")
    print("          Use load_event_calendar() and load_news_articles() to add data")
    print("          Then call get_features_for_timestamp() or build_macro_features()")
