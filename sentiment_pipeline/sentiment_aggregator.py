import os
import sqlite3
from datetime import datetime, timedelta, timezone

# --- Configuration ---
# Use the same project directory and DB name as the fetcher
PROJECT_DIR = os.environ.get(
    "PROJECT_DIR", "C:\\Users\\Admin\\Documents\\MyProjectFolder"
)
DB_NAME = os.path.join(PROJECT_DIR, "news_database.db")


def generate_report():
    """Connects to the DB, calculates metrics, and prints a summary report."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Define the 24-hour window
        time_window = datetime.now(timezone.utc) - timedelta(hours=24)
        time_filter = time_window.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Fetch all analyzed scores within the last 24 hours
        cursor.execute(
            """
            SELECT headline, ai_sentiment_score, source
            FROM news_data
            WHERE ai_sentiment_score IS NOT NULL AND timestamp > ?
            ORDER BY ai_sentiment_score DESC
        """,
            (time_filter,),
        )

        results = cursor.fetchall()

        if not results:
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] --- SENTIMENT REPORT (24H) ---"
            )
            print("No analyzed articles found in the last 24 hours.")
            print("--------------------------------------------------")
            return

        total_articles = len(results)
        total_score = sum(r[1] for r in results)

        # Calculate metrics
        avg_score = total_score / total_articles if total_articles else 0
        positive_count = sum(1 for r in results if r[1] > 0.1)
        negative_count = sum(1 for r in results if r[1] < -0.1)
        neutral_count = total_articles - positive_count - negative_count

        # --- Print the Summary ---
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] --- TRADING SENTIMENT REPORT (24H) ---"
        )
        print(f"Total Analyzed Articles: {total_articles}")
        print(f"Time Window Start: {time_filter}")
        print("--------------------------------------------------")
        print(f"1. Aggregate Sentiment Score (Avg): {avg_score:.3f}")

        if avg_score > 0.1:
            market_mood = "BULLISH (Upbeat)"
        elif avg_score < -0.1:
            market_mood = "BEARISH (Caution)"
        else:
            market_mood = "NEUTRAL (HOLD)"

        print(f"2. Implied Market Mood: {market_mood}")
        print(f"3. Article Breakdown:")
        print(f"   - Positive (>0.1): {positive_count}")
        print(f"   - Negative (<-0.1): {negative_count}")
        print(f"   - Neutral (Other): {neutral_count}")
        print("--------------------------------------------------")

        # --- Print Top/Bottom Headlines ---
        print("4. Top 3 Bullish Headlines:")
        top_bullish = results[:3]
        for i, (headline, score, source) in enumerate(top_bullish):
            print(f"   {i+1}. [{source}] ({score:.2f}) {headline[:80]}...")

        print("\n5. Top 3 Bearish Headlines:")
        top_bearish = results[-3:][::-1]  # Reverse the order for worst first
        for i, (headline, score, source) in enumerate(top_bearish):
            print(f"   {i+1}. [{source}] ({score:.2f}) {headline[:80]}...")

        print("--------------------------------------------------")

    except sqlite3.Error as e:
        print(f"An error occurred accessing the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    generate_report()
