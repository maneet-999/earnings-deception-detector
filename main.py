"""
main.py — run the full pipeline for a list of tickers.

Usage:
    python main.py --tickers AAPL MSFT TSLA --start-year 2016
    python main.py --train          # train model after scraping
    python main.py --dashboard      # launch Streamlit dashboard
"""

import argparse
import logging
import time

from database import init_db
from transcript_scraper import TranscriptScraper
from financial_fetcher import FinancialFetcher
from pipeline import FeaturePipeline
from database import SessionLocal, Transcript
from trainer import DeceptionModelTrainer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Tickers known to have had restatements — good for validating your pipeline
VALIDATION_TICKERS = [
    "LCNHF",   # Luckin Coffee — 2020 fraud
    "WYY",     # WideOpenWest — restatement 2019
    "MFAC",    # Medallion Financial — accounting issues
]

EXAMPLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "BAC", "GS",
]


def run_scraping(tickers: list[str], start_year: int):
    scraper  = TranscriptScraper()
    fetcher  = FinancialFetcher()

    for ticker in tickers:
        log.info(f"\n{'='*50}")
        log.info(f"Processing {ticker}")
        log.info(f"{'='*50}")

        # 1. Scrape transcripts
        transcripts = scraper.fetch_bulk(ticker, start_year=start_year)
        for t in transcripts:
            scraper.save(t)

        # 2. Pull financials + restatement flags
        fin_df = fetcher.get_quarterly_financials(ticker, start_year=start_year)
        restatements = fetcher.get_restatement_flags(ticker)
        fetcher.save_financials(ticker, fin_df, restatements)

        time.sleep(2)

    log.info("\nScraping complete.")


def run_feature_extraction():
    db = SessionLocal()
    pipeline = FeaturePipeline()

    try:
        unparsed = (
            db.query(Transcript)
            .filter(Transcript.raw_text != None)
            .all()
        )
        log.info(f"Extracting features for {len(unparsed)} transcripts")

        for t in unparsed:
            try:
                vector = pipeline.run(t.id)
                if vector:
                    pipeline.save(vector)
            except Exception as e:
                log.error(f"Feature extraction failed for transcript {t.id}: {e}")
    finally:
        db.close()


def run_training():
    trainer = DeceptionModelTrainer()

    log.info("Loading features...")
    df = trainer.load_features()

    if df.empty:
        log.error("No labelled features found.")
        return

    log.info("Training final model directly (small dataset)...")
    model = trainer.train_final_model(df)
    log.info("Model saved to models/deception_model.json")

def main():
    parser = argparse.ArgumentParser(description="Earnings Call Deception Detector")
    parser.add_argument("--tickers", nargs="+", default=EXAMPLE_TICKERS[:5])
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--scrape",    action="store_true")
    parser.add_argument("--features",  action="store_true")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--all",       action="store_true")
    args = parser.parse_args()

    init_db()

    if args.all or args.scrape:
        run_scraping(args.tickers, args.start_year)

    if args.all or args.features:
        run_feature_extraction()

    if args.all or args.train:
        run_training()


if __name__ == "__main__":
    main()
