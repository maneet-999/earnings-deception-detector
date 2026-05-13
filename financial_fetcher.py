"""
EDGAR & yfinance financial data scraper.

Two jobs:
1. Pull quarterly financials (revenue, margins, EPS) via yfinance
2. Pull restatement flags from EDGAR — these are your ground truth labels

Usage:
    fetcher = FinancialFetcher()
    df = fetcher.get_quarterly_financials("AAPL", start_year=2014)
    restated = fetcher.get_restatement_flags("AAPL")
"""

import time
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from database import SessionLocal, Company, Quarterly

log = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_HEADERS = {
    "User-Agent": "earnings-detector research@example.com",  # SEC requires contact info
    "Accept-Encoding": "gzip, deflate",
}


class FinancialFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(EDGAR_HEADERS)

    # ------------------------------------------------------------------
    # Quarterly financials via yfinance
    # ------------------------------------------------------------------

    def get_quarterly_financials(
        self, ticker: str, start_year: int = 2014
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per quarter containing:
        revenue, gross_margin, eps_actual, eps_estimate, eps_surprise,
        revenue_growth_yoy, fiscal_year, quarter
        """
        log.info(f"Fetching financials for {ticker}")
        stock = yf.Ticker(ticker)

        # yfinance returns quarterly income statement
        income = stock.quarterly_income_stmt
        if income is None or income.empty:
            log.warning(f"No income data for {ticker}")
            return pd.DataFrame()

        income = income.T.copy()
        income.index = pd.to_datetime(income.index)
        income = income.sort_index()

        records = []
        for date, row in income.iterrows():
            if date.year < start_year:
                continue

            revenue = row.get("Total Revenue", None)
            cogs = row.get("Cost Of Revenue", None)
            gross_profit = row.get("Gross Profit", None)

            gross_margin = None
            if gross_profit is not None and revenue and revenue != 0:
                gross_margin = gross_profit / revenue

            # EPS — pull from earnings history
            eps_data = self._get_eps_surprise(stock, date)

            records.append({
                "ticker": ticker.upper(),
                "date": date,
                "fiscal_year": date.year,
                "quarter": (date.month - 1) // 3 + 1,
                "revenue": revenue,
                "gross_margin": gross_margin,
                "eps_actual": eps_data.get("actual"),
                "eps_estimate": eps_data.get("estimate"),
                "eps_surprise": eps_data.get("surprise"),
            })

        df = pd.DataFrame(records)

        # YoY revenue growth
        if not df.empty and "revenue" in df.columns:
            df = df.sort_values("date")
            df["revenue_growth"] = df["revenue"].pct_change(periods=4)

        return df

    def _get_eps_surprise(self, stock: yf.Ticker, date: pd.Timestamp) -> dict:
        """Match EPS estimate vs actual for a given quarter date."""
        try:
            earnings = stock.earnings_dates
            if earnings is None or earnings.empty:
                return {}
            # Find the closest earnings date within 14 days
            earnings.index = pd.to_datetime(earnings.index)
            diff = abs(earnings.index - date)
            closest_idx = diff.argmin()
            if diff[closest_idx].days > 14:
                return {}
            row = earnings.iloc[closest_idx]
            actual = row.get("Reported EPS")
            estimate = row.get("EPS Estimate")
            surprise = None
            if actual is not None and estimate is not None and estimate != 0:
                surprise = (actual - estimate) / abs(estimate)
            return {"actual": actual, "estimate": estimate, "surprise": surprise}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Restatement flags from EDGAR (ground truth labels)
    # ------------------------------------------------------------------

    def get_restatement_flags(self, ticker: str) -> list[dict]:
        """
        Pull 8-K filings from EDGAR for a company and flag quarters
        where an earnings restatement was announced.

        An 8-K Item 4.02 specifically signals a non-reliance on
        previously issued financial statements — this is your gold label.
        """
        cik = self._get_cik(ticker)
        if not cik:
            log.warning(f"No CIK found for {ticker}")
            return []

        filings_url = (
            f"{EDGAR_BASE}/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}"
            f"&type=8-K&dateb=&owner=include&count=100&search_text="
        )
        try:
            resp = self.session.get(filings_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"EDGAR 8-K search failed for {ticker}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        restatements = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            filing_date_text = cells[3].get_text(strip=True)
            link_tag = cells[1].find("a")
            if not link_tag:
                continue

            filing_url = "https://www.sec.gov" + link_tag["href"]

            # Check if this 8-K contains Item 4.02 (non-reliance on financials)
            if self._is_restatement_filing(filing_url):
                try:
                    filing_date = datetime.strptime(filing_date_text, "%Y-%m-%d")
                except ValueError:
                    continue
                restatements.append({
                    "ticker": ticker.upper(),
                    "restatement_date": filing_date,
                    "filing_url": filing_url,
                })
            time.sleep(0.3)  # Be polite to EDGAR

        log.info(f"Found {len(restatements)} restatement filings for {ticker}")
        return restatements

    def _is_restatement_filing(self, filing_index_url: str) -> bool:
        """Check if an 8-K filing index contains a restatement notice."""
        try:
            resp = self.session.get(filing_index_url, timeout=10)
            text = resp.text.upper()
            # Item 4.02 = Non-Reliance on Previously Issued Financial Statements
            return "ITEM 4.02" in text or "NON-RELIANCE" in text
        except Exception:
            return False

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Look up SEC CIK number from ticker symbol."""
        url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?company=&CIK={ticker}&type=8-K&action=getcompany"
        try:
            resp = self.session.get(url, timeout=10)
            match = __import__("re").search(r"CIK=(\d+)", resp.url)
            if match:
                return match.group(1).zfill(10)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Save to database
    # ------------------------------------------------------------------

    def save_financials(
        self,
        ticker: str,
        financials_df: pd.DataFrame,
        restatement_flags: list[dict],
    ):
        """Merge financials and restatement flags, then persist."""
        if financials_df.empty:
            return

        db = SessionLocal()
        try:
            company = db.query(Company).filter_by(ticker=ticker).first()
            if not company:
                company = Company(ticker=ticker)
                db.add(company)
                db.flush()

            # Build a set of (year, quarter) pairs that had restatements
            restated_periods = set()
            for r in restatement_flags:
                d = r["restatement_date"]
                restated_periods.add((d.year, (d.month - 1) // 3 + 1))

            for _, row in financials_df.iterrows():
                fy = int(row["fiscal_year"])
                q = int(row["quarter"])

                existing = db.query(Quarterly).filter_by(
                    ticker=ticker, fiscal_year=fy, quarter=q
                ).first()
                if existing:
                    continue

                q_obj = Quarterly(
                    company_id=company.id,
                    ticker=ticker,
                    fiscal_year=fy,
                    quarter=q,
                    revenue=row.get("revenue"),
                    gross_margin=row.get("gross_margin"),
                    eps_actual=row.get("eps_actual"),
                    eps_estimate=row.get("eps_estimate"),
                    eps_surprise=row.get("eps_surprise"),
                    revenue_growth=row.get("revenue_growth"),
                    had_restatement=(fy, q) in restated_periods,
                )
                db.add(q_obj)

            db.commit()
            log.info(f"Saved financials for {ticker}")

        except Exception as e:
            db.rollback()
            log.error(f"Error saving financials for {ticker}: {e}")
            raise
        finally:
            db.close()
