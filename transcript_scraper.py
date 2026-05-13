"""
Transcript scraper — Motley Fool & Seeking Alpha.

Motley Fool publishes free earnings call transcripts at predictable URLs.
Seeking Alpha requires a session token (set SEEKING_ALPHA_TOKEN in .env).

Usage:
    scraper = TranscriptScraper()
    transcript = scraper.fetch("AAPL", 2024, 1)   # Q1 FY2024
    scraper.save(transcript)
"""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from database import SessionLocal, Company, Transcript, SpeakerTurn

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Polite crawl delay in seconds
CRAWL_DELAY = 2.5


@dataclass
class RawTranscript:
    ticker: str
    fiscal_year: int
    quarter: int
    call_date: Optional[datetime]
    source_url: str
    raw_text: str
    speaker_turns: list = field(default_factory=list)


class TranscriptScraper:
    """
    Scrapes earnings call transcripts from Motley Fool.
    Falls back to Seeking Alpha if MF returns no result.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sa_token = os.getenv("SEEKING_ALPHA_TOKEN", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[RawTranscript]:
        """Fetch a transcript, trying Motley Fool first then Seeking Alpha."""
        log.info(f"Fetching {ticker} Q{quarter} FY{year}")

        result = self._fetch_motley_fool(ticker, year, quarter)
        if result:
            return result

        log.info(f"MF miss — trying Seeking Alpha for {ticker} Q{quarter} FY{year}")
        return self._fetch_seeking_alpha(ticker, year, quarter)

    def fetch_bulk(self, ticker: str, start_year: int = 2014) -> list[RawTranscript]:
        """Fetch all available quarters for a ticker since start_year."""
        results = []
        current_year = datetime.now().year
        for year in range(start_year, current_year + 1):
            for quarter in range(1, 5):
                result = self.fetch(ticker, year, quarter)
                if result:
                    results.append(result)
                time.sleep(CRAWL_DELAY)
        return results

    def save(self, raw: RawTranscript) -> int:
        """Persist a transcript to the database. Returns transcript ID."""
        db = SessionLocal()
        try:
            company = db.query(Company).filter_by(ticker=raw.ticker).first()
            if not company:
                company = Company(ticker=raw.ticker)
                db.add(company)
                db.flush()

            existing = db.query(Transcript).filter_by(
                ticker=raw.ticker,
                fiscal_year=raw.fiscal_year,
                quarter=raw.quarter,
            ).first()

            if existing:
                log.info(f"Already exists: {raw.ticker} Q{raw.quarter} {raw.fiscal_year}")
                return existing.id

            transcript = Transcript(
                company_id=company.id,
                ticker=raw.ticker,
                fiscal_year=raw.fiscal_year,
                quarter=raw.quarter,
                call_date=raw.call_date,
                source_url=raw.source_url,
                raw_text=raw.raw_text,
                is_parsed=len(raw.speaker_turns) > 0,
            )
            db.add(transcript)
            db.flush()

            for turn in raw.speaker_turns:
                db.add(SpeakerTurn(transcript_id=transcript.id, **turn))

            db.commit()
            log.info(f"Saved {raw.ticker} Q{raw.quarter} {raw.fiscal_year} — {len(raw.speaker_turns)} turns")
            return transcript.id

        except Exception as e:
            db.rollback()
            log.error(f"DB error saving {raw.ticker}: {e}")
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Motley Fool
    # ------------------------------------------------------------------

    def _fetch_motley_fool(self, ticker: str, year: int, quarter: int) -> Optional[RawTranscript]:
        """
        Motley Fool transcript URLs follow the pattern:
        /earnings/call-transcripts/{year}/{month}/{day}/{ticker-slug}-q{q}-{year}-earnings...
        We use their search endpoint to find the right URL.
        """
        search_url = (
            f"https://www.fool.com/earnings/call-transcripts/"
            f"?symbol={ticker.upper()}"
        )
        try:
            resp = self.session.get(search_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"MF search failed for {ticker}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        target_url = self._find_mf_transcript_url(soup, ticker, year, quarter)
        if not target_url:
            return None

        time.sleep(CRAWL_DELAY)
        return self._parse_motley_fool_page(target_url, ticker, year, quarter)

    def _find_mf_transcript_url(
        self, soup: BeautifulSoup, ticker: str, year: int, quarter: int
    ) -> Optional[str]:
        """Find the best matching transcript link from MF search results."""
        quarter_patterns = [f"q{quarter}-{year}", f"q{quarter}{year}"]

        for link in soup.find_all("a", href=True):
            href = link["href"].lower()
            if "earnings/call-transcripts" in href:
                for pat in quarter_patterns:
                    if pat in href and ticker.lower() in href:
                        return "https://www.fool.com" + link["href"]
        return None

    def _parse_motley_fool_page(
        self, url: str, ticker: str, year: int, quarter: int
    ) -> Optional[RawTranscript]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"MF page fetch failed {url}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        article = soup.find("div", class_=re.compile(r"article-body|transcript"))
        if not article:
            article = soup.find("article")
        if not article:
            return None

        raw_text = article.get_text(separator="\n")
        call_date = self._extract_date(soup)
        speaker_turns = self._parse_speaker_turns(raw_text)

        return RawTranscript(
            ticker=ticker.upper(),
            fiscal_year=year,
            quarter=quarter,
            call_date=call_date,
            source_url=url,
            raw_text=raw_text,
            speaker_turns=speaker_turns,
        )

    # ------------------------------------------------------------------
    # Seeking Alpha fallback
    # ------------------------------------------------------------------

    def _fetch_seeking_alpha(
        self, ticker: str, year: int, quarter: int
    ) -> Optional[RawTranscript]:
        """
        Seeking Alpha API — requires a bearer token in .env.
        Free SA accounts give limited API calls; use sparingly.
        """
        if not self.sa_token:
            log.warning("No SEEKING_ALPHA_TOKEN set — skipping SA fallback")
            return None

        api_url = (
            f"https://seekingalpha.com/api/v3/earnings_call_transcripts"
            f"?filter[ticker]={ticker.upper()}"
            f"&filter[year]={year}"
            f"&filter[quarter]={quarter}"
        )
        try:
            resp = self.session.get(
                api_url,
                headers={**HEADERS, "Authorization": f"Bearer {self.sa_token}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"SA API failed for {ticker}: {e}")
            return None

        if not data.get("data"):
            return None

        item = data["data"][0]
        raw_text = item.get("attributes", {}).get("content", "")
        if not raw_text:
            return None

        speaker_turns = self._parse_speaker_turns(raw_text)
        return RawTranscript(
            ticker=ticker.upper(),
            fiscal_year=year,
            quarter=quarter,
            call_date=None,
            source_url=api_url,
            raw_text=raw_text,
            speaker_turns=speaker_turns,
        )

    # ------------------------------------------------------------------
    # Parser — speaker turn extraction
    # ------------------------------------------------------------------

    def _parse_speaker_turns(self, text: str) -> list[dict]:
        """
        Earnings transcripts have a consistent structure:
            Speaker Name (Role/Company):
            [paragraph of speech]

        We detect speaker headers with a regex and split accordingly.
        """
        # Matches lines like:
        #   "Satya Nadella -- Chief Executive Officer"
        #   "John Smith (Analyst, Goldman Sachs)"
        #   "OPERATOR"
        speaker_re = re.compile(
            r"^([A-Z][A-Za-z\s\.\-']+)"       # name
            r"(?:\s*(?:--|—|:|\()"             # separator
            r"([A-Za-z\s,\.]+))?",             # optional role
            re.MULTILINE,
        )

        lines = text.split("\n")
        turns = []
        current_speaker = None
        current_role = None
        current_section = "PREPARED"
        buffer = []

        qa_markers = re.compile(
            r"questions?\s+and\s+answers?|q\s*&\s*a\s+session", re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if qa_markers.search(line):
                current_section = "QA"
                continue

            m = speaker_re.match(line)
            if m and len(line) < 120:
                # Flush previous buffer
                if current_speaker and buffer:
                    turns.append(self._make_turn(
                        current_speaker, current_role,
                        current_section, len(turns), buffer
                    ))
                    buffer = []

                current_speaker = m.group(1).strip()
                current_role = self._classify_role(
                    current_speaker, m.group(2) or ""
                )
            else:
                buffer.append(line)

        # Flush last speaker
        if current_speaker and buffer:
            turns.append(self._make_turn(
                current_speaker, current_role,
                current_section, len(turns), buffer
            ))

        return turns

    def _make_turn(
        self, name: str, role: str, section: str, index: int, lines: list
    ) -> dict:
        return {
            "speaker_name": name,
            "speaker_role": role,
            "section": section,
            "turn_index": index,
            "text": " ".join(lines).strip(),
        }

    def _classify_role(self, name: str, role_hint: str) -> str:
        """Classify speaker into CEO | CFO | ANALYST | OPERATOR | OTHER."""
        combined = f"{name} {role_hint}".upper()

        if any(t in combined for t in ["CEO", "CHIEF EXECUTIVE", "PRESIDENT"]):
            return "CEO"
        if any(t in combined for t in ["CFO", "CHIEF FINANCIAL", "FINANCE OFFICER"]):
            return "CFO"
        if any(t in combined for t in [
            "ANALYST", "RESEARCH", "GOLDMAN", "MORGAN", "JPMORGAN",
            "BARCLAYS", "CITI", "WELLS", "BOFA", "UBS", "PIPER"
        ]):
            return "ANALYST"
        if "OPERATOR" in combined:
            return "OPERATOR"
        return "OTHER"

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Try to pull a date from common meta tags."""
        for attr in ["article:published_time", "datePublished", "date"]:
            tag = soup.find("meta", property=attr) or soup.find("meta", itemprop=attr)
            if tag and tag.get("content"):
                try:
                    return datetime.fromisoformat(tag["content"][:19])
                except ValueError:
                    pass
        return None
