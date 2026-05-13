import requests
import re
import time
from database import SessionLocal, Company, Transcript, SpeakerTurn

HEADERS = {"User-Agent": "earnings-detector research@gmail.com"}

def get_all_ciks():
    """Get ticker -> CIK mapping from EDGAR."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    data = resp.json()
    mapping = {}
    for val in data.values():
        mapping[val["ticker"]] = str(val["cik_str"]).zfill(10)
    return mapping

def get_transcript(ticker, cik, year, quarter):
    """Find earnings call transcript from EDGAR 8-K filings."""
    
    quarter_dates = {
        1: ("01-01", "04-30"),
        2: ("04-01", "07-31"),
        3: ("07-01", "10-31"),
        4: ("10-01", "12-31"),
    }
    start, end = quarter_dates[quarter]
    start_date = f"{year}-{start}"
    end_date = f"{year}-{end}"

    # Get all filings for this company
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms     = filings.get("form", [])
    dates     = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])

    print(f"  Scanning {len(forms)} filings for {ticker}...")

    for form, date, accession in zip(forms, dates, accessions):
        if form != "8-K":
            continue
        if not (start_date <= date <= end_date):
            continue

        print(f"  Checking 8-K filed {date}...")
        acc_clean = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"

        try:
            resp = requests.get(index_url, headers=HEADERS, timeout=10)
            time.sleep(0.5)

            # Find all document links in this filing
            links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.(htm|txt))"',
                resp.text, re.IGNORECASE
            )

            for link, ext in links:
                link_lower = link.lower()
                # Look for exhibit 99 — that's where transcripts live
                if any(x in link_lower for x in ["ex99", "ex-99", "exhibit99", "transcript"]):
                    doc_url = "https://www.sec.gov" + link
                    doc_resp = requests.get(doc_url, headers=HEADERS, timeout=15)
                    time.sleep(0.5)

                    # Strip HTML tags
                    text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Must be long enough and have Q&A markers
                    if len(text) > 3000 and any(
                        w in text.lower() for w in ["question", "analyst", "operator", "q&a"]
                    ):
                        print(f"  Found transcript! Length: {len(text)} chars")
                        return text, doc_url

        except Exception as e:
            print(f"  Error checking filing: {e}")
            continue

    return None, None


def parse_turns(text):
    """Extract speaker turns from transcript."""
    turns = []
    current_speaker = "UNKNOWN"
    current_role = "OTHER"
    current_section = "PREPARED"
    buffer = []
    index = 0

    role_map = {
        "CEO": "CEO", "CHIEF EXECUTIVE": "CEO", "PRESIDENT": "CEO",
        "CFO": "CFO", "CHIEF FINANCIAL": "CFO",
        "ANALYST": "ANALYST", "OPERATOR": "OPERATOR",
    }

    for line in text.split("."):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        if any(w in line.lower() for w in ["question and answer", "q&a session", "open for questions"]):
            current_section = "QA"

        # Detect speaker change — short lines with role keywords
        found_role = None
        line_upper = line.upper()
        for keyword, role in role_map.items():
            if keyword in line_upper and len(line) < 120:
                found_role = role
                break

        if found_role and buffer:
            turns.append({
                "speaker_name": current_speaker[:100],
                "speaker_role": current_role,
                "section": current_section,
                "turn_index": index,
                "text": " ".join(buffer).strip(),
            })
            index += 1
            buffer = []
            current_speaker = line[:100]
            current_role = found_role
        else:
            if len(line) > 15:
                buffer.append(line)

    if buffer:
        turns.append({
            "speaker_name": current_speaker[:100],
            "speaker_role": current_role,
            "section": current_section,
            "turn_index": index,
            "text": " ".join(buffer).strip(),
        })

    return turns


def save_transcript(ticker, year, quarter, text, url):
    """Save to database."""
    db = SessionLocal()
    try:
        company = db.query(Company).filter_by(ticker=ticker).first()
        if not company:
            company = Company(ticker=ticker)
            db.add(company)
            db.flush()

        existing = db.query(Transcript).filter_by(
            ticker=ticker, fiscal_year=year, quarter=quarter
        ).first()
        if existing:
            print(f"  Already saved: {ticker} Q{quarter} {year}")
            return existing.id

        turns = parse_turns(text)
        transcript = Transcript(
            company_id=company.id,
            ticker=ticker,
            fiscal_year=year,
            quarter=quarter,
            source_url=url,
            raw_text=text,
            is_parsed=len(turns) > 0,
        )
        db.add(transcript)
        db.flush()

        for turn in turns:
            db.add(SpeakerTurn(transcript_id=transcript.id, **turn))

        db.commit()
        print(f"  Saved {ticker} Q{quarter} {year} — {len(turns)} speaker turns")
        return transcript.id

    except Exception as e:
        db.rollback()
        print(f"  DB error: {e}")
        return None
    finally:
        db.close()


if __name__ == "__main__":
    print("Loading CIK map from EDGAR...")
    cik_map = get_all_ciks()

    tickers = [
    "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA",
    "AAPL", "META", "JPM", "V", "WMT",
    "NFLX", "PYPL", "INTC", "AMD", "CRM"
    ]
    quarters = [
    (2019,1),(2019,2),(2019,3),(2019,4),
    (2020,1),(2020,2),(2020,3),(2020,4),
    (2021,1),(2021,2),(2021,3),(2021,4),
    (2022,1),(2022,2),(2022,3),(2022,4),
    (2023,1),(2023,2),(2023,3),(2023,4),
    ]
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            print(f"No CIK found for {ticker}")
            continue

        print(f"\n{'='*40}")
        print(f"Processing {ticker} (CIK: {cik})")
        print(f"{'='*40}")

        for year, quarter in quarters:
            print(f"\n--- Q{quarter} {year} ---")
            text, url = get_transcript(ticker, cik, year, quarter)
            if text:
                save_transcript(ticker, year, quarter, text, url)
            else:
                print(f"  No transcript found")
            time.sleep(1)

    print("\nAll done!")