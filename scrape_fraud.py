import time
import requests
import re
from database import SessionLocal, Company, Transcript, SpeakerTurn

HEADERS = {"User-Agent": "earnings-detector research@gmail.com"}

# These companies had confirmed accounting fraud/restatements
FRAUD_TICKERS = {
    "NUAN":  "0001097672",   # Nuance Communications — restatement
    "GE":    "0000040987",   # GE — accounting issues 2017-2019
    "WLTW":  "0001140536",   # Willis Towers Watson
    "IEX":   "0001316644",   # IDEX — fraud
    "PRGO":  "0001585583",   # Perrigo — restatement 2019
}

def get_all_ciks():
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    data = resp.json()
    mapping = {}
    for val in data.values():
        mapping[val["ticker"]] = str(val["cik_str"]).zfill(10)
    return mapping

def get_transcript(cik, year, quarter):
    quarter_dates = {
        1: ("01-01", "04-30"),
        2: ("04-01", "07-31"),
        3: ("07-01", "10-31"),
        4: ("10-01", "12-31"),
    }
    start, end = quarter_dates[quarter]
    start_date = f"{year}-{start}"
    end_date   = f"{year}-{end}"

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    data = resp.json()

    filings   = data.get("filings", {}).get("recent", {})
    forms     = filings.get("form", [])
    dates     = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])

    for form, date, accession in zip(forms, dates, accessions):
        if form != "8-K":
            continue
        if not (start_date <= date <= end_date):
            continue

        acc_clean = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"

        try:
            resp = requests.get(index_url, headers=HEADERS, timeout=10)
            time.sleep(0.4)
            links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.(htm|txt))"',
                resp.text, re.IGNORECASE
            )
            for link, ext in links:
                if any(x in link.lower() for x in ["ex99", "ex-99", "exhibit99", "transcript"]):
                    doc_url = "https://www.sec.gov" + link
                    doc_resp = requests.get(doc_url, headers=HEADERS, timeout=15)
                    time.sleep(0.4)
                    text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 3000 and any(
                        w in text.lower() for w in ["question", "analyst", "operator"]
                    ):
                        return text, doc_url
        except:
            continue

    return None, None

def save_transcript(ticker, year, quarter, text, url, label):
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
            print(f"  Already exists: {ticker} Q{quarter} {year}")
            return existing.id

        transcript = Transcript(
            company_id=company.id,
            ticker=ticker,
            fiscal_year=year,
            quarter=quarter,
            source_url=url,
            raw_text=text,
            is_parsed=True,
        )
        db.add(transcript)
        db.commit()
        print(f"  Saved {ticker} Q{quarter} {year}")
        return transcript.id
    except Exception as e:
        db.rollback()
        print(f"  Error: {e}")
        return None
    finally:
        db.close()

if __name__ == "__main__":
    print("Loading CIK map...")
    cik_map = get_all_ciks()

    quarters = [(2018,1),(2018,2),(2018,3),(2018,4),
                (2019,1),(2019,2),(2019,3),(2019,4)]

    for ticker, hardcoded_cik in FRAUD_TICKERS.items():
        cik = cik_map.get(ticker, hardcoded_cik)
        print(f"\n{'='*40}")
        print(f"Scraping {ticker} (CIK: {cik})")
        print(f"{'='*40}")

        for year, quarter in quarters:
            print(f"  Q{quarter} {year}...")
            text, url = get_transcript(cik, year, quarter)
            if text:
                save_transcript(ticker, year, quarter, text, url, label=1)
            else:
                print(f"  No transcript found")
            time.sleep(0.5)

    print("\nDone!")