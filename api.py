"""
Earnings Call Deception Detector — FastAPI Service

Endpoints:
  GET  /                              — API info
  GET  /health                        — liveness check
  GET  /companies                     — list all scored companies
  GET  /score/{ticker}                — all quarterly scores for a ticker
  GET  /score/{ticker}/{year}/{quarter} — single quarter score + SHAP drivers
  POST /score/batch                   — score multiple tickers at once
  GET  /leaderboard                   — most suspicious companies right now
  GET  /compare?tickers=MSFT,GOOGL    — compare companies side by side

Run:
  uvicorn api:app --reload --port 8000

Docs auto-generated at:
  http://localhost:8000/docs    (Swagger UI)
  http://localhost:8000/redoc   (ReDoc)
"""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Flat imports — works with your file structure
from database import SessionLocal, FeatureVector, Transcript, Company

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_COLS = [
    "uncertainty_ratio", "hedging_ratio", "negation_ratio",
    "first_person_ratio", "positive_ratio", "negative_ratio",
    "uncertainty_delta", "hedging_delta", "sentiment_delta",
    "mean_qa_similarity", "evasion_rate", "cfo_evasion_rate", "ceo_evasion_rate",
    "prepared_sentiment", "qa_sentiment", "sentiment_gap",
]

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Earnings Call Deception Detector",
    description="""
## What this API does

Scores quarterly earnings call transcripts for linguistic deception signals
using three NLP feature families:

- **Loughran-McDonald** linguistic uncertainty and hedging ratios
- **sentence-BERT Q&A evasion** — semantic similarity between analyst questions and executive answers
- **FinBERT sentiment gap** — how optimistic management sounds vs what the numbers show

All predictions include **SHAP feature attribution** so you know exactly why a score is high or low.

## Risk levels
- **HIGH** (>65) — statistically unusual language patterns, elevated restatement risk
- **MEDIUM** (40–65) — some signals present, monitor
- **LOW** (<40) — language consistent with historical baseline
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model():
    """Load XGBoost model once and cache it."""
    try:
        model = xgb.XGBClassifier()
        model.load_model("models/deception_model.json")
        log.info("Model loaded successfully")
        return model
    except Exception as e:
        log.error(f"Could not load model: {e}")
        return None


def predict(feature_dict: dict, model) -> dict:
    """Score a feature dict — returns risk_score, level, probability, SHAP drivers."""
    try:
        import shap
        X = np.array([[feature_dict.get(c, 0) or 0 for c in FEATURE_COLS]])
        prob  = float(model.predict_proba(X)[0][1])
        score = round(prob * 100, 1)
        level = "HIGH" if score > 65 else "MEDIUM" if score > 40 else "LOW"

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
        drivers = sorted(
            [{"feature": f, "shap_value": round(float(s), 4), "direction": "increases_risk" if s > 0 else "reduces_risk"}
             for f, s in zip(FEATURE_COLS, shap_vals)],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )
        return {
            "risk_score":  score,
            "risk_level":  level,
            "probability": round(prob, 4),
            "top_drivers": drivers[:5],
        }
    except Exception as e:
        log.warning(f"Prediction failed: {e}")
        return {"risk_score": 0, "risk_level": "UNKNOWN", "probability": 0, "top_drivers": []}


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    n_companies: int
    n_quarters: int


class ScoreResponse(BaseModel):
    ticker: str
    fiscal_year: int
    quarter: int
    period: str
    risk_score: float
    risk_level: str
    probability: float
    top_drivers: list[dict]
    feature_values: dict


class CompanyScore(BaseModel):
    ticker: str
    fiscal_year: int
    quarter: int
    period: str
    risk_score: float
    risk_level: str


class BatchRequest(BaseModel):
    tickers: list[str]
    fiscal_year: int
    quarter: int

    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["MSFT", "GOOGL", "TSLA"],
                "fiscal_year": 2023,
                "quarter": 2
            }
        }


class CompareResponse(BaseModel):
    tickers: list[str]
    fiscal_year: int
    quarter: int
    comparison: list[dict]
    highest_risk: str
    lowest_risk: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html><body style="font-family:monospace;background:#0a0b0d;color:#f0f1f5;padding:48px;max-width:600px">
      <h1 style="color:#c8a96e;font-size:24px;margin-bottom:8px">Earnings Deception Detector API</h1>
      <p style="color:#9ba3b8;margin-bottom:24px">v1.0.0 — Real-time earnings call deception scoring</p>
      <div style="margin-bottom:8px"><a href="/docs" style="color:#c8a96e">→ Swagger UI (interactive docs)</a></div>
      <div style="margin-bottom:8px"><a href="/redoc" style="color:#c8a96e">→ ReDoc (clean reference)</a></div>
      <div style="margin-bottom:8px"><a href="/health" style="color:#c8a96e">→ Health check</a></div>
      <div style="margin-bottom:8px"><a href="/leaderboard" style="color:#c8a96e">→ Risk leaderboard</a></div>
      <div style="margin-top:32px;color:#5c6480;font-size:12px">
        Built with Loughran-McDonald · FinBERT · sentence-BERT · XGBoost
      </div>
    </body></html>
    """


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — confirms model and database are ready."""
    db = SessionLocal()
    try:
        n_companies = db.query(FeatureVector.ticker).distinct().count()
        n_quarters  = db.query(FeatureVector).count()
    finally:
        db.close()

    model = load_model()
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
        n_companies=n_companies,
        n_quarters=n_quarters,
    )


@app.get("/companies", tags=["Data"])
def list_companies():
    """List all companies with scored transcripts."""
    db = SessionLocal()
    try:
        rows = db.query(FeatureVector.ticker).distinct().all()
        tickers = sorted([r[0] for r in rows])
        return {
            "count": len(tickers),
            "companies": tickers,
        }
    finally:
        db.close()


@app.get("/score/{ticker}", response_model=list[ScoreResponse], tags=["Scoring"])
def score_ticker(ticker: str):
    """
    Return all quarterly deception scores for a company.

    Scores are sorted chronologically. Each score includes:
    - `risk_score` — 0 to 100 (higher = more suspicious)
    - `risk_level` — HIGH / MEDIUM / LOW
    - `top_drivers` — SHAP feature attribution explaining the score
    - `feature_values` — raw NLP feature values for full transparency
    """
    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python main.py --train")

    db = SessionLocal()
    try:
        rows = (
            db.query(FeatureVector)
            .filter(FeatureVector.ticker == ticker.upper())
            .order_by(FeatureVector.fiscal_year, FeatureVector.quarter)
            .all()
        )
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No data for {ticker.upper()}. Run the scraper first."
            )

        results = []
        for row in rows:
            feature_dict = {col: getattr(row, col) for col in FEATURE_COLS}
            pred = predict(feature_dict, model)
            results.append(ScoreResponse(
                ticker=row.ticker,
                fiscal_year=row.fiscal_year,
                quarter=row.quarter,
                period=f"Q{row.quarter} {row.fiscal_year}",
                risk_score=pred["risk_score"],
                risk_level=pred["risk_level"],
                probability=pred["probability"],
                top_drivers=pred["top_drivers"],
                feature_values={k: round(v, 4) if v else 0 for k, v in feature_dict.items()},
            ))
        return results
    finally:
        db.close()


@app.get("/score/{ticker}/{year}/{quarter}", response_model=ScoreResponse, tags=["Scoring"])
def score_single_quarter(ticker: str, year: int, quarter: int):
    """
    Score a specific quarter for a company.

    Example: `/score/MSFT/2023/2` → Microsoft Q2 2023
    """
    if quarter not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Quarter must be 1, 2, 3, or 4")

    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    db = SessionLocal()
    try:
        row = (
            db.query(FeatureVector)
            .filter_by(ticker=ticker.upper(), fiscal_year=year, quarter=quarter)
            .first()
        )
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"No data for {ticker.upper()} Q{quarter} {year}"
            )

        feature_dict = {col: getattr(row, col) for col in FEATURE_COLS}
        pred = predict(feature_dict, model)

        return ScoreResponse(
            ticker=row.ticker,
            fiscal_year=row.fiscal_year,
            quarter=row.quarter,
            period=f"Q{row.quarter} {row.fiscal_year}",
            risk_score=pred["risk_score"],
            risk_level=pred["risk_level"],
            probability=pred["probability"],
            top_drivers=pred["top_drivers"],
            feature_values={k: round(v, 4) if v else 0 for k, v in feature_dict.items()},
        )
    finally:
        db.close()


@app.post("/score/batch", response_model=list[CompanyScore], tags=["Scoring"])
def score_batch(request: BatchRequest):
    """
    Score multiple companies for the same quarter.
    Returns results sorted by risk score descending (most suspicious first).

    Useful for screening a watchlist of companies after earnings season.
    """
    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    db = SessionLocal()
    results = []
    try:
        for ticker in request.tickers:
            row = (
                db.query(FeatureVector)
                .filter_by(
                    ticker=ticker.upper(),
                    fiscal_year=request.fiscal_year,
                    quarter=request.quarter,
                )
                .first()
            )
            if not row:
                continue

            feature_dict = {col: getattr(row, col) for col in FEATURE_COLS}
            pred = predict(feature_dict, model)
            results.append(CompanyScore(
                ticker=ticker.upper(),
                fiscal_year=request.fiscal_year,
                quarter=request.quarter,
                period=f"Q{request.quarter} {request.fiscal_year}",
                risk_score=pred["risk_score"],
                risk_level=pred["risk_level"],
            ))

        return sorted(results, key=lambda x: x.risk_score, reverse=True)
    finally:
        db.close()


@app.get("/leaderboard", tags=["Analytics"])
def leaderboard(limit: int = Query(default=20, ge=1, le=100)):
    """
    Most suspicious companies based on their most recent quarter.
    Ranked by deception risk score descending.

    This is the endpoint to embed in a monitoring dashboard
    or run after each earnings season.
    """
    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    db = SessionLocal()
    try:
        all_fvs = db.query(FeatureVector).all()

        # Get most recent quarter per ticker
        latest_per_ticker = {}
        for fv in all_fvs:
            existing = latest_per_ticker.get(fv.ticker)
            if not existing or (fv.fiscal_year, fv.quarter) > (existing.fiscal_year, existing.quarter):
                latest_per_ticker[fv.ticker] = fv

        scored = []
        for ticker, fv in latest_per_ticker.items():
            feature_dict = {col: getattr(fv, col) for col in FEATURE_COLS}
            pred = predict(feature_dict, model)
            scored.append({
                "ticker":      ticker,
                "fiscal_year": fv.fiscal_year,
                "quarter":     fv.quarter,
                "period":      f"Q{fv.quarter} {fv.fiscal_year}",
                "risk_score":  pred["risk_score"],
                "risk_level":  pred["risk_level"],
            })

        scored.sort(key=lambda x: x["risk_score"], reverse=True)
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(scored[:limit]),
            "leaderboard": scored[:limit],
        }
    finally:
        db.close()


@app.get("/compare", response_model=CompareResponse, tags=["Analytics"])
def compare(
    tickers: str = Query(..., description="Comma-separated tickers e.g. MSFT,GOOGL,TSLA"),
    year: int = Query(..., description="Fiscal year e.g. 2023"),
    quarter: int = Query(..., description="Quarter 1-4"),
):
    """
    Compare multiple companies side-by-side for the same quarter.

    Example: `/compare?tickers=MSFT,GOOGL,TSLA&year=2023&quarter=2`

    Returns a ranked comparison with feature-level breakdown —
    useful for relative screening across a peer group.
    """
    if quarter not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Quarter must be 1-4")

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 tickers")
    if len(ticker_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers per comparison")

    model = load_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    db = SessionLocal()
    comparison = []
    try:
        for ticker in ticker_list:
            row = (
                db.query(FeatureVector)
                .filter_by(ticker=ticker, fiscal_year=year, quarter=quarter)
                .first()
            )
            if not row:
                continue

            feature_dict = {col: getattr(row, col) for col in FEATURE_COLS}
            pred = predict(feature_dict, model)
            comparison.append({
                "ticker":             ticker,
                "risk_score":         pred["risk_score"],
                "risk_level":         pred["risk_level"],
                "top_driver":         pred["top_drivers"][0]["feature"] if pred["top_drivers"] else "—",
                "uncertainty_ratio":  round(feature_dict.get("uncertainty_ratio") or 0, 4),
                "evasion_rate":       round(feature_dict.get("evasion_rate") or 0, 4),
                "sentiment_gap":      round(feature_dict.get("sentiment_gap") or 0, 4),
                "hedging_ratio":      round(feature_dict.get("hedging_ratio") or 0, 4),
            })

        if not comparison:
            raise HTTPException(status_code=404, detail="No data found for any of the provided tickers")

        comparison.sort(key=lambda x: x["risk_score"], reverse=True)

        return CompareResponse(
            tickers=ticker_list,
            fiscal_year=year,
            quarter=quarter,
            comparison=comparison,
            highest_risk=comparison[0]["ticker"],
            lowest_risk=comparison[-1]["ticker"],
        )
    finally:
        db.close()
