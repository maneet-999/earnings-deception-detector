"""
Database setup — SQLite for local dev, swap DATABASE_URL in .env for Postgres in prod.
Stores raw transcripts, parsed speaker turns, financials, and feature vectors.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Boolean, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "earnings.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id        = Column(Integer, primary_key=True)
    ticker    = Column(String(10), unique=True, nullable=False)
    name      = Column(String(200))
    cik       = Column(String(20))          # SEC Central Index Key
    sector    = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    transcripts = relationship("Transcript", back_populates="company")
    financials  = relationship("Quarterly", back_populates="company")


class Transcript(Base):
    __tablename__ = "transcripts"

    id          = Column(Integer, primary_key=True)
    company_id  = Column(Integer, ForeignKey("companies.id"), nullable=False)
    ticker      = Column(String(10), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    quarter     = Column(Integer, nullable=False)   # 1-4
    call_date   = Column(DateTime)
    source_url  = Column(String(500))
    raw_text    = Column(Text)                       # full transcript
    is_parsed   = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "fiscal_year", "quarter"),
    )

    company      = relationship("Company", back_populates="transcripts")
    speaker_turns = relationship("SpeakerTurn", back_populates="transcript")
    features     = relationship("FeatureVector", back_populates="transcript", uselist=False)


class SpeakerTurn(Base):
    """One continuous block of speech from a single speaker."""
    __tablename__ = "speaker_turns"

    id            = Column(Integer, primary_key=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=False)
    speaker_name  = Column(String(200))
    speaker_role  = Column(String(50))  # CEO | CFO | ANALYST | OPERATOR | OTHER
    section       = Column(String(50))  # PREPARED | QA
    turn_index    = Column(Integer)     # order within transcript
    text          = Column(Text)

    transcript = relationship("Transcript", back_populates="speaker_turns")


class Quarterly(Base):
    """Reported financials pulled from EDGAR / yfinance for a single quarter."""
    __tablename__ = "quarterly_financials"

    id              = Column(Integer, primary_key=True)
    company_id      = Column(Integer, ForeignKey("companies.id"), nullable=False)
    ticker          = Column(String(10), nullable=False)
    fiscal_year     = Column(Integer, nullable=False)
    quarter         = Column(Integer, nullable=False)
    revenue         = Column(Float)
    gross_margin    = Column(Float)      # as decimal e.g. 0.42
    eps_actual      = Column(Float)
    eps_estimate    = Column(Float)
    eps_surprise    = Column(Float)      # (actual - estimate) / abs(estimate)
    revenue_growth  = Column(Float)      # yoy
    had_restatement = Column(Boolean, default=False)  # ground truth label
    restatement_date = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "fiscal_year", "quarter"),
    )

    company = relationship("Company", back_populates="financials")


class FeatureVector(Base):
    """One row per transcript — the ML feature matrix."""
    __tablename__ = "feature_vectors"

    id            = Column(Integer, primary_key=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), unique=True)
    ticker        = Column(String(10))
    fiscal_year   = Column(Integer)
    quarter       = Column(Integer)

    # --- Linguistic (CEO prepared remarks) ---
    uncertainty_ratio    = Column(Float)   # uncertain words / total words
    hedging_ratio        = Column(Float)   # hedge words / total
    negation_ratio       = Column(Float)
    first_person_ratio   = Column(Float)   # I/we per sentence
    positive_ratio       = Column(Float)   # LM positive words
    negative_ratio       = Column(Float)   # LM negative words

    # --- Temporal drift vs company baseline ---
    uncertainty_delta    = Column(Float)   # vs 8-quarter rolling avg
    hedging_delta        = Column(Float)
    sentiment_delta      = Column(Float)

    # --- Q&A evasion ---
    mean_qa_similarity   = Column(Float)   # avg cosine sim question↔answer
    evasion_rate         = Column(Float)   # % of QA turns below threshold
    cfo_evasion_rate     = Column(Float)
    ceo_evasion_rate     = Column(Float)

    # --- FinBERT sentiment ---
    prepared_sentiment   = Column(Float)   # -1 to 1
    qa_sentiment         = Column(Float)
    sentiment_gap        = Column(Float)   # vs reported financials

    # --- Label ---
    label                = Column(Integer, nullable=True)  # 1=high risk, 0=low risk

    transcript = relationship("Transcript", back_populates="features")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


if __name__ == "__main__":
    init_db()
