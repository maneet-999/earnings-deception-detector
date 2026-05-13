"""
FinBERT sentiment scorer + sentiment-gap feature.

FinBERT is a BERT model fine-tuned on financial text by Araci (2019).
It outputs positive / negative / neutral probabilities specifically
calibrated for financial language — unlike VADER, it won't misread
"outstanding liabilities" as positive.

The key feature here isn't raw sentiment — it's the GAP between
how positive management sounds and what their actual numbers show.
A CFO who sounds bullish while margins are collapsing is a red flag.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        log.info("Loading FinBERT (~440MB download on first run)")
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,              # return all 3 labels
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline


# FinBERT can handle ~512 tokens — chunk long texts and average
MAX_CHUNK_CHARS = 1800


@dataclass
class SentimentFeatures:
    prepared_sentiment: float     # -1 (neg) to +1 (pos) for prepared remarks
    qa_sentiment: float           # same for Q&A responses
    sentiment_gap: float          # sentiment_score - financial_health_score
    ceo_sentiment: float
    cfo_sentiment: float
    finbert_positive: float       # raw probabilities
    finbert_negative: float
    finbert_neutral: float


class FinBERTScorer:
    """
    Scores financial text sentiment using FinBERT, then computes
    a divergence signal against reported financials.
    """

    def score_text(self, text: str) -> dict:
        """
        Returns {'positive': float, 'negative': float, 'neutral': float}
        averaged across chunks if text is long.
        """
        if not text or not text.strip():
            return {"positive": 0.333, "negative": 0.333, "neutral": 0.333}

        chunks = self._chunk_text(text)
        pipe = _get_finbert()

        all_scores = []
        for chunk in chunks:
            try:
                result = pipe(chunk)[0]
                scores = {item["label"].lower(): item["score"] for item in result}
                all_scores.append(scores)
            except Exception as e:
                log.warning(f"FinBERT chunk failed: {e}")

        if not all_scores:
            return {"positive": 0.333, "negative": 0.333, "neutral": 0.333}

        avg = {
            "positive": np.mean([s.get("positive", 0) for s in all_scores]),
            "negative": np.mean([s.get("negative", 0) for s in all_scores]),
            "neutral":  np.mean([s.get("neutral", 0)  for s in all_scores]),
        }
        return avg

    def sentiment_score(self, scores: dict) -> float:
        """
        Convert {positive, negative, neutral} to a single [-1, +1] score.
        Neutral is counted as a weak signal in both directions (zero here).
        """
        return round(scores["positive"] - scores["negative"], 4)

    def compute_features(
        self,
        prepared_text: str,
        qa_text: str,
        ceo_text: str,
        cfo_text: str,
        financials: Optional[dict] = None,
    ) -> SentimentFeatures:
        """
        Main entry point. Pass in text blocks and optionally a dict of
        reported financials to compute the sentiment gap.

        financials example:
            {
                "eps_surprise": -0.12,    # missed by 12%
                "revenue_growth": 0.03,   # grew 3% yoy
                "gross_margin": 0.38,     # 38%
            }
        """
        prep_scores = self.score_text(prepared_text)
        qa_scores   = self.score_text(qa_text)
        ceo_scores  = self.score_text(ceo_text)
        cfo_scores  = self.score_text(cfo_text)

        prep_sent = self.sentiment_score(prep_scores)
        qa_sent   = self.sentiment_score(qa_scores)

        # Sentiment gap: how positive are they vs how good are the numbers?
        financial_health = self._financial_health_score(financials)
        gap = round(prep_sent - financial_health, 4)

        return SentimentFeatures(
            prepared_sentiment=prep_sent,
            qa_sentiment=qa_sent,
            sentiment_gap=gap,
            ceo_sentiment=self.sentiment_score(ceo_scores),
            cfo_sentiment=self.sentiment_score(cfo_scores),
            finbert_positive=round(prep_scores["positive"], 4),
            finbert_negative=round(prep_scores["negative"], 4),
            finbert_neutral=round(prep_scores["neutral"], 4),
        )

    def _financial_health_score(self, financials: Optional[dict]) -> float:
        """
        Build a [-1, +1] score from raw reported financial metrics.

        This is what we compare management's language against.
        Simple linear combination — weights tuned from academic literature.
        """
        if not financials:
            return 0.0

        score = 0.0
        weight_total = 0.0

        # EPS surprise: missed vs beat — most predictive single metric
        eps_surprise = financials.get("eps_surprise")
        if eps_surprise is not None:
            score += np.clip(eps_surprise * 2, -1, 1) * 0.45
            weight_total += 0.45

        # Revenue growth: > 10% = strong, < -10% = weak
        rev_growth = financials.get("revenue_growth")
        if rev_growth is not None:
            score += np.clip(rev_growth * 5, -1, 1) * 0.30
            weight_total += 0.30

        # Gross margin: >50% = strong (tech), 30-50% = normal, <30% = weak
        gross_margin = financials.get("gross_margin")
        if gross_margin is not None:
            score += np.clip((gross_margin - 0.35) * 4, -1, 1) * 0.25
            weight_total += 0.25

        if weight_total == 0:
            return 0.0

        return round(score / weight_total, 4)

    def _chunk_text(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
        """Split text into chunks that fit FinBERT's context window."""
        sentences = text.replace("\n", " ").split(". ")
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) < max_chars:
                current += sentence + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + ". "
        if current:
            chunks.append(current.strip())
        return chunks or [text[:max_chars]]
