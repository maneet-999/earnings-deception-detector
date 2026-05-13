"""
Feature pipeline — ties together LM scorer, QA evasion, and FinBERT.

Takes a transcript + financials, outputs a clean FeatureVector row
ready to be written to the DB or handed to the model.

Usage:
    pipeline = FeaturePipeline()
    vector = pipeline.run(transcript_id=42)
    pipeline.save(vector)
"""

import logging
from dataclasses import asdict

from database import SessionLocal, Transcript, SpeakerTurn, Quarterly, FeatureVector
from lm_scorer import LMScorer
from qa_evasion import QAEvasionScorer
from finbert_scorer import FinBERTScorer

log = logging.getLogger(__name__)


class FeaturePipeline:

    def __init__(self):
        self.lm      = LMScorer()
        self.qa      = QAEvasionScorer()
        self.finbert = FinBERTScorer()

    def run(self, transcript_id: int) -> dict | None:
        """
        Build a complete feature vector for one transcript.
        Returns a dict ready to insert into FeatureVector table.
        """
        db = SessionLocal()
        try:
            transcript = db.query(Transcript).get(transcript_id)
            if not transcript:
                log.error(f"Transcript {transcript_id} not found")
                return None

            turns = (
                db.query(SpeakerTurn)
                .filter_by(transcript_id=transcript_id)
                .order_by(SpeakerTurn.turn_index)
                .all()
            )

            # Reported financials for sentiment gap
            financials = (
                db.query(Quarterly)
                .filter_by(
                    ticker=transcript.ticker,
                    fiscal_year=transcript.fiscal_year,
                    quarter=transcript.quarter,
                )
                .first()
            )

            # Historical feature vectors for delta calculation (last 8 quarters)
            history_ids = self._get_historical_transcript_ids(
                db, transcript.ticker, transcript.fiscal_year, transcript.quarter
            )
            historical_lm = self._load_historical_lm(db, history_ids)

        finally:
            db.close()

        # ---------------------------------------------------------------- #
        # 1. Split text by speaker role and section                        #
        # ---------------------------------------------------------------- #
        def text_for(role=None, section=None):
            filtered = [
                t.text for t in turns
                if (role is None or t.speaker_role == role)
                and (section is None or t.section == section)
                and t.text
            ]
            return " ".join(filtered)

        ceo_prepared = text_for("CEO", "PREPARED")
        cfo_prepared = text_for("CFO", "PREPARED")
        all_prepared = text_for(section="PREPARED")
        all_qa       = text_for(section="QA")
        ceo_qa       = text_for("CEO", "QA")
        cfo_qa       = text_for("CFO", "QA")

        # ---------------------------------------------------------------- #
        # 2. Linguistic features on CEO prepared remarks                   #
        # ---------------------------------------------------------------- #
        lm_features = self.lm.score(ceo_prepared or all_prepared)
        delta = self.lm.score_delta(lm_features, historical_lm)

        # ---------------------------------------------------------------- #
        # 3. Q&A evasion                                                   #
        # ---------------------------------------------------------------- #
        turns_dicts = [
            {
                "speaker_role": t.speaker_role,
                "section": t.section,
                "text": t.text,
                "turn_index": t.turn_index,
            }
            for t in turns
        ]
        qa_features = self.qa.score(turns_dicts)

        # ---------------------------------------------------------------- #
        # 4. FinBERT sentiment + gap                                       #
        # ---------------------------------------------------------------- #
        fin_dict = None
        if financials:
            fin_dict = {
                "eps_surprise":   financials.eps_surprise,
                "revenue_growth": financials.revenue_growth,
                "gross_margin":   financials.gross_margin,
            }

        sentiment_features = self.finbert.compute_features(
            prepared_text=all_prepared,
            qa_text=all_qa,
            ceo_text=ceo_qa,
            cfo_text=cfo_qa,
            financials=fin_dict,
        )

        # ---------------------------------------------------------------- #
        # 5. Ground truth label                                            #
        # ---------------------------------------------------------------- #
        label = None
        if financials and financials.had_restatement:
            label = 1
        elif financials:
            label = 0  # confirmed clean quarter

        # ---------------------------------------------------------------- #
        # 6. Assemble feature vector dict                                  #
        # ---------------------------------------------------------------- #
        vector = {
            "transcript_id":       transcript_id,
            "ticker":              transcript.ticker,
            "fiscal_year":         transcript.fiscal_year,
            "quarter":             transcript.quarter,

            # LM linguistic
            "uncertainty_ratio":   lm_features.uncertainty_ratio,
            "hedging_ratio":       lm_features.hedging_ratio,
            "negation_ratio":      lm_features.negation_ratio,
            "first_person_ratio":  lm_features.first_person_ratio,
            "positive_ratio":      lm_features.positive_ratio,
            "negative_ratio":      lm_features.negative_ratio,

            # Temporal delta
            "uncertainty_delta":   delta["uncertainty_delta"],
            "hedging_delta":       delta["hedging_delta"],
            "sentiment_delta":     delta["sentiment_delta"],

            # Q&A evasion
            "mean_qa_similarity":  qa_features.mean_similarity,
            "evasion_rate":        qa_features.evasion_rate,
            "cfo_evasion_rate":    qa_features.cfo_evasion_rate,
            "ceo_evasion_rate":    qa_features.ceo_evasion_rate,

            # FinBERT sentiment
            "prepared_sentiment":  sentiment_features.prepared_sentiment,
            "qa_sentiment":        sentiment_features.qa_sentiment,
            "sentiment_gap":       sentiment_features.sentiment_gap,

            # Label
            "label": label,
        }

        log.info(
            f"Features done: {transcript.ticker} "
            f"Q{transcript.quarter} {transcript.fiscal_year} "
            f"| label={label}"
        )
        return vector

    def save(self, vector: dict):
        """Write a feature vector dict to the DB."""
        db = SessionLocal()
        try:
            existing = db.query(FeatureVector).filter_by(
                transcript_id=vector["transcript_id"]
            ).first()
            if existing:
                for k, v in vector.items():
                    setattr(existing, k, v)
            else:
                db.add(FeatureVector(**vector))
            db.commit()
        except Exception as e:
            db.rollback()
            log.error(f"Error saving feature vector: {e}")
            raise
        finally:
            db.close()

    # ---------------------------------------------------------------- #
    # Helpers                                                           #
    # ---------------------------------------------------------------- #

    def _get_historical_transcript_ids(
        self, db, ticker: str, current_year: int, current_quarter: int, n: int = 8
    ) -> list[int]:
        """Get transcript IDs for the last N quarters (excluding current)."""
        all_transcripts = (
            db.query(Transcript)
            .filter(
                Transcript.ticker == ticker,
                (Transcript.fiscal_year < current_year)
                | (
                    (Transcript.fiscal_year == current_year)
                    & (Transcript.quarter < current_quarter)
                ),
            )
            .order_by(Transcript.fiscal_year.desc(), Transcript.quarter.desc())
            .limit(n)
            .all()
        )
        return [t.id for t in all_transcripts]

    def _load_historical_lm(self, db, transcript_ids: list[int]):
        """Load LM feature objects for historical quarters."""
        from lm_scorer import LinguisticFeatures
        lm = LMScorer()
        historical = []

        for tid in transcript_ids:
            fv = db.query(FeatureVector).filter_by(transcript_id=tid).first()
            if fv:
                historical.append(
                    LinguisticFeatures(
                        uncertainty_ratio=fv.uncertainty_ratio or 0,
                        hedging_ratio=fv.hedging_ratio or 0,
                        negation_ratio=fv.negation_ratio or 0,
                        positive_ratio=fv.positive_ratio or 0,
                        negative_ratio=fv.negative_ratio or 0,
                        first_person_ratio=fv.first_person_ratio or 0,
                        word_count=0,
                        sentence_count=0,
                        avg_sentence_length=0,
                    )
                )
        return historical
