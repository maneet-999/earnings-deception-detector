"""
Loughran-McDonald linguistic feature extractor.

The LM word lists are the gold standard for financial NLP — built specifically
for business/financial text (unlike VADER or general sentiment tools which
misclassify financial terms like "liability" or "outstanding" as negative/positive).

Word lists sourced from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
We bundle a curated subset here. Download the full CSV from the link above
and place it at data/LM_MasterDictionary.csv for production use.
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Curated LM word lists (subset — load full CSV for production)      #
# ------------------------------------------------------------------ #

LM_UNCERTAINTY = {
    "approximately", "appear", "appears", "appeared", "believe", "believed",
    "believes", "could", "depends", "dependent", "doubt", "doubtful",
    "estimate", "estimated", "estimates", "feel", "feels", "felt",
    "fluctuate", "fluctuates", "fluctuating", "guess", "hope", "hopes",
    "if", "imprecise", "likelihood", "likely", "may", "maybe",
    "might", "nearly", "occasionally", "often", "ordinarily", "perhaps",
    "possibly", "possible", "predict", "predicted", "probable", "probably",
    "rough", "roughly", "seems", "seldom", "some", "sometimes",
    "somewhat", "suggest", "suggests", "typically", "uncertain", "uncertainly",
    "uncertainty", "unclear", "undecided", "undetermined", "unlikely",
    "unpredictable", "usually", "vague", "variability", "variable",
    "varies", "whether", "yet",
}

LM_HEDGING = {
    "about", "almost", "around", "broadly", "certain extent",
    "essentially", "fairly", "generally", "in part", "largely",
    "mainly", "more or less", "mostly", "nominally", "on balance",
    "overall", "partially", "partly", "predominantly", "primarily",
    "principally", "rather", "reasonably", "relatively", "roughly",
    "selectively", "significantly", "slightly", "somewhat", "substantially",
    "sufficiently", "to some extent", "typically",
}

LM_NEGATION = {
    "cannot", "cant", "could not", "couldn't", "decline", "declined",
    "deficit", "deny", "did not", "didn't", "does not", "doesn't",
    "do not", "don't", "fail", "failed", "has not", "hasn't",
    "have not", "haven't", "impossible", "insufficient", "lack", "lacked",
    "lacking", "lacks", "limitation", "limited", "limiting", "never",
    "no", "nobody", "none", "nor", "not", "nothing", "nowhere",
    "unable", "was not", "wasn't", "were not", "weren't", "will not",
    "won't", "without",
}

LM_POSITIVE = {
    "achieve", "achieved", "achievement", "advantage", "attractive",
    "best", "better", "boost", "capital", "careful", "certain",
    "clearly", "commitment", "committed", "competitive", "confident",
    "confidence", "consistent", "deliver", "delivered", "disciplined",
    "diverse", "efficiency", "excellent", "exceptional", "excited",
    "exciting", "expand", "expanded", "expansion", "favor", "favorable",
    "gain", "gained", "gains", "great", "growth", "healthy", "high",
    "improve", "improved", "improvement", "increases", "increasing",
    "innovative", "leader", "leading", "margin", "momentum", "new",
    "opportunity", "outperform", "outstanding", "positive", "profitable",
    "profitability", "progress", "record", "resilient", "solid",
    "strength", "strong", "strongest", "success", "successful",
    "superior", "sustainable", "value", "win", "wins",
}

LM_NEGATIVE = {
    "abandon", "adverse", "against", "allegation", "although",
    "breach", "burden", "challenge", "challenges", "complaint",
    "concern", "concerns", "constraint", "curtail", "damage",
    "damages", "decline", "decrease", "defaults", "deficit",
    "delay", "difficulties", "difficulty", "dispute", "downturn",
    "drop", "error", "exceed", "exceed",  "fail", "failed", "failure",
    "falling", "fault", "flaw", "forced", "harm", "impair",
    "impaired", "impairment", "inadequate", "infringement", "injunction",
    "insolvency", "investigate", "investigated", "investigation",
    "lawsuit", "liability", "litigation", "loss", "losses",
    "lower", "material weakness", "miss", "missed", "noncompliance",
    "obligation", "penalty", "poor", "problem", "problems",
    "reduce", "reduced", "reduction", "restate", "restated",
    "restatement", "risk", "risks", "shortfall", "slow", "suffer",
    "suffering", "uncertain", "uncertainty", "unfavorable", "violation",
    "weakness", "worse", "writedown", "writeoff",
}

# First-person singular/plural (evasion of accountability)
FIRST_PERSON = {"i", "we", "our", "my", "us", "myself", "ourselves"}


@dataclass
class LinguisticFeatures:
    uncertainty_ratio: float
    hedging_ratio: float
    negation_ratio: float
    positive_ratio: float
    negative_ratio: float
    first_person_ratio: float
    word_count: int
    sentence_count: int
    avg_sentence_length: float


class LMScorer:
    """
    Computes Loughran-McDonald word list ratios for a block of text.

    All ratios are per 100 words to make them scale-invariant.
    We also track first-person usage per sentence (evasion of specificity).
    """

    def __init__(self, lm_csv_path: str = "data/LM_MasterDictionary.csv"):
        self.uncertainty = LM_UNCERTAINTY
        self.hedging = LM_HEDGING
        self.negation = LM_NEGATION
        self.positive = LM_POSITIVE
        self.negative = LM_NEGATIVE

        # If the full LM CSV exists, load it (overrides the built-in lists)
        if Path(lm_csv_path).exists():
            self._load_full_lm(lm_csv_path)

    def _load_full_lm(self, path: str):
        """Load the full LM Master Dictionary from the official CSV."""
        log.info(f"Loading full LM dictionary from {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().upper() for c in df.columns]
        words = df["WORD"].str.lower()

        if "UNCERTAINTY" in df.columns:
            self.uncertainty = set(words[df["UNCERTAINTY"] > 0])
        if "NEGATIVE" in df.columns:
            self.negative = set(words[df["NEGATIVE"] > 0])
        if "POSITIVE" in df.columns:
            self.positive = set(words[df["POSITIVE"] > 0])
        log.info("Full LM dictionary loaded.")

    # -------------------------------------------------------------- #

    def score(self, text: str) -> LinguisticFeatures:
        """Score a block of text and return LinguisticFeatures."""
        if not text or not text.strip():
            return self._zero_features()

        sentences = self._split_sentences(text)
        tokens = self._tokenize(text)

        if not tokens:
            return self._zero_features()

        n = len(tokens)
        token_set = tokens  # keep as list for counting

        uncertainty  = sum(1 for t in token_set if t in self.uncertainty)
        hedging      = self._count_phrases(text.lower(), self.hedging)
        negation     = sum(1 for t in token_set if t in self.negation)
        positive     = sum(1 for t in token_set if t in self.positive)
        negative     = sum(1 for t in token_set if t in self.negative)
        first_person = sum(1 for t in token_set if t in FIRST_PERSON)

        n_sents = max(len(sentences), 1)

        return LinguisticFeatures(
            uncertainty_ratio  = round(uncertainty  / n * 100, 4),
            hedging_ratio      = round(hedging      / n * 100, 4),
            negation_ratio     = round(negation     / n * 100, 4),
            positive_ratio     = round(positive     / n * 100, 4),
            negative_ratio     = round(negative     / n * 100, 4),
            first_person_ratio = round(first_person / n_sents, 4),
            word_count         = n,
            sentence_count     = n_sents,
            avg_sentence_length = round(n / n_sents, 2),
        )

    def score_delta(
        self,
        current: LinguisticFeatures,
        history: list[LinguisticFeatures],
    ) -> dict:
        """
        Compute how much this quarter's scores deviate from
        the company's own rolling average (last N quarters).

        A CEO who suddenly uses 3x more uncertainty language than
        their own baseline is a much stronger signal than the raw ratio.
        """
        if not history:
            return {
                "uncertainty_delta": 0.0,
                "hedging_delta": 0.0,
                "sentiment_delta": 0.0,
            }

        avg_uncertainty = sum(h.uncertainty_ratio for h in history) / len(history)
        avg_hedging     = sum(h.hedging_ratio     for h in history) / len(history)
        avg_sentiment   = sum(
            h.positive_ratio - h.negative_ratio for h in history
        ) / len(history)

        curr_sentiment = current.positive_ratio - current.negative_ratio

        return {
            "uncertainty_delta": round(
                current.uncertainty_ratio - avg_uncertainty, 4
            ),
            "hedging_delta": round(
                current.hedging_ratio - avg_hedging, 4
            ),
            "sentiment_delta": round(
                curr_sentiment - avg_sentiment, 4
            ),
        }

    # -------------------------------------------------------------- #
    #  Helpers                                                         #
    # -------------------------------------------------------------- #

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, return word tokens."""
        return re.findall(r"\b[a-z]+\b", text.lower())

    def _split_sentences(self, text: str) -> list[str]:
        """Naive sentence splitter — good enough for transcript text."""
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def _count_phrases(self, text: str, phrase_set: set) -> int:
        """Count multi-word phrases (hedging list has some 2-word phrases)."""
        count = 0
        for phrase in phrase_set:
            count += text.count(phrase)
        return count

    def _zero_features(self) -> LinguisticFeatures:
        return LinguisticFeatures(
            uncertainty_ratio=0, hedging_ratio=0, negation_ratio=0,
            positive_ratio=0, negative_ratio=0, first_person_ratio=0,
            word_count=0, sentence_count=0, avg_sentence_length=0,
        )
