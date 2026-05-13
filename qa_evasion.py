"""
Q&A evasion scorer.

The core idea: an analyst asks about gross margin. If the executive's
answer is semantically similar to the question, they actually answered it.
If it's not, they pivoted.

We embed both question and answer using sentence-BERT (MiniLM-L6-v2 —
fast, small, runs on CPU), compute cosine similarity, and flag turns
below a threshold as evasions.

This is the most novel feature in the project. No published system
does this on earnings calls — make sure you mention that.
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Lazy import — only load the model when first needed
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformers model (first time — ~80MB download)")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


# Cosine similarity below this threshold = evasion
EVASION_THRESHOLD = 0.30

# Topics analysts commonly ask about — used to weight "important" questions
HIGH_STAKES_TOPICS = [
    "guidance", "margin", "revenue", "growth", "outlook", "forecast",
    "profit", "earnings", "cash", "debt", "headcount", "layoffs",
    "investigation", "restatement", "accounting", "compliance",
]


@dataclass
class QAEvasionFeatures:
    mean_similarity: float       # avg cosine sim across all QA pairs
    evasion_rate: float          # % of turns below EVASION_THRESHOLD
    high_stakes_evasion_rate: float  # evasion rate on sensitive topics only
    ceo_evasion_rate: float
    cfo_evasion_rate: float
    qa_pair_count: int


class QAEvasionScorer:
    """
    Scores how well executives actually answer analyst questions.

    Workflow:
    1. Extract QA pairs: (analyst_question, executive_response)
    2. Embed both with sentence-BERT
    3. Compute cosine similarity per pair
    4. Aggregate into evasion metrics
    """

    def __init__(self, threshold: float = EVASION_THRESHOLD):
        self.threshold = threshold

    def score(self, speaker_turns: list[dict]) -> QAEvasionFeatures:
        """
        speaker_turns: list of dicts with keys:
            speaker_role, section, text, turn_index
        """
        qa_turns = [t for t in speaker_turns if t.get("section") == "QA"]

        if len(qa_turns) < 2:
            return self._zero_features()

        pairs = self._extract_pairs(qa_turns)
        if not pairs:
            return self._zero_features()

        model = _get_model()

        questions  = [p["question"] for p in pairs]
        answers    = [p["answer"]   for p in pairs]
        responders = [p["responder"] for p in pairs]
        topics     = [p["is_high_stakes"] for p in pairs]

        # Batch encode — much faster than encoding one at a time
        q_embeddings = model.encode(questions, batch_size=32, show_progress_bar=False)
        a_embeddings = model.encode(answers,   batch_size=32, show_progress_bar=False)

        similarities = self._cosine_similarities(q_embeddings, a_embeddings)

        evasions = [s < self.threshold for s in similarities]
        hs_evasions = [
            e for e, hs in zip(evasions, topics) if hs
        ]

        def role_evasion(role: str) -> float:
            role_evade = [e for e, r in zip(evasions, responders) if r == role]
            return sum(role_evade) / len(role_evade) if role_evade else 0.0

        return QAEvasionFeatures(
            mean_similarity=round(float(np.mean(similarities)), 4),
            evasion_rate=round(sum(evasions) / len(evasions), 4),
            high_stakes_evasion_rate=round(
                sum(hs_evasions) / len(hs_evasions) if hs_evasions else 0.0, 4
            ),
            ceo_evasion_rate=round(role_evasion("CEO"), 4),
            cfo_evasion_rate=round(role_evasion("CFO"), 4),
            qa_pair_count=len(pairs),
        )

    # ---------------------------------------------------------------- #
    #  Pair extraction                                                   #
    # ---------------------------------------------------------------- #

    def _extract_pairs(self, qa_turns: list[dict]) -> list[dict]:
        """
        Match each ANALYST question turn to the next EXECUTIVE response turn.
        Handles the common pattern: Analyst → CEO → (CFO adds on) →
        next Analyst question.
        """
        pairs = []
        i = 0
        while i < len(qa_turns) - 1:
            turn = qa_turns[i]
            if turn.get("speaker_role") != "ANALYST":
                i += 1
                continue

            question = turn.get("text", "").strip()
            if not question or len(question.split()) < 5:
                i += 1
                continue

            # Collect the executive response(s) that follow
            answer_parts = []
            responder = None
            j = i + 1
            while j < len(qa_turns):
                next_turn = qa_turns[j]
                if next_turn.get("speaker_role") == "ANALYST":
                    break
                if next_turn.get("speaker_role") in ("CEO", "CFO", "OTHER"):
                    if not responder:
                        responder = next_turn["speaker_role"]
                    answer_parts.append(next_turn.get("text", ""))
                j += 1

            answer = " ".join(answer_parts).strip()
            if answer and len(answer.split()) > 10:
                pairs.append({
                    "question": question,
                    "answer": answer,
                    "responder": responder or "OTHER",
                    "is_high_stakes": self._is_high_stakes(question),
                })

            i = j if j > i else i + 1

        return pairs

    def _is_high_stakes(self, question: str) -> bool:
        """Flag questions touching on sensitive financial/legal topics."""
        q_lower = question.lower()
        return any(topic in q_lower for topic in HIGH_STAKES_TOPICS)

    # ---------------------------------------------------------------- #
    #  Math                                                             #
    # ---------------------------------------------------------------- #

    def _cosine_similarities(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> list[float]:
        """Vectorised cosine similarity — no sklearn dependency needed."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (a_norm * b_norm).sum(axis=1).tolist()

    def _zero_features(self) -> QAEvasionFeatures:
        return QAEvasionFeatures(
            mean_similarity=0, evasion_rate=0,
            high_stakes_evasion_rate=0,
            ceo_evasion_rate=0, cfo_evasion_rate=0,
            qa_pair_count=0,
        )
