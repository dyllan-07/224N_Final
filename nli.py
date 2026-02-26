"""
NLI-based support scoring using FacebookAI/roberta-large-mnli.

Given a query q (post + community note) and a candidate passage p (excerpt),
computes how well p supports the claim in q via entailment probability.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "FacebookAI/roberta-large-mnli"
ENTAILMENT_LABEL = "ENTAILMENT"
DEFAULT_MAX_LENGTH = 256


def _get_entailment_id(model) -> int:
    """Return the label id for ENTAILMENT from the model config."""
    if hasattr(model.config, "label2id"):
        return model.config.label2id[ENTAILMENT_LABEL]
    id2label = getattr(model.config, "id2label", {})
    for idx, name in id2label.items():
        if name == ENTAILMENT_LABEL:
            return int(idx)
    return 2  # fallback: FacebookAI/roberta-large-mnli uses 2 for ENTAILMENT


class NLISupportScorer:
    """
    Scores how well a candidate passage supports a claim (query) using
    roberta-large-mnli. Support score = P(entailment | premise=passage, hypothesis=query).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._entailment_id = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            ).to(self.device)
            self._model.eval()
            self._entailment_id = _get_entailment_id(self._model)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def score(self, query: str, passage: str) -> float:
        """
        Compute support score: how well `passage` supports the claim in `query`.

        In NLI terms: premise = passage (evidence), hypothesis = query (claim).
        Returns P(entailment), in [0, 1].

        Args:
            query: Concatenation of original post + community note (the claim).
            passage: Candidate excerpt/source passage.

        Returns:
            Support score in [0, 1].
        """
        inputs = self.tokenizer(
            passage,
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self._entailment_id].item()

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Compute support scores for multiple (query, passage) pairs.

        Args:
            pairs: List of (query, passage) tuples.

        Returns:
            List of support scores in [0, 1].
        """
        if not pairs:
            return []
        premises = [p for _, p in pairs]
        hypotheses = [q for q, _ in pairs]
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return [probs[i, self._entailment_id].item() for i in range(len(pairs))]


# Convenience: lazy-loaded singleton for simple use from other modules
_scorer: NLISupportScorer | None = None


def get_scorer(
    model_id: str = MODEL_ID,
    device: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> NLISupportScorer:
    """Return a shared NLISupportScorer instance (lazy-loaded)."""
    global _scorer
    if _scorer is None:
        _scorer = NLISupportScorer(model_id=model_id, device=device, max_length=max_length)
    return _scorer


def support_score(query: str, passage: str, **scorer_kwargs) -> float:
    """
    One-off support score for (query, passage). Uses shared scorer if available.

    Args:
        query: Post + community note (claim).
        passage: Candidate excerpt.
        **scorer_kwargs: Passed to NLISupportScorer if creating one.

    Returns:
        Support score in [0, 1].
    """
    return get_scorer(**scorer_kwargs).score(query, passage)
