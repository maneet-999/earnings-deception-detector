# Contributing to Earnings Call Deception Detector

Thank you for your interest in contributing. This document covers how to get started.

---

## What we need most

- **More training data** — restatement/fraud company transcripts from EDGAR
- **New NLP features** — any linguistically-motivated signal not already in the pipeline
- **Indian market support** — NSE/BSE earnings call sources
- **Bug fixes** — especially around scraper robustness and edge cases

---

## Getting started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOURUSERNAME/earnings-deception-detector.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Set up the dev environment (see README installation steps)
5. Make your changes
6. Test that the pipeline runs end to end: `python main.py --features`
7. Submit a pull request

---

## Code style

- Follow existing naming conventions
- Add docstrings to any new functions
- Keep feature extractors stateless (input text → output features, no side effects)
- Log using the standard `logging` module, not `print()`

---

## Adding a new feature extractor

1. Create your scorer in `features/your_scorer.py`
2. Add the feature column to `FEATURE_COLS` in `trainer.py` and `dashboard_v6.py`
3. Integrate it into `features/pipeline.py`
4. Add it to `FLABELS`, `FDESC`, and `RDIR` in the dashboard

---

## Reporting bugs

Open a GitHub issue with:
- Python version
- Steps to reproduce
- Error message / traceback
- Expected vs actual behaviour

---

## Pull request checklist

- [ ] Code runs without errors
- [ ] `python main.py --features` completes successfully
- [ ] New features are documented in the PR description
- [ ] No API keys or database files committed
