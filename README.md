# Chaos Chaser Recognizer Light V4 — Hybrid Exact

V4 is designed for Render Free and exact-printing recognition without PyTorch/EasyOCR.

Recognition signals:
- repeated card-name OCR across multiple header preprocess variants
- strict bottom collector-number OCR
- denominator validation against TCGdex set.cardCount.official/total
- HP tie-breaker
- ability/attack OCR tie-breaker against full TCGdex card metadata
- honest confidence + top candidates

This keeps the backend lightweight while being much harder to fool by one bad OCR token.
