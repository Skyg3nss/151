# Chaos Chaser Recognizer Light V3

This version removes the memory-heavy `pokemon-card-recognizer` master reference entirely.

Pipeline:
1. Tesseract reads the top name area.
2. Tesseract reads the bottom collector-number strip.
3. The backend searches TCGdex by name.
4. It fetches candidate cards and validates localId + printed set count.
5. Exact name+number matches win immediately.
6. If the number is not reliable, it returns ranked same-name candidates so the main
   Chaos Chaser app can use its existing visual matcher as a fallback.

No torch. No EasyOCR. No master.pkl.
