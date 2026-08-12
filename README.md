# Chaos Chaser Recognizer Light V2

This build exists because the original `pokemon-card-recognizer` backend exceeded
Render Free's 512 MiB RAM limit.

## What changed

The upstream `CardRecognizer` hardcodes `OCRMethod.EASYOCR`. EasyOCR uses PyTorch and
was the memory-heavy part.

This V2 intentionally **does not import CardRecognizer**.

Instead it keeps the useful Pokemon-specific part:

- the package's bundled `master.pkl` reference
- the package's `WordClassifier`
- the package's Pokemon card lookup data

and replaces only EasyOCR with the system Tesseract binary.

So recognition is:

image -> Tesseract words -> upstream Pokemon WordClassifier -> exact card lookup.

## Update your existing GitHub repo

You can replace the files in the repository you already connected to Render.

Upload these files to the ROOT of the repo and replace the old versions:

- app.py
- Dockerfile
- render.yaml
- README.md

You can delete the old `requirements.txt`; this Dockerfile installs dependencies itself.

Then commit the changes. Render should auto-deploy.

## What to test

1. Wait for Render deploy.
2. Open:
   `/health`

Expected:

```json
{"ok":true,"status":"healthy","engine":"tesseract-reference-light"}
```

3. Open `/warmup` with POST (not a normal browser GET). The simplest next step is
   to let ChatGPT make the frontend tester once `/health` works.

## Why this should use much less memory

No EasyOCR.
No torch.
No torchvision.
No CUDA libraries.

The heaviest persistent object should now be the Pokemon master reference matrix +
normal Python/numpy/pandas runtime rather than a neural OCR model.

## Important

This is an experiment to see if the upstream project's Pokemon reference/classifier
is good enough when fed Tesseract text. Accuracy may differ from EasyOCR, but this is
the cleanest way to keep the useful Pokemon-specific recognition while targeting the
512 MiB free instance.


## V2.1 patch
Pinned `ocr-ops` to `0.0.0.4.3.1` for current Python 3.11 Docker compatibility.

## V2.2 build fix

The real compatibility problem was the Python patch version, not just `ocr-ops`.

Both `ocr-ops 0.0.0.4.3.2` and `algo-ops 0.0.1.7.1` require Python <= 3.11.7.
The generic `python:3.11-slim` Docker tag now resolves to a newer 3.11.x release,
so V2.2 pins the container itself to `python:3.11.7-slim`.

This lets the intended package versions install without pulling EasyOCR/PyTorch.
