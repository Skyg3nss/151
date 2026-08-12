import os
import re
import tempfile
import threading
import traceback
import importlib.util
from pathlib import Path

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

# IMPORTANT:
# We DO NOT import CardRecognizer here.
# CardRecognizer hardcodes EasyOCR, which pulls PyTorch into RAM.
#
# Instead we reuse the package's own bundled MASTER reference + WordClassifier,
# but feed it text from lightweight system Tesseract.
from pokemon_card_recognizer.classifier.core.word_classifier import WordClassifier

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_classifier = None
_classifier_lock = threading.Lock()
_run_lock = threading.Lock()


def master_reference_path():
    spec = importlib.util.find_spec("pokemon_card_recognizer")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("pokemon_card_recognizer package not found")

    root = Path(next(iter(spec.submodule_search_locations)))
    p = root / "reference" / "data" / "ref_build" / "master.pkl"

    if not p.exists():
        raise RuntimeError(f"Bundled master reference not found at {p}")

    return str(p)


def get_classifier():
    global _classifier

    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is None:
            _classifier = WordClassifier(
                ref_pkl_path=master_reference_path(),
                vect_method="encapsulation_match",
                classification_method="shared_words",
            )

    return _classifier


def normalize_word(word):
    word = str(word or "").lower().strip()
    word = re.sub(r"[^\w]+", "", word)
    return word


def preprocess_variants(path):
    """
    Tesseract is much lighter than EasyOCR but likes clean, sharp text.
    Build several card-oriented views. No OpenCV/torch required.
    """
    img = Image.open(path).convert("RGB")

    # Keep RAM predictable on huge phone photos.
    max_w = 1400
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize(
            (max_w, max(1, int(img.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    variants = []

    # Full card
    variants.append(("full", img))

    # Top 24%: Pokemon/card name + HP
    variants.append(("top", img.crop((0, 0, img.width, int(img.height * 0.26)))))

    # Bottom 28%: collector info, artist, rules
    variants.append(
        (
            "bottom",
            img.crop(
                (
                    0,
                    int(img.height * 0.72),
                    img.width,
                    img.height,
                )
            ),
        )
    )

    # Main lower text region: attacks/abilities are extremely useful to the
    # recognizer because the reference contains those words too.
    variants.append(
        (
            "middle",
            img.crop(
                (
                    0,
                    int(img.height * 0.42),
                    img.width,
                    int(img.height * 0.88),
                )
            ),
        )
    )

    processed = []
    for name, view in variants:
        gray = ImageOps.grayscale(view)
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        gray = ImageEnhance.Sharpness(gray).enhance(1.6)
        gray = gray.filter(ImageFilter.SHARPEN)

        # Upscale smaller text regions to help Tesseract.
        if gray.width < 1800:
            scale = min(2.0, 1800 / max(1, gray.width))
            if scale > 1.05:
                gray = gray.resize(
                    (
                        int(gray.width * scale),
                        int(gray.height * scale),
                    ),
                    Image.Resampling.LANCZOS,
                )

        processed.append((name, gray))

    return processed


def ocr_words(path):
    words = []

    for name, img in preprocess_variants(path):
        if name == "top":
            psm = 6
        elif name == "bottom":
            psm = 11
        else:
            psm = 11

        data = pytesseract.image_to_data(
            img,
            lang="eng",
            config=f"--oem 1 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

        count = len(data.get("text", []))
        for i in range(count):
            raw = data["text"][i]
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1

            # Keep even moderate confidence because Pokemon names/card attacks
            # are unusual words and the package's vocabulary will filter them.
            if conf < 15:
                continue

            word = normalize_word(raw)
            if len(word) < 2:
                continue

            words.append(word)

    # Preserve repeats: reference matching uses word frequency.
    return words


def serialize_card(card):
    # pokemontcgsdk Card objects vary slightly by SDK version,
    # so use getattr throughout.
    set_obj = getattr(card, "set", None)

    set_name = None
    set_id = None
    printed_total = None

    if set_obj is not None:
        set_name = getattr(set_obj, "name", None)
        set_id = getattr(set_obj, "id", None)
        printed_total = (
            getattr(set_obj, "printedTotal", None)
            or getattr(set_obj, "total", None)
        )

    return {
        "name": getattr(card, "name", None),
        "number": getattr(card, "number", None),
        "id": getattr(card, "id", None),
        "set": set_name,
        "set_id": set_id,
        "printed_total": printed_total,
    }


@app.get("/")
def home():
    return Response(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Chaos Chaser Recognizer Test</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0d0d10;color:#fff;font-family:Arial,Helvetica,sans-serif}
.wrap{max-width:620px;margin:auto;padding:28px 18px 50px}
h1{font-size:28px;margin:4px 0 8px}
.sub{color:#aaa;margin:0 0 24px;line-height:1.45}
.card{background:#17181d;border:1px solid #292a31;border-radius:24px;padding:18px;box-shadow:0 12px 40px #0005}
.drop{display:block;border:2px dashed #3a3b44;border-radius:20px;padding:28px 16px;text-align:center;cursor:pointer;background:#111217}
.drop strong{display:block;font-size:19px;margin-bottom:7px}.drop span{color:#999;font-size:14px}
input{display:none}.preview{width:100%;max-height:480px;object-fit:contain;border-radius:16px;margin-top:16px;display:none;background:#09090b}
button{width:100%;margin-top:16px;border:0;border-radius:17px;padding:16px;font-size:18px;font-weight:700;color:#fff;background:#e20b43;cursor:pointer}
button:disabled{opacity:.45}.status{margin-top:16px;padding:14px;border-radius:14px;background:#101116;color:#bbb;display:none}
.result{margin-top:16px;padding:18px;border-radius:18px;background:#101116;display:none}
.name{font-size:26px;font-weight:800}.meta{color:#bbb;margin-top:7px}.conf{margin-top:12px;font-weight:700}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#aaa;font-size:12px;margin-top:18px}
.small{font-size:12px;color:#777;margin-top:18px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
<h1>Chaos Chaser Scanner Test</h1>
<p class="sub">Upload or photograph one Pokémon card. The live Light recognizer will try to identify the exact printing.</p>
<div class="card">
<label class="drop">
<strong>📷 Choose Pokémon card</strong>
<span>Tap here to use camera or gallery</span>
<input id="file" type="file" accept="image/*" capture="environment">
</label>
<img id="preview" class="preview">
<button id="go" disabled>Recognize card</button>
<div id="status" class="status">Reading card… the first scan after Render sleeps can take longer.</div>
<div id="result" class="result"></div>
</div>
<div class="small">Chaos Chaser Recognizer Light V2.5 · test interface</div>
</div>
<script>
const f=document.getElementById('file'),p=document.getElementById('preview'),
b=document.getElementById('go'),s=document.getElementById('status'),r=document.getElementById('result');
f.onchange=()=>{if(!f.files[0])return;p.src=URL.createObjectURL(f.files[0]);p.style.display='block';b.disabled=false;r.style.display='none'};
b.onclick=async()=>{
 if(!f.files[0])return;
 b.disabled=true;s.style.display='block';r.style.display='none';
 const fd=new FormData();fd.append('image',f.files[0],f.files[0].name||'card.jpg');
 try{
  const res=await fetch('/recognize',{method:'POST',body:fd});
  const raw=await res.text();

  let data=null;
  try{
    data=JSON.parse(raw);
  }catch{
    data=null;
  }

  if(data?.ok&&data.card){
   const c=data.card;
   r.innerHTML='<div class="name">'+(c.name||'Unknown')+'</div>'+
   '<div class="meta">'+(c.set||'Unknown set')+' · #'+(c.number||'?')+'</div>'+
   '<div class="conf">Confidence: '+Math.round((data.confidence||0)*100)+'%</div>'+
   '<pre>'+JSON.stringify(data,null,2)+'</pre>';
  }else if(data){
   r.innerHTML='<div class="name">Recognizer error</div>'+
   '<div class="meta">HTTP '+res.status+'</div>'+
   '<pre>'+JSON.stringify(data,null,2)+'</pre>';
  }else{
   /*
     V2.5: Render/proxy sometimes returns an HTML error page.
     Show it instead of throwing Unexpected token '<'.
   */
   const safe=raw
      .replace(/&/g,'&amp;')
      .replace(/</g,'&lt;')
      .replace(/>/g,'&gt;');

   r.innerHTML='<div class="name">Server returned HTML</div>'+
   '<div class="meta">HTTP '+res.status+' · this is the real Render response</div>'+
   '<pre>'+safe.slice(0,12000)+'</pre>';
  }

  r.style.display='block';
 }catch(e){
  r.innerHTML='<div class="name">Request failed</div><pre>'+String(e)+'</pre>';r.style.display='block';
 }
 s.style.display='none';b.disabled=false;
};
</script>
</body>
</html>""", mimetype="text/html")



@app.get("/diagnostics")
def diagnostics():
    info = {
        "ok": True,
        "python": os.sys.version,
        "tesseract": None,
        "package_found": False,
        "master_reference_exists": False,
        "classifier_loaded": _classifier is not None,
    }

    try:
        info["tesseract"] = str(pytesseract.get_tesseract_version())
    except Exception as exc:
        info["tesseract_error"] = str(exc)

    try:
        spec = importlib.util.find_spec("pokemon_card_recognizer")
        info["package_found"] = bool(spec)
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            p = root / "reference" / "data" / "ref_build" / "master.pkl"
            info["master_reference_exists"] = p.exists()
            info["master_reference_path"] = str(p)
    except Exception as exc:
        info["reference_error"] = str(exc)

    return jsonify(info)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "status": "healthy",
            "engine": "tesseract-reference-light",
        }
    )


@app.post("/warmup")
def warmup():
    try:
        classifier = get_classifier()
        return jsonify(
            {
                "ok": True,
                "ready": True,
                "reference": classifier.reference.name,
                "cards": len(classifier.reference.cards),
                "vocab_size": classifier.reference.vocab.size,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return (
            jsonify(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                }
            ),
            500,
        )


@app.post("/recognize")
def recognize():
    if "image" not in request.files:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing_image",
                    "detail": "Send multipart/form-data field named image.",
                }
            ),
            400,
        )

    upload = request.files["image"]
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "empty_upload"}), 400

    suffix = Path(secure_filename(upload.filename)).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            upload.save(tmp)
            tmp_path = tmp.name

        with _run_lock:
            classifier = get_classifier()

            words = ocr_words(tmp_path)
            if not words:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "no_text",
                            "detail": "Tesseract could not read useful text.",
                        }
                    ),
                    422,
                )

            # This is the SAME WordClassifier/reference logic used by the
            # original project. Only the OCR engine was changed.
            result = classifier.classify(
                [words],
                include_probs=True,
                mechanism="sequential",
            )

            if result is None or len(result) == 0:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "no_prediction",
                            "ocr_words": words[:80],
                        }
                    ),
                    422,
                )

            pred = result[0]
            card = classifier.reference.lookup_card_prediction(pred)

            top = []
            if pred.all_probs is not None:
                probs = pred.all_probs
                # Return the best few predictions for debugging/testing.
                ranked = sorted(
                    range(len(probs)),
                    key=lambda i: probs[i],
                    reverse=True,
                )[:5]

                for idx in ranked:
                    try:
                        c = classifier.reference.lookup_by_index(idx)
                        top.append(
                            {
                                "card": serialize_card(c),
                                "score": float(probs[idx]),
                            }
                        )
                    except Exception:
                        pass

        return jsonify(
            {
                "ok": True,
                "card": serialize_card(card),
                "confidence": float(pred.conf),
                "top_matches": top,
                "ocr_words": words[:100],
                "engine": "tesseract + bundled master reference",
            }
        )

    except Exception as exc:
        traceback.print_exc()
        return (
            jsonify(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                }
            ),
            500,
        )

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "ok": False,
        "error": "image_too_large",
        "detail": "Uploaded image is too large for this server."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    # Flask-side unhandled exceptions should still be JSON.
    return jsonify({
        "ok": False,
        "error": "internal_server_error",
        "detail": str(error)
    }), 500


@app.errorhandler(Exception)
def catch_all_error(error):
    traceback.print_exc()
    return jsonify({
        "ok": False,
        "error": type(error).__name__,
        "detail": str(error)
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
