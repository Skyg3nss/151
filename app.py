import os
import re
import tempfile
import traceback
from pathlib import Path
from urllib.parse import quote

import requests
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

TCGDEX = "https://api.tcgdex.net/v2/en"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "ChaosChaserRecognizer/3.0"})

# ---------- Helpers ----------

def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def norm_name(s):
    s = clean_text(s).lower()
    s = re.sub(r"[^a-z0-9 .'\-]", " ", s)
    s = re.sub(r"\b(hp|basic|stage|evolves|from)\b", " ", s)
    s = re.sub(r"\b\d{2,3}\b", " ", s)
    return clean_text(s)

def normalize_local_id(s):
    s = str(s or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("O","0")
    if s.isdigit():
        s = str(int(s))
    return s

def preprocess(img, contrast=2.0, sharpen=1.8, upscale=2.0):
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = ImageEnhance.Sharpness(gray).enhance(sharpen)
    gray = gray.filter(ImageFilter.SHARPEN)
    if upscale > 1:
        gray = gray.resize(
            (int(gray.width*upscale), int(gray.height*upscale)),
            Image.Resampling.LANCZOS
        )
    return gray

def ocr_image(img, psm=6, whitelist=None):
    cfg = f"--oem 1 --psm {psm}"
    if whitelist:
        cfg += f' -c tessedit_char_whitelist="{whitelist}"'
    return pytesseract.image_to_string(img, lang="eng", config=cfg)

def number_candidates(text):
    text = clean_text(text)
    text = text.replace("|","/").replace("\\","/")
    text = re.sub(r"(?<=\d)[Oo](?=\d|/)", "0", text)
    text = re.sub(r"(?<=\d)[Il](?=\d|/)", "1", text)

    out = []
    seen = set()

    def add(num, den=None, raw=""):
        local = normalize_local_id(num)
        if not local:
            return
        denom = int(den) if den and str(den).isdigit() else None
        key = (local, denom)
        if key in seen:
            return
        seen.add(key)
        out.append({"localId": local, "denominator": denom, "raw": raw})

    for m in re.finditer(r"\b([A-Z]{0,3}\d{1,3}[A-Z]{0,2})\s*/\s*(\d{2,3})\b", text, re.I):
        add(m.group(1), m.group(2), m.group(0))

    for m in re.finditer(r"\b(\d{1,3})\s+(\d{2,3})\b", text):
        add(m.group(1), m.group(2), m.group(0))

    for m in re.finditer(r"\b([A-Z]{0,3}\d{1,3}[A-Z]{0,2})\b", text, re.I):
        add(m.group(1), None, m.group(0))

    return out

def name_queries(text):
    """
    Generate a few likely card-name queries from top OCR.
    We keep this intentionally small to avoid API spam.
    """
    lines = [norm_name(x) for x in str(text or "").splitlines()]
    lines = [x for x in lines if x and len(x) >= 3]

    phrases = []
    for line in lines[:5]:
        # Keep ex/GX/V/VSTAR suffixes if OCR saw them.
        words = line.split()
        for n in (4,3,2,1):
            if len(words) >= n:
                phrases.append(" ".join(words[:n]))

    # Also try a globally cleaned version.
    all_clean = norm_name(text)
    if all_clean:
        words = all_clean.split()
        for n in (4,3,2,1):
            if len(words) >= n:
                phrases.append(" ".join(words[:n]))

    # De-dupe longest first.
    unique = []
    seen = set()
    for q in phrases:
        q = clean_text(q)
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    unique.sort(key=len, reverse=True)
    return unique[:8]

def tcgdex_card_briefs(query):
    r = HTTP.get(f"{TCGDEX}/cards", params={"name": query}, timeout=8)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

def tcgdex_card(card_id):
    r = HTTP.get(f"{TCGDEX}/cards/{quote(str(card_id), safe='')}", timeout=8)
    r.raise_for_status()
    return r.json()

def exact_name_score(ocr_query, card_name):
    a = norm_name(ocr_query)
    b = norm_name(card_name)
    if not a or not b:
        return 0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.88
    aw, bw = set(a.split()), set(b.split())
    inter = len(aw & bw)
    union = max(1, len(aw | bw))
    return inter / union

def recognize_card(path):
    img = Image.open(path).convert("RGB")

    # Limit huge phone shots for predictable RAM.
    if img.width > 1600:
        ratio = 1600 / img.width
        img = img.resize(
            (1600, max(1, int(img.height*ratio))),
            Image.Resampling.LANCZOS
        )

    w, h = img.size

    # Name is at the very top. Use two slightly different crops.
    top1 = img.crop((0, 0, w, int(h*0.20)))
    top2 = img.crop((0, 0, w, int(h*0.27)))

    top_texts = [
        ocr_image(preprocess(top1, 2.1, 2.0, 2.2), psm=6),
        ocr_image(preprocess(top2, 1.8, 1.7, 1.8), psm=11),
    ]
    top_text = "\n".join(top_texts)

    # Collector number: bottom strip only.
    bottom = img.crop((0, int(h*0.84), w, h))
    bottom2 = img.crop((0, int(h*0.90), w, h))
    bottom_text = "\n".join([
        ocr_image(
            preprocess(bottom, 2.4, 2.0, 2.4),
            psm=11,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/|- "
        ),
        ocr_image(
            preprocess(bottom2, 2.2, 2.0, 2.8),
            psm=11,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/|- "
        )
    ])

    nums = number_candidates(bottom_text)
    queries = name_queries(top_text)

    if not queries:
        return {
            "ok": False,
            "error": "name_ocr_failed",
            "ocr": {"top": top_text, "bottom": bottom_text}
        }

    # Gather candidate briefs from the best OCR name guesses.
    briefs = {}
    used_queries = []
    for q in queries:
        try:
            rows = tcgdex_card_briefs(q)
        except Exception:
            continue
        if rows:
            used_queries.append(q)
        for row in rows[:80]:
            cid = row.get("id")
            if cid:
                briefs[cid] = row
        # Stop once we have a reasonable candidate pool.
        if len(briefs) >= 12:
            break

    if not briefs:
        return {
            "ok": False,
            "error": "no_name_candidates",
            "queries": queries,
            "ocr": {"top": top_text, "bottom": bottom_text}
        }

    # Score candidates by name first.
    scored = []
    for row in briefs.values():
        best_name = max(exact_name_score(q, row.get("name")) for q in used_queries or queries)
        scored.append((best_name, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_name_score = scored[0][0]
    # Keep candidates close to best name match.
    shortlist = [row for score,row in scored if score >= max(0.45, best_name_score-0.18)][:18]

    # Number validation against REAL TCGdex card metadata.
    exact = []
    enriched = []

    for brief in shortlist:
        try:
            card = tcgdex_card(brief["id"])
        except Exception:
            continue

        local_id = normalize_local_id(card.get("localId"))
        set_info = card.get("set") or {}
        counts = set_info.get("cardCount") or {}
        official = counts.get("official")
        total = counts.get("total")

        num_score = 0.0
        matched_num = None
        for n in nums:
            if normalize_local_id(n["localId"]) != local_id:
                continue

            matched_num = n
            if n["denominator"] is not None:
                if n["denominator"] in (official, total):
                    num_score = 1.0
                    break
                else:
                    # Wrong denominator: don't trust it.
                    continue
            else:
                num_score = max(num_score, 0.72)

        name_score = max(exact_name_score(q, card.get("name")) for q in used_queries or queries)
        combined = name_score*0.62 + num_score*0.38

        item = {
            "id": card.get("id"),
            "name": card.get("name"),
            "number": card.get("localId"),
            "set": set_info.get("name"),
            "set_id": set_info.get("id"),
            "set_official_count": official,
            "set_total_count": total,
            "image": card.get("image"),
            "name_score": round(name_score, 4),
            "number_score": round(num_score, 4),
            "score": round(combined, 4),
            "matched_number_ocr": matched_num,
        }

        enriched.append(item)
        if name_score >= 0.82 and num_score >= 0.99:
            exact.append(item)

    if exact:
        exact.sort(key=lambda x: x["score"], reverse=True)
        winner = exact[0]
        confidence = min(0.995, 0.93 + (winner["name_score"]-0.82)*0.3)
        return {
            "ok": True,
            "mode": "name+number_exact",
            "card": winner,
            "confidence": round(confidence, 4),
            "top_matches": exact[:5],
            "ocr": {"top": top_text, "bottom": bottom_text},
            "queries": used_queries,
            "number_candidates": nums,
        }

    enriched.sort(key=lambda x: x["score"], reverse=True)

    if not enriched:
        return {
            "ok": False,
            "error": "candidate_lookup_failed",
            "ocr": {"top": top_text, "bottom": bottom_text},
            "queries": used_queries,
            "number_candidates": nums,
        }

    # If number was unavailable, return candidates rather than pretending exact certainty.
    winner = enriched[0]
    return {
        "ok": True,
        "mode": "name_candidate",
        "card": winner,
        "confidence": round(min(0.89, winner["score"]), 4),
        "needs_visual_confirmation": winner["number_score"] < 0.99,
        "top_matches": enriched[:8],
        "ocr": {"top": top_text, "bottom": bottom_text},
        "queries": used_queries,
        "number_candidates": nums,
    }

# ---------- UI ----------

@app.get("/")
def home():
    return Response(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Chaos Chaser Scanner Test</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0d0d10;color:#fff;font-family:Arial,Helvetica,sans-serif}
.wrap{max-width:620px;margin:auto;padding:28px 18px 50px}h1{font-size:28px;margin:4px 0 8px}
.sub{color:#aaa;line-height:1.45;margin:0 0 24px}.card{background:#17181d;border:1px solid #292a31;border-radius:24px;padding:18px}
.drop{display:block;border:2px dashed #3a3b44;border-radius:20px;padding:28px 16px;text-align:center;cursor:pointer;background:#111217}
.drop strong{display:block;font-size:19px;margin-bottom:7px}.drop span{color:#999;font-size:14px}input{display:none}
.preview{width:100%;max-height:480px;object-fit:contain;border-radius:16px;margin-top:16px;display:none;background:#09090b}
button{width:100%;margin-top:16px;border:0;border-radius:17px;padding:16px;font-size:18px;font-weight:700;color:white;background:#e20b43}
button:disabled{opacity:.45}.status{margin-top:16px;padding:14px;border-radius:14px;background:#101116;color:#bbb;display:none}
.result{margin-top:16px;padding:18px;border-radius:18px;background:#101116;display:none}.name{font-size:26px;font-weight:800}
.meta{color:#bbb;margin-top:7px}.conf{margin-top:12px;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#aaa;font-size:12px;margin-top:18px}
.small{font-size:12px;color:#777;margin-top:18px;text-align:center}
</style>
</head>
<body><div class="wrap">
<h1>Chaos Chaser Scanner Test V3</h1>
<p class="sub">Ultra-light test: Tesseract reads name + collector number, then TCGdex verifies the exact real printing.</p>
<div class="card">
<label class="drop"><strong>📷 Choose Pokémon card</strong><span>Camera or gallery</span>
<input id="file" type="file" accept="image/*" capture="environment"></label>
<img id="preview" class="preview">
<button id="go" disabled>Recognize card</button>
<div id="status" class="status">Reading name + collector number…</div>
<div id="result" class="result"></div>
</div><div class="small">No PyTorch · no EasyOCR · no master.pkl</div>
</div>
<script>
const f=document.getElementById('file'),p=document.getElementById('preview'),b=document.getElementById('go'),
s=document.getElementById('status'),r=document.getElementById('result');
f.onchange=()=>{if(!f.files[0])return;p.src=URL.createObjectURL(f.files[0]);p.style.display='block';b.disabled=false;r.style.display='none'};
b.onclick=async()=>{
 if(!f.files[0])return;b.disabled=true;s.style.display='block';r.style.display='none';
 const fd=new FormData();fd.append('image',f.files[0],f.files[0].name||'card.jpg');
 try{
  const res=await fetch('/recognize',{method:'POST',body:fd});
  const raw=await res.text();let data=null;try{data=JSON.parse(raw)}catch{}
  if(data?.ok&&data.card){
   const c=data.card;
   r.innerHTML='<div class="name">'+(c.name||'Unknown')+'</div>'+
   '<div class="meta">'+(c.set||'Unknown set')+' · #'+(c.number||'?')+'</div>'+
   '<div class="conf">'+data.mode+' · '+Math.round((data.confidence||0)*100)+'%</div>'+
   '<pre>'+JSON.stringify(data,null,2)+'</pre>';
  }else if(data){
   r.innerHTML='<div class="name">No exact match</div><pre>'+JSON.stringify(data,null,2)+'</pre>';
  }else{
   r.innerHTML='<div class="name">Server error</div><div class="meta">HTTP '+res.status+'</div><pre>'+raw.replace(/</g,'&lt;').slice(0,10000)+'</pre>';
  }
  r.style.display='block';
 }catch(e){r.innerHTML='<div class="name">Request failed</div><pre>'+String(e)+'</pre>';r.style.display='block'}
 s.style.display='none';b.disabled=false;
};
</script></body></html>""", mimetype="text/html")

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "status": "healthy",
        "engine": "tesseract+tcgdex",
        "pytorch": False,
        "easyocr": False,
        "master_reference": False,
    })

@app.post("/recognize")
def recognize():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "missing_image"}), 400

    upload = request.files["image"]
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "empty_upload"}), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp"}:
        suffix = ".jpg"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            upload.save(tmp)
            tmp_path = tmp.name
        return jsonify(recognize_card(tmp_path))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc)
        }), 500
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except OSError: pass

if __name__ == "__main__":
    port=int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0",port=port)
