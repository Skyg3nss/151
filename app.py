import os
import re
import math
import tempfile
import traceback
from pathlib import Path
from urllib.parse import quote
from functools import lru_cache
from difflib import SequenceMatcher

import requests
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops
import pytesseract
from pytesseract import Output

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

TCGDEX = "https://api.tcgdex.net/v2/en"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "ChaosChaserRecognizer/4.0"})

# ============================================================
# Text / OCR helpers
# ============================================================

STOP_WORDS = {
    "hp","basic","stage","stage1","stage2","evolves","from","pokemon","card",
    "the","of","and","for","with","this","that","into","your","you","opponent",
    "damage","energy","weakness","resistance","retreat","illustrator","illus",
    # Common header OCR junk from physical cards / borders.
    "pe","pee","oq","ing","nos","eee","cay","zl","wy","ang"
}

def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def normalize_word(s):
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9.'\-]", "", s).strip(".'-")
    return s

def norm_name(s):
    s = clean_text(s).lower()
    s = re.sub(r"[^a-z0-9 .'\-]", " ", s)
    s = re.sub(r"\b(hp|basic|stage|stage1|stage2|evolves|from)\b", " ", s)
    s = re.sub(r"\b\d{2,3}\b", " ", s)
    return clean_text(s)

def normalize_local_id(s):
    s = str(s or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    # OCR confusions only when in a number-like token.
    s = s.replace("O", "0")
    if s.isdigit():
        s = str(int(s))
    return s

def add_white_border(img, px=24):
    return ImageOps.expand(img, border=px, fill="white")

def prepare_gray(img, scale=2.0, contrast=2.0, sharp=1.7):
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = ImageEnhance.Sharpness(gray).enhance(sharp)
    gray = gray.filter(ImageFilter.SHARPEN)
    if scale > 1.0:
        gray = gray.resize(
            (max(1, int(gray.width*scale)), max(1, int(gray.height*scale))),
            Image.Resampling.LANCZOS
        )
    return add_white_border(gray, 24)

def prepare_binary(img, scale=2.3, threshold=165):
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    if scale > 1.0:
        gray = gray.resize(
            (max(1, int(gray.width*scale)), max(1, int(gray.height*scale))),
            Image.Resampling.LANCZOS
        )
    bw = gray.point(lambda p: 255 if p > threshold else 0)
    return add_white_border(bw, 28)

def ocr_data(img, psm=7, whitelist=None):
    cfg = f"--oem 1 --psm {psm}"
    if whitelist:
        cfg += f' -c tessedit_char_whitelist="{whitelist}"'
    return pytesseract.image_to_data(
        img,
        lang="eng",
        config=cfg,
        output_type=Output.DICT
    )

def ocr_text(img, psm=11, whitelist=None):
    cfg = f"--oem 1 --psm {psm}"
    if whitelist:
        cfg += f' -c tessedit_char_whitelist="{whitelist}"'
    return pytesseract.image_to_string(img, lang="eng", config=cfg)

def useful_tokens_from_data(data, min_conf=12):
    out = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = data["text"][i]
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if conf < min_conf:
            continue
        word = normalize_word(raw)
        if len(word) < 2:
            continue
        out.append((word, conf))
    return out

def extract_name_signals(img):
    """
    Multiple small header passes, specifically optimized for card names.
    Tesseract docs recommend different PSM modes for small regions; use a
    single-line pass plus sparse-text pass, with rescale/binarization/borders.
    """
    w, h = img.size

    header_tight = img.crop((int(w*.05), int(h*.015), int(w*.83), int(h*.155)))
    header_wide  = img.crop((0, 0, w, int(h*.235)))

    variants = [
        ("tight_line", prepare_gray(header_tight, 2.4, 2.0, 1.9), 7),
        ("tight_binary", prepare_binary(header_tight, 2.6, 160), 7),
        ("wide_sparse", prepare_gray(header_wide, 1.9, 1.8, 1.6), 11),
    ]

    all_tokens = []
    lines = []

    for label, view, psm in variants:
        data = ocr_data(view, psm=psm)
        tokens = useful_tokens_from_data(data, min_conf=10)
        all_tokens.append({"label": label, "tokens": tokens})

        line = " ".join(word for word, _ in tokens)
        if line:
            lines.append(line)

    # Aggregate token evidence across independent crops.
    scores = {}
    counts = {}
    best_conf = {}
    first = {}
    pos = 0

    for pass_info in all_tokens:
        seen_this_pass = set()
        for word, conf in pass_info["tokens"]:
            if len(word) < 3 or len(word) > 24:
                continue
            if word in STOP_WORDS or word.isdigit():
                continue
            # Reject ugly digit-heavy OCR words.
            if sum(ch.isdigit() for ch in word) > 1:
                continue

            first.setdefault(word, pos)
            pos += 1
            best_conf[word] = max(best_conf.get(word, 0), conf)

            # Repetition across independent OCR passes matters more than one
            # lucky high-confidence garbage token.
            if word not in seen_this_pass:
                counts[word] = counts.get(word, 0) + 1
                seen_this_pass.add(word)

    for word in counts:
        repeat = counts[word]
        conf = best_conf.get(word, 0) / 100.0
        alpha_ratio = sum(c.isalpha() for c in word) / max(1, len(word))

        score = repeat*12 + conf*5 + alpha_ratio*3
        if 4 <= len(word) <= 14:
            score += 2
        if repeat >= 2:
            score += 8
        if len(word) >= 18:
            score -= 4

        scores[word] = score

    ranked_words = sorted(
        scores,
        key=lambda word: (-scores[word], first.get(word, 9999))
    )

    queries = []
    seen = set()

    def add(q):
        q = norm_name(q)
        if not q or q in seen:
            return
        seen.add(q)
        queries.append(q)

    # Strong standalone words first. This is what makes Kakuna win over junk.
    for word in ranked_words:
        if counts.get(word, 0) >= 2:
            add(word)
    for word in ranked_words:
        add(word)

    # Multi-word names: derive compact windows from OCR lines.
    for line in lines:
        words = [
            normalize_word(x) for x in line.split()
            if normalize_word(x) and normalize_word(x) not in STOP_WORDS
        ]
        words = [w for w in words if 2 <= len(w) <= 24 and not w.isdigit()]
        for size in (4, 3, 2):
            if len(words) < size:
                continue
            for i in range(len(words)-size+1):
                add(" ".join(words[i:i+size]))

    return {
        "queries": queries[:18],
        "ranked_words": [
            {
                "word": w,
                "passes": counts.get(w, 0),
                "confidence": round(best_conf.get(w, 0), 1),
                "score": round(scores.get(w, 0), 2),
            }
            for w in ranked_words[:18]
        ],
        "lines": lines,
    }

def number_candidates(text):
    text = clean_text(text)
    text = text.replace("|", "/").replace("\\", "/")
    text = re.sub(r"(?<=\d)[Oo](?=\d|/)", "0", text)
    text = re.sub(r"(?<=\d)[Il](?=\d|/)", "1", text)

    out = []
    seen = set()

    def add(num, den=None, raw=""):
        local = normalize_local_id(num)
        if not local:
            return
        denom = int(den) if den and str(den).isdigit() else None
        if denom is not None and not (5 <= denom <= 999):
            return
        key = (local, denom)
        if key in seen:
            return
        seen.add(key)
        out.append({"localId": local, "denominator": denom, "raw": raw})

    # Highest confidence: explicit collector-number slash.
    for m in re.finditer(
        r"\b([A-Z]{0,4}\d{1,3}[A-Z]{0,3})\s*/\s*(\d{2,3})\b", text, re.I
    ):
        add(m.group(1), m.group(2), m.group(0))

    # OCR may drop the slash but preserve separation.
    for m in re.finditer(r"\b(\d{1,3})\s+(\d{2,3})\b", text):
        add(m.group(1), m.group(2), m.group(0))

    # Numerator only, useful after name filtering.
    for m in re.finditer(r"\b([A-Z]{0,4}\d{1,3}[A-Z]{0,3})\b", text, re.I):
        add(m.group(1), None, m.group(0))

    return out

def extract_number_signals(img):
    w, h = img.size

    # Modern cards usually print collector info along the lower-left band.
    # Also keep a wider fallback for older layouts.
    crops = [
        ("bottom_left_tight", img.crop((0, int(h*.875), int(w*.73), h))),
        ("bottom_left_low",   img.crop((0, int(h*.91),  int(w*.82), h))),
        ("bottom_wide",       img.crop((0, int(h*.835), w, h))),
    ]

    texts = []

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/|- "

    for label, crop in crops:
        # Two preprocessing styles for the tight region; one for others.
        if label == "bottom_left_tight":
            views = [
                (prepare_gray(crop, 2.7, 2.3, 2.0), 11),
                (prepare_binary(crop, 2.9, 160), 11),
            ]
        else:
            views = [(prepare_gray(crop, 2.4, 2.1, 1.8), 11)]

        for view, psm in views:
            txt = ocr_text(view, psm=psm, whitelist=whitelist)
            if clean_text(txt):
                texts.append(txt)

    joined = "\n".join(texts)
    return {
        "text": joined,
        "candidates": number_candidates(joined),
    }

def extract_body_text(img):
    w, h = img.size
    # Ability / attack names are a powerful tie-breaker between same-name
    # printings if collector number OCR fails.
    mid = img.crop((int(w*.03), int(h*.36), int(w*.97), int(h*.84)))
    view = prepare_gray(mid, 1.7, 1.7, 1.5)
    text = ocr_text(view, psm=11)
    return clean_text(text)

def extract_hp(name_lines):
    joined = " ".join(name_lines)
    vals = []
    for m in re.finditer(r"\b(\d{2,3})\b", joined):
        n = int(m.group(1))
        if 20 <= n <= 400 and n % 10 == 0:
            vals.append(n)
    return vals[:6]

# ============================================================
# TCGdex + candidate scoring
# ============================================================

@lru_cache(maxsize=128)
def tcgdex_card_briefs(query):
    r = HTTP.get(f"{TCGDEX}/cards", params={"name": query}, timeout=8)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

@lru_cache(maxsize=512)
def tcgdex_card(card_id):
    r = HTTP.get(f"{TCGDEX}/cards/{quote(str(card_id), safe='')}", timeout=8)
    r.raise_for_status()
    return r.json()

def similarity(a, b):
    a = norm_name(a)
    b = norm_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.90
    seq = SequenceMatcher(None, a, b).ratio()
    aw, bw = set(a.split()), set(b.split())
    tok = len(aw & bw) / max(1, len(aw | bw))
    return max(seq*0.88, tok)

def token_overlap_score(ocr_text, reference_text):
    def toks(s):
        return {
            normalize_word(x)
            for x in str(s or "").split()
            if len(normalize_word(x)) >= 4
            and normalize_word(x) not in STOP_WORDS
        }

    a, b = toks(ocr_text), toks(reference_text)
    if not a or not b:
        return 0.0

    exact = len(a & b)
    # A few attack/ability terms matching is extremely informative.
    base = exact / max(1, min(len(a), len(b), 8))

    # Also reward fuzzy token pairs for OCR spelling errors.
    fuzzy_hits = 0
    for x in a:
        if any(SequenceMatcher(None, x, y).ratio() >= .79 for y in b):
            fuzzy_hits += 1
    fuzzy = fuzzy_hits / max(1, min(len(a), 8))

    return min(1.0, base*.55 + fuzzy*.45)

def card_reference_text(card):
    parts = [card.get("name", "")]

    for ability in card.get("abilities") or []:
        parts.extend([ability.get("name", ""), ability.get("effect", "")])

    for attack in card.get("attacks") or []:
        parts.extend([
            attack.get("name", ""),
            attack.get("effect", ""),
            str(attack.get("damage", "")),
        ])

    return " ".join(parts)

def find_name_candidates(signals):
    queries = signals["queries"]
    briefs = {}
    successful = []

    # Query strong names first. Stop early when TCGdex gives an exact name hit.
    for q in queries:
        try:
            rows = tcgdex_card_briefs(q)
        except Exception:
            continue
        if not rows:
            continue

        rows = sorted(
            rows,
            key=lambda row: similarity(q, row.get("name", "")),
            reverse=True
        )
        successful.append(q)

        for row in rows[:90]:
            if row.get("id"):
                briefs[row["id"]] = row

        best = max((similarity(q, row.get("name", "")) for row in rows), default=0)
        # A standalone exact name such as "kakuna" is all we need.
        if best >= .985:
            break

        if len(briefs) >= 24:
            break

    return list(briefs.values()), successful

def score_candidates(img):
    name_sig = extract_name_signals(img)
    num_sig = extract_number_signals(img)
    body_text = extract_body_text(img)
    hp_candidates = extract_hp(name_sig["lines"])

    briefs, used_queries = find_name_candidates(name_sig)

    if not briefs:
        return {
            "ok": False,
            "error": "no_name_candidates",
            "queries": name_sig["queries"],
            "ranked_name_words": name_sig["ranked_words"],
            "ocr": {
                "header_lines": name_sig["lines"],
                "bottom": num_sig["text"],
                "body": body_text,
            }
        }

    # First rank briefs by name and keep a small pool.
    brief_ranked = []
    for brief in briefs:
        name_score = max(
            (similarity(q, brief.get("name", "")) for q in used_queries or name_sig["queries"]),
            default=0
        )
        brief_ranked.append((name_score, brief))

    brief_ranked.sort(key=lambda x: x[0], reverse=True)
    best_name = brief_ranked[0][0]
    shortlist = [
        row for score, row in brief_ranked
        if score >= max(.54, best_name-.16)
    ][:24]

    rows = []

    for brief in shortlist:
        try:
            card = tcgdex_card(brief["id"])
        except Exception:
            continue

        set_info = card.get("set") or {}
        counts = set_info.get("cardCount") or {}
        local_id = normalize_local_id(card.get("localId"))
        official = counts.get("official")
        total = counts.get("total")

        name_score = max(
            (similarity(q, card.get("name", "")) for q in used_queries or name_sig["queries"]),
            default=0
        )

        number_score = 0.0
        number_exact = False
        number_evidence = None

        for n in num_sig["candidates"]:
            if normalize_local_id(n["localId"]) != local_id:
                continue

            if n["denominator"] is not None:
                if n["denominator"] in (official, total):
                    number_score = 1.0
                    number_exact = True
                    number_evidence = n
                    break
                # Explicit wrong denominator does NOT count.
                continue

            # Numerator only: useful, but never enough by itself to claim exact.
            if number_score < .68:
                number_score = .68
                number_evidence = n

        hp_score = 0.0
        card_hp = card.get("hp")
        if hp_candidates and card_hp:
            try:
                hp_int = int(card_hp)
                if hp_int in hp_candidates:
                    hp_score = 1.0
            except Exception:
                pass

        body_score = token_overlap_score(body_text, card_reference_text(card))

        # Exact printed number is king. Body text is a strong same-name
        # tie-breaker. HP is useful but not unique.
        combined = (
            name_score*.43 +
            number_score*.37 +
            body_score*.15 +
            hp_score*.05
        )

        rows.append({
            "id": card.get("id"),
            "name": card.get("name"),
            "number": card.get("localId"),
            "set": set_info.get("name"),
            "set_id": set_info.get("id"),
            "set_official_count": official,
            "set_total_count": total,
            "image": card.get("image"),
            "hp": card_hp,
            "name_score": round(name_score, 4),
            "number_score": round(number_score, 4),
            "body_score": round(body_score, 4),
            "hp_score": round(hp_score, 4),
            "score": round(combined, 4),
            "number_exact": number_exact,
            "matched_number_ocr": number_evidence,
        })

    rows.sort(key=lambda x: x["score"], reverse=True)

    if not rows:
        return {
            "ok": False,
            "error": "candidate_lookup_failed",
            "queries": used_queries,
            "ocr": {
                "header_lines": name_sig["lines"],
                "bottom": num_sig["text"],
                "body": body_text,
            }
        }

    winner = rows[0]
    runner = rows[1] if len(rows) > 1 else None
    margin = winner["score"] - (runner["score"] if runner else 0)

    # Confidence policy:
    # - name + real collector denominator => essentially exact
    # - otherwise need a healthy score + margin; never fake 99%.
    if winner["number_exact"] and winner["name_score"] >= .80:
        mode = "name+number_exact"
        confidence = min(.995, .94 + margin*.2)
        exact = True
    elif (
        winner["name_score"] >= .84 and
        winner["body_score"] >= .34 and
        margin >= .08
    ):
        mode = "name+body_strong"
        confidence = min(.94, .72 + winner["body_score"]*.18 + margin*.25)
        exact = False
    else:
        mode = "ranked_candidate"
        confidence = min(.88, max(.35, winner["score"]))
        exact = False

    return {
        "ok": True,
        "mode": mode,
        "exact": exact,
        "card": winner,
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "top_matches": rows[:8],
        "queries": used_queries,
        "ranked_name_words": name_sig["ranked_words"],
        "number_candidates": num_sig["candidates"],
        "hp_candidates": hp_candidates,
        "ocr": {
            "header_lines": name_sig["lines"],
            "bottom": num_sig["text"],
            "body": body_text,
        }
    }

# ============================================================
# Routes / test UI
# ============================================================

@app.get("/")
def home():
    return Response(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Chaos Chaser Scanner Test V4</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0d0d10;color:#fff;font-family:Arial,Helvetica,sans-serif}
.wrap{max-width:650px;margin:auto;padding:28px 18px 50px}h1{font-size:28px;margin:4px 0 8px}
.sub{color:#aaa;line-height:1.45;margin:0 0 24px}.panel{background:#17181d;border:1px solid #292a31;border-radius:24px;padding:18px}
.drop{display:block;border:2px dashed #3a3b44;border-radius:20px;padding:28px 16px;text-align:center;cursor:pointer;background:#111217}
.drop strong{display:block;font-size:19px;margin-bottom:7px}.drop span{color:#999;font-size:14px}input{display:none}
.preview{width:100%;max-height:500px;object-fit:contain;border-radius:16px;margin-top:16px;display:none;background:#09090b}
button{width:100%;margin-top:16px;border:0;border-radius:17px;padding:16px;font-size:18px;font-weight:700;color:#fff;background:#e20b43}
button:disabled{opacity:.45}.status{margin-top:16px;padding:14px;border-radius:14px;background:#101116;color:#bbb;display:none}
.result{margin-top:16px;padding:18px;border-radius:18px;background:#101116;display:none}.name{font-size:27px;font-weight:800}
.meta{color:#bbb;margin-top:7px}.conf{margin-top:12px;font-weight:700}.exact{color:#32d17d}.warn{color:#ffbd4a}
.matches{display:grid;gap:10px;margin-top:16px}.match{display:flex;gap:12px;align-items:center;background:#17181d;padding:10px;border-radius:14px}
.match img{width:48px;height:67px;object-fit:cover;border-radius:5px;background:#222}.match b{display:block}.match small{color:#aaa}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#999;font-size:11px;margin-top:18px;max-height:360px;overflow:auto}
.small{font-size:12px;color:#777;margin-top:18px;text-align:center}
</style>
</head>
<body><div class="wrap">
<h1>Chaos Chaser Scanner Test V4</h1>
<p class="sub">Hybrid exact recognition: name + collector number + HP + attack/ability text, all verified against real TCGdex card data.</p>
<div class="panel">
<label class="drop"><strong>📷 Choose Pokémon card</strong><span>Use a clear photo with the full card visible</span>
<input id="file" type="file" accept="image/*" capture="environment"></label>
<img id="preview" class="preview">
<button id="go" disabled>Recognize card</button>
<div id="status" class="status">Reading card… this can take a few seconds on Render Free.</div>
<div id="result" class="result"></div>
</div>
<div class="small">No PyTorch · no EasyOCR · no large in-memory card database</div>
</div>
<script>
const f=document.getElementById('file'),p=document.getElementById('preview'),b=document.getElementById('go'),
s=document.getElementById('status'),r=document.getElementById('result');
f.onchange=()=>{if(!f.files[0])return;p.src=URL.createObjectURL(f.files[0]);p.style.display='block';b.disabled=false;r.style.display='none'};
const esc=x=>String(x??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
b.onclick=async()=>{
 if(!f.files[0])return;b.disabled=true;s.style.display='block';r.style.display='none';
 const fd=new FormData();fd.append('image',f.files[0],f.files[0].name||'card.jpg');
 try{
  const res=await fetch('/recognize',{method:'POST',body:fd});
  const raw=await res.text();let data=null;try{data=JSON.parse(raw)}catch{}
  if(data?.ok&&data.card){
   const c=data.card;
   const exact=data.exact?'exact':'';
   let html='<div class="name">'+esc(c.name||'Unknown')+'</div>'+
   '<div class="meta">'+esc(c.set||'Unknown set')+' · #'+esc(c.number||'?')+'</div>'+
   '<div class="conf '+(data.exact?'exact':'warn')+'">'+esc(data.mode)+' · '+Math.round((data.confidence||0)*100)+'%</div>';

   if(Array.isArray(data.top_matches)&&data.top_matches.length){
     html+='<div class="matches">';
     for(const m of data.top_matches.slice(0,4)){
       const img=m.image?m.image+'/low.webp':'';
       html+='<div class="match">'+(img?'<img src="'+esc(img)+'">':'')+
       '<div><b>'+esc(m.name)+'</b><small>'+esc(m.set)+' · #'+esc(m.number)+' · score '+Math.round((m.score||0)*100)+'%</small></div></div>';
     }
     html+='</div>';
   }

   html+='<pre>'+esc(JSON.stringify(data,null,2))+'</pre>';
   r.innerHTML=html;
  }else if(data){
   r.innerHTML='<div class="name">No match yet</div><pre>'+esc(JSON.stringify(data,null,2))+'</pre>';
  }else{
   r.innerHTML='<div class="name">Server error</div><div class="meta">HTTP '+res.status+'</div><pre>'+esc(raw.slice(0,10000))+'</pre>';
  }
  r.style.display='block';
 }catch(e){r.innerHTML='<div class="name">Request failed</div><pre>'+esc(String(e))+'</pre>';r.style.display='block'}
 s.style.display='none';b.disabled=false;
};
</script></body></html>""", mimetype="text/html")

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "status": "healthy",
        "engine": "tesseract+tcgdex-hybrid-v4",
        "pytorch": False,
        "easyocr": False,
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

        img = Image.open(tmp_path).convert("RGB")

        # Keep memory + runtime predictable.
        if img.width > 1700:
            ratio = 1700 / img.width
            img = img.resize(
                (1700, max(1, int(img.height*ratio))),
                Image.Resampling.LANCZOS
            )

        result = score_candidates(img)
        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc)
        }), 500
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
