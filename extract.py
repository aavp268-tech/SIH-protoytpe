"""Requirement extraction: product, application, technical parameters, safety cues (review sec. 5)."""
import re
PARAMS = {"grade": r"\b(?:fe\s?\d{3}[a-z]?|\d{2}\s?grade|grade\s?\d{2,3})\b",
          "voltage": r"\b\d{2,4}(?:/\d{2,4})?\s?(?:kv|v)\b", "current": r"\b\d{1,3}\s?a\b",
          "size": r"\b\d{1,4}(?:\.\d+)?\s?(?:mm|cm)\b", "pressure": r"\b(?:pn\s?\d+|\d+(?:\.\d+)?\s?(?:mpa|bar))\b"}
APPS = ["building construction", "water supply", "drinking water", "house wiring", "marine", "precast",
        "masonry", "reinforcement", "structural", "office", "foundation"]
SAFETY = ["safety", "fire", "flame", "shock", "insulation", "earthing", "protection"]

def extract_requirements(text, vocab):
    t = text.lower()
    toks = set(re.findall(r"[a-z0-9]+", t))
    return {"products": sorted(toks & vocab),
            "applications": [a for a in APPS if a in t],
            "parameters": {k: sorted(set(m.replace(" ", "") for m in re.findall(p, t))) for k, p in PARAMS.items()
                           if re.findall(p, t)},
            "safety": [s for s in SAFETY if s in toks]}

def pdf_to_text(f):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
