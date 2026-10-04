#!/usr/bin/env python3
"""finalize.py - build rules.json from whatever extract.py has already cached (no API calls).
Run:  python finalize.py            (safe to rerun any time; rerun after more extraction)
"""
import glob, json, re
from difflib import SequenceMatcher
from pathlib import Path
from jsonschema import Draft202012Validator
import extract as ex

validator = Draft202012Validator(json.loads(ex.SCHEMA_PATH.read_text()))
latest = {}  # (doc, chunk) -> newest cache file
for f in glob.glob(str(ex.CACHE_DIR / "*.json")):
    m = re.match(r"(.+)\.(\d+)\.[0-9a-f]{12}\.json$", Path(f).name)
    if not m:
        continue
    k = (m.group(1), int(m.group(2)))
    if k not in latest or Path(f).stat().st_mtime > Path(latest[k]).stat().st_mtime:
        latest[k] = f


CURLY = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "}

def norm_map(text):
    """Normalised text plus, for each normalised char, its index in the original."""
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(text):
        ch = CURLY.get(ch, ch)
        if ch.isspace():
            if not prev_space:
                out.append(" "); idx.append(i); prev_space = True
        else:
            out.append(ch); idx.append(i); prev_space = False
    return "".join(out), idx

_cache = {}
def repair_quote(span, text, doc):
    """Longest passage of the model's quote that appears verbatim in the source (original text)."""
    if doc not in _cache:
        _cache[doc] = norm_map(text)
    dn, idx = _cache[doc]
    qn = ex.norm(span)
    if len(qn) < 20:
        return None
    m = SequenceMatcher(None, dn, qn, autojunk=False).find_longest_match(0, len(dn), 0, len(qn))
    if m.size < 40:
        return None
    start, end = idx[m.a], idx[m.a + m.size - 1] + 1
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()

def fill_required(rec, doc):
    d = {"title": (rec.get("category") or "rule").replace("_", " ").title(),
         "requirement": rec.get("title") or "See quoted source text.",
         "citation": "Not stated in source (" + doc + ")",
         "jurisdiction": "Unknown"}
    for k, v in d.items():
        if not isinstance(rec.get(k), str) or not rec.get(k).strip():
            rec[k] = v
    if rec.get("conflict_flag") is None:
        rec["conflict_flag"] = False

kept, rejected, n = [], [], 0
docs = sorted({d for d, _ in latest})
for doc in docs:
    path = Path("corpus/text") / f"{doc}.txt"
    if not path.exists():
        continue
    text, url = ex.read_doc(path)
    for (d, i) in sorted(k for k in latest if k[0] == doc):
        raw = json.loads(Path(latest[(d, i)]).read_text())["raw"]
        for rec in ex.parse_rules(raw):
            if not isinstance(rec, dict):
                continue
            n += 1
            rec["team_rule_id"] = f"r-{n:04d}"
            rec["source_doc_id"], rec["source_url"] = doc, url
            rec.setdefault("overrides", [])
            ex.fix_status(rec)
            fill_required(rec, doc)
            if not ex.quote_ok(rec.get("quoted_span", ""), text):
                fixed = repair_quote(rec.get("quoted_span", ""), text, doc)
                if not fixed:
                    rejected.append({"doc": doc, "reason": "quote not verbatim", "record": rec}); continue
                rec["quoted_span"] = fixed
            errs = [e.message for e in validator.iter_errors(rec)]
            if errs:
                rejected.append({"doc": doc, "reason": f"schema: {errs[:3]}", "record": rec}); continue
            kept.append(rec)

Path("rules.json").write_text(json.dumps({"rules": kept}, indent=2, ensure_ascii=False))
Path("rejected.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False))
print(f"{len(docs)} cached docs -> {len(kept)} rules in rules.json, {len(rejected)} rejected (rejected.json)")
print("Last doc processed:", docs[-1] if docs else None)
