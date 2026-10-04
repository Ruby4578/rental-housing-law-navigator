#!/usr/bin/env python3
"""extract.py - corpus text files -> rules.json (Groq, verbatim-quote checked).

Usage:
    python extract.py                      # whole corpus/text/
    python extract.py corpus/text/D001.txt # specific files (good for a first test)
    python extract.py --out rules_test.json corpus/text/D001.txt corpus/text/D003.txt

Setup:
    pip install requests python-dotenv jsonschema
    .env  ->  GROQ_API_KEY=...   (and optionally GROQ_MODEL=...)
    .gitignore must contain  .env  and  cache/

Outputs:
    rules.json       validated rules whose quoted_span was found verbatim in the source
    rejected.json    records dropped (bad quote / schema failure) with the reason
    cache/*.json     raw model output per chunk (audit log + saves quota on reruns)
"""
import argparse, glob, hashlib, json, os, re, sys, time
from pathlib import Path

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

load_dotenv()

ROOT = Path(__file__).parent
SCHEMA_PATH = next((p for p in (ROOT / "rule_record_schema.json", ROOT / "rule_record.schema.json") if p.exists()), ROOT / "rule_record_schema.json")
CORPUS_GLOB = str(ROOT / "corpus" / "text" / "*.txt")
CACHE_DIR = ROOT / "cache"
AS_OF = "2026-10-01"
CHUNK_CHARS = 14000          # ~3.5k tokens; keeps each call well under free-tier TPM
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # confirm current name in Groq console

CATEGORIES = [
    "rent_increase_limits", "just_cause_eviction", "security_deposits",
    "application_screening_fees", "screening_restrictions", "algorithmic_rent_setting",
]

SYSTEM = f"""You extract residential rental-housing rules from legal source text.
Return ONLY a JSON object: {{"rules": [ ... ]}}. If the text contains no rule in the allowed
categories, return {{"rules": []}}. Never invent rules, dates or numbers that are not in the text.

Allowed categories: {", ".join(CATEGORIES)}.
(Relocation payments, registration fees, news items and procedure are NOT categories - skip them
unless they directly set one of the allowed categories.)

Each rule object has these keys:
  jurisdiction      "CA"/"NJ"/"MA" for state law, or "City, ST" e.g. "Berkeley, CA"
  level             "state" or "city"
  category          one of the allowed categories
  status            "in_force" | "not_yet_effective" | "pending" | "failed", as of {AS_OF}.
                    Enacted but future effective date = not_yet_effective. Bills not yet signed = pending.
                    Failed/struck ballot measures = failed.
  title             short name of the rule
  requirement       1-2 plain-language sentences
  key_value         headline number/formula or null
  coverage_conditions  who is covered (year built, unit counts, owner type) or null
  exemptions        text or null
  effective_date    YYYY, YYYY-MM or YYYY-MM-DD, or null if the text does not state or clearly imply it
  citation          official cite, e.g. "Berkeley Municipal Code 13.63.030"
  quoted_span       text COPIED EXACTLY, character for character, from the source (one contiguous
                    passage, 20-300 chars, no ellipses, no edits) that supports the rule
  confidence        0..1
  conflict_flag     true if the text or another law creates an unresolved conflict/ambiguity
  conflict_note     explain the conflict (e.g. two possible effective dates), else null
  interaction       how this rule interacts with state/local law (supersedes/yields), or null

Rules for dates: do not guess. If an ordinance was passed to print but the adoption/effective date is
not stated, set effective_date null, set conflict_flag true and say why in conflict_note.
"""


def read_doc(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.match(r"\s*SOURCE:\s*(\S+)", text)
    return text, (m.group(1) if m else "")


def chunk(text, size=CHUNK_CHARS):
    """Split on line boundaries so no sentence is cut mid-line."""
    if len(text) <= size:
        return [text]
    chunks, cur, n = [], [], 0
    for line in text.splitlines(keepends=True):
        if n + len(line) > size and cur:
            chunks.append("".join(cur)); cur, n = [], 0
        cur.append(line); n += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def call_llm(system, user):
    """The ONLY provider-specific function. Swap this body for Gemini/OpenAI etc."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY missing - put it in .env (never in code or chat).")
    body = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    for attempt in range(6):
        r = requests.post(GROQ_URL, json=body, timeout=120,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after", 0)) or min(2 ** attempt * 3, 60)
            print(f"  {r.status_code}, waiting {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("Groq: too many retries")


def cached_call(doc_id, idx, user):
    CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.sha256((MODEL + SYSTEM + user).encode()).hexdigest()[:12]
    f = CACHE_DIR / f"{doc_id}.{idx}.{h}.json"
    if f.exists():
        return json.loads(f.read_text())["raw"]
    raw = call_llm(SYSTEM, user)
    f.write_text(json.dumps({"model": MODEL, "doc": doc_id, "chunk": idx, "raw": raw}, indent=1))
    return raw


def parse_rules(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    rules = data.get("rules", data) if isinstance(data, dict) else data
    return rules if isinstance(rules, list) else []


def norm(s):
    """Whitespace + typographic-quote normalisation ONLY (no other fuzziness)."""
    s = (s.replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " "))
    return re.sub(r"\s+", " ", s).strip()


def quote_ok(span, doc_text):
    return len(norm(span)) >= 20 and norm(span) in norm(doc_text)


def fix_status(rec):
    """Enacted + future effective date can't be in_force at AS_OF."""
    ed = rec.get("effective_date")
    if rec.get("status") == "in_force" and ed and ed > AS_OF[:len(ed)] and len(ed) == 10:
        rec["status"] = "not_yet_effective"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default="rules.json")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(CORPUS_GLOB))
    if not files:
        sys.exit(f"No files found ({CORPUS_GLOB})")
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))

    kept, rejected, n = [], [], 0
    for path in files:
        doc_id = Path(path).stem
        text, url = read_doc(path)
        print(f"{doc_id}: {len(text)} chars")
        for i, ch in enumerate(chunk(text)):
            user = f"Source document {doc_id} (part {i + 1}):\n\n{ch}"
            try:
                recs = parse_rules(cached_call(doc_id, i, user))
            except Exception as e:  # keep going; one bad doc must not kill the run
                rejected.append({"doc": doc_id, "reason": f"llm error: {e}"})
                continue
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                n += 1
                rec["team_rule_id"] = f"r-{n:04d}"
                rec["source_doc_id"] = doc_id
                rec["source_url"] = url
                rec.setdefault("overrides", [])
                fix_status(rec)
                if not quote_ok(rec.get("quoted_span", ""), text):
                    rejected.append({"doc": doc_id, "reason": "quote not verbatim in source", "record": rec})
                    continue
                errs = [e.message for e in validator.iter_errors(rec)]
                if errs:
                    rejected.append({"doc": doc_id, "reason": f"schema: {errs[:3]}", "record": rec})
                    continue
                kept.append(rec)
        print(f"  -> {sum(1 for k in kept if k['source_doc_id'] == doc_id)} kept")

    Path(args.out).write_text(json.dumps({"rules": kept}, indent=2, ensure_ascii=False))
    Path("rejected.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False))
    print(f"\n{len(kept)} rules -> {args.out}; {len(rejected)} rejected -> rejected.json")


if __name__ == "__main__":
    main()
