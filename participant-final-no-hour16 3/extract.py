#!/usr/bin/env python3
"""Module A: automated rule extraction from the corpus using the Gemini API.

Usage:
    export GEMINI_API_KEY=...            # never hard-code or commit the key
    python extract.py --corpus ./corpus --schema ./schema/rule_record.schema.json \
                      --out rules.json [--model gemini-2.5-flash] [--only D024,D025]

Design:
  * One LLM call per document chunk; raw model output is cached on disk
    (cache/<key>.json) so reruns are free and the cache doubles as the audit log.
  * Every record is checked: JSON-schema valid, and its quoted_span must appear
    VERBATIM (whitespace-normalised) in the source text. Records that fail the
    quote check are dropped and logged, never silently kept (no invented citations).
  * Status is reconciled against the as-of date so enacted-but-future laws are
    'not_yet_effective'.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

AS_OF_DEFAULT = "2026-10-01"
PROMPT_VERSION = "v1"
CHUNK_CHARS = 120_000   # split very long docs; Gemini handles long context but outputs are capped
CHUNK_OVERLAP = 3_000

SYSTEM_PROMPT = """You extract structured rule records from US housing law text.

Return ONLY a JSON array (possibly empty). One object per distinct rule. Rules:
- Use ONLY what the provided text says. Never add rules or citations from memory.
- If the document contains no tenant/landlord rule in the six categories, return [].
- Categories: rent_increase_limits, just_cause_eviction, security_deposits,
  application_screening_fees, screening_restrictions, algorithmic_rent_setting.
- level is "state" or "city". jurisdiction is a state code ("CA","NJ","MA") or "City, ST".
- status (as of {as_of}): in_force | not_yet_effective (enacted, effective date after
  {as_of}) | pending (bill not enacted) | failed (struck/defeated/withdrawn).
- effective_date: ISO 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'; null if the text gives none.
  If the text gives two different effective dates, use the one in the official
  ordinance/statute text and describe the discrepancy in conflict_note, with conflict_flag true.
- coverage_conditions: an OBJECT with these keys (null when the text is silent):
    text (plain summary), co_on_or_before (YYYY-MM-DD certificate-of-occupancy cutoff),
    built_on_or_before_year (int), min_units (int), max_units (int),
    depends_on_owner_type (bool), applies_citywide (bool), other (string).
  Do not guess. Missing facts stay null.
- exemptions: plain text, null if none stated.
- overrides: leave [] (the system links rules afterwards); put local-vs-state precedence in
  'interaction' (e.g. "State cap yields where local rent control applies"), else null.
- quoted_span: a sentence or clause copied EXACTLY, character for character, from the text
  (at least 20 characters). Do not paraphrase, fix typos, or join separate passages.
- citation: official cite as written in the text (e.g. "Cal. Civ. Code § 1947.12").
- confidence 0-1. Lower it when the text is a summary page rather than the statute itself.
- Plain-language 'requirement' of 1-2 sentences that a renter could act on. Never advise how
  to avoid or structure around a rule.
Fields: team_rule_id (put "tmp"), jurisdiction, level, category, status, title, requirement,
key_value, coverage_conditions, exemptions, overrides, interaction, effective_date, citation,
quoted_span, confidence, conflict_flag, conflict_note.
"""


def norm(s: str) -> str:
    """Whitespace/quote-normalise for verbatim matching."""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_source(quote: str, source_norm: str) -> bool:
    q = norm(quote)
    return len(q) >= 20 and q in source_norm


def chunk_text(text: str):
    if len(text) <= CHUNK_CHARS:
        yield text
        return
    i = 0
    while i < len(text):
        yield text[i:i + CHUNK_CHARS]
        i += CHUNK_CHARS - CHUNK_OVERLAP


def strip_header(text: str) -> str:
    return text  # corpus files carry a URL/retrieval header; harmless to leave in


def parse_json_array(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("rules", [data])
    if not isinstance(data, list):
        raise ValueError("model did not return a JSON array")
    return data


class Gemini:
    def __init__(self, model: str):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY in your environment.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def __call__(self, system: str, user: str) -> str:
        from google.genai import types
        delay = 5
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                return resp.text
            except Exception as e:  # rate limits / transient errors
                print(f"  retry {attempt+1} after error: {str(e)[:120]}", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("Gemini call failed after retries")


def reconcile_status(rec: dict, as_of: str) -> dict:
    eff = rec.get("effective_date")
    if eff and rec.get("status") == "in_force":
        padded = eff + "-01-01"[len(eff) - 4:] if len(eff) < 10 else eff
        if padded > as_of:
            rec["status"] = "not_yet_effective"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--schema", default="schema/rule_record.schema.json")
    ap.add_argument("--out", default="rules.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--as-of", default=AS_OF_DEFAULT)
    ap.add_argument("--only", default="", help="comma-separated doc_ids")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--log", default="audit_log.jsonl")
    args = ap.parse_args()

    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(json.load(open(args.schema)))
    corpus = Path(args.corpus)
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    llm = Gemini(args.model)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    with open(corpus / "corpus_manifest.csv", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    rules, n = [], 0
    log = open(args.log, "a", encoding="utf-8")
    for row in manifest:
        doc_id = row["doc_id"]
        if row["status"] != "ok" or not row["text_file"]:
            continue
        if only and doc_id not in only:
            continue
        text = (corpus / row["text_file"]).read_text(encoding="utf-8", errors="replace")
        src_norm = norm(text)
        system = SYSTEM_PROMPT.format(as_of=args.as_of)
        print(f"{doc_id} {row['jurisdictions']} ({len(text):,} chars)")
        for ci, chunk in enumerate(chunk_text(strip_header(text))):
            key = hashlib.sha256(
                f"{PROMPT_VERSION}|{args.model}|{args.as_of}|{row['sha256']}|{ci}".encode()
            ).hexdigest()[:24]
            cpath = cache / f"{doc_id}_{ci}_{key}.json"
            if cpath.exists():
                raw = json.loads(cpath.read_text())["raw"]
            else:
                user = (f"Document {doc_id} | jurisdiction hint: {row['jurisdictions']} | "
                        f"URL: {row['url']} | retrieved: {row['retrieved_at']}\n\n"
                        f"=== TEXT START ===\n{chunk}\n=== TEXT END ===")
                raw = llm(system, user)
                cpath.write_text(json.dumps({"doc_id": doc_id, "model": args.model, "raw": raw}))
            try:
                recs = parse_json_array(raw)
            except Exception as e:
                log.write(json.dumps({"doc": doc_id, "event": "parse_error", "err": str(e)}) + "\n")
                continue
            for rec in recs:
                n += 1
                rec["team_rule_id"] = f"r-{n:04d}"
                rec["source_doc_id"] = doc_id
                rec["source_url"] = row["url"]
                rec["retrieved_at"] = row["retrieved_at"]
                rec.setdefault("overrides", [])
                rec.setdefault("conflict_flag", False)
                reconcile_status(rec, args.as_of)
                errs = [e.message for e in validator.iter_errors(rec)]
                quote_ok = quote_in_source(rec.get("quoted_span", ""), src_norm)
                if errs or not quote_ok:
                    log.write(json.dumps({"doc": doc_id, "event": "dropped",
                                          "rule": rec.get("title"), "schema_errors": errs[:3],
                                          "quote_found": quote_ok}) + "\n")
                    continue
                log.write(json.dumps({"doc": doc_id, "event": "kept", "id": rec["team_rule_id"],
                                      "title": rec.get("title")}) + "\n")
                rules.append(rec)

    Path(args.out).write_text(json.dumps({"rules": rules}, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rules)} verified rules to {args.out} (see {args.log} for dropped records)")


if __name__ == "__main__":
    main()
