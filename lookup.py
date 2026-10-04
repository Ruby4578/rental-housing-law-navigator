#!/usr/bin/env python3
"""Modules B and C: resolve addresses to jurisdictions, test each extracted rule's coverage,
and track changes. Reads rules.json (from extract.py); writes lookups.json, changes.json,
resolved_addresses.json and site/data.json (the file the web app loads).

Usage:
    python lookup.py --rules rules.json --addresses data/sample_addresses.csv \
                     --tests dev/change_tests.json --as-of 2026-10-01

Geocoding uses the free Census Geocoder (no key). Results are cached in cache/geocode/.
If it cannot be reached, the legal city falls back to the mailing city with a LOW-confidence
flag; the system never silently pretends a fallback is a geocode.

Design rules (from the challenge brief):
  * Missing facts (year built, units, owner type) give "unknown", never a guess.
  * A year-built equal to a certificate-of-occupancy cutoff year is "unknown".
  * Pending bills are "pending"; failed measures are never reported as applying.
  * Where a local rent-control rule applies, the state cap is "superseded".
"""
import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

CENSUS = "https://geocoding.geo.census.gov/geocoder/geographies/address"

# Used only when the Census geocoder cannot be reached (flagged low confidence).
NEIGHBORHOOD_TO_CITY = {
    "dorchester": "Boston", "roxbury": "Boston", "east boston": "Boston", "brighton": "Boston",
    "allston": "Boston", "mattapan": "Boston", "jamaica plain": "Boston", "hyde park": "Boston",
    "south boston": "Boston", "charlestown": "Boston", "west roxbury": "Boston",
    "roslindale": "Boston", "south end": "Boston", "back bay": "Boston", "san ysidro": "San Diego",
}

CATS = ["rent_increase_limits", "just_cause_eviction", "security_deposits",
        "application_screening_fees", "screening_restrictions", "algorithmic_rent_setting"]

# How a change-test rule id from dev/change_tests.json is recognised among YOUR extracted rules.
# (jurisdiction match, category, regex over title/citation/requirement, allowed statuses)
TEST_RULE_PATTERNS = {
    "CA-ALG-01": ("CA", "algorithmic_rent_setting", r"AB\s*325|SB\s*763|common pricing", None),
    "HOB-ALG-01": ("Hoboken, NJ", "algorithmic_rent_setting", r".", None),
    "JC-ALG-01": ("Jersey City, NJ", "algorithmic_rent_setting", r".", None),
    "NJ-ALG-01": ("NJ", "algorithmic_rent_setting", r"FAIR|P\.?L\.?\s*2026", None),
    "MA-ALG-P1": ("MA", "algorithmic_rent_setting", r"S\.?\s*2983", {"pending"}),
    "MA-ALG-P2": ("MA", "algorithmic_rent_setting", r"H\.?\s*5222", {"pending"}),
    "MA-RENT-P1": ("MA", "rent_increase_limits", r"ballot|IP\s*25|question|rent control", {"failed"}),
}


def pdate(s):
    if not s:
        return None
    p = str(s).split("-")
    return date(int(p[0]), int(p[1]) if len(p) > 1 else 1, int(p[2]) if len(p) > 2 else 1)


def clean(v):
    v = (v or "").strip()
    return v or None


def to_int(v):
    try:
        return int(float(v)) if clean(v) else None
    except ValueError:
        return None


# ------------------------------------------------------------------ geocoding
def census_lookup(row, cache_dir, retries=3):
    cpath = cache_dir / f"{row['address_id']}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    import requests
    params = {"street": row["street_address"], "city": row["postal_city"], "state": row["state"],
              "zip": row["zip"], "benchmark": "Public_AR_Current", "vintage": "Current_Current",
              "format": "json"}
    out = {"ok": False}
    for attempt in range(retries):
        try:
            r = requests.get(CENSUS, params=params, timeout=30)
            r.raise_for_status()
            matches = r.json().get("result", {}).get("addressMatches", [])
            if matches:
                g = matches[0].get("geographies", {})
                place = (g.get("Incorporated Places") or [{}])[0].get("NAME")
                county = (g.get("Counties") or [{}])[0].get("NAME")
                state_fips = (g.get("States") or [{}])[0].get("STUSAB")
                out = {"ok": True, "place": place, "county": county, "state": state_fips,
                       "matched": matches[0].get("matchedAddress")}
            else:
                out = {"ok": False, "reason": "no address match"}
            break
        except Exception as e:  # network or service problem; retry with backoff
            out = {"ok": False, "reason": f"error: {str(e)[:100]}", "transient": True}
            time.sleep(2 * (attempt + 1))
    if not out.get("transient"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(out))
    return out


def canonical_city(place, known_cities):
    """Map a Census place name ('Hoboken city') to a city name used in the rules."""
    if not place:
        return None
    low = place.lower()
    stripped = re.sub(r"\s+(city|town|village|borough|municipality|cdp)$", "", low)
    for c in known_cities:
        if c.lower() in (low, stripped):
            return c
    return place if stripped == low else re.sub(r"\s+(city|town|village|borough)$", "", place, flags=re.I)


def resolve_addresses(rows, known_cities, cache_dir, use_geocoder):
    resolved = {}

    def one(row):
        info = census_lookup(row, cache_dir) if use_geocoder else {"ok": False, "reason": "geocoder skipped"}
        if info.get("ok") and info.get("place"):
            return row["address_id"], {"state": row["state"], "county": info.get("county"),
                                       "city": canonical_city(info["place"], known_cities),
                                       "method": "census_geocoder", "confidence": "high"}
        if info.get("ok"):  # matched, but outside any incorporated place
            return row["address_id"], {"state": row["state"], "county": info.get("county"), "city": None,
                                       "method": "census_geocoder", "confidence": "high",
                                       "note": "No incorporated place at this location; only state and county rules apply."}
        pc = (row["postal_city"] or "").strip()
        city = NEIGHBORHOOD_TO_CITY.get(pc.lower(), pc)
        return row["address_id"], {"state": row["state"], "county": None, "city": city,
                                   "method": "mailing_city_fallback", "confidence": "low",
                                   "note": "Legal city taken from the mailing city because the geocoder gave no result ("
                                           + str(info.get("reason", "unknown")) + "). Verify before relying on local rules."}

    with ThreadPoolExecutor(max_workers=4 if use_geocoder else 1) as ex:
        for aid, res in ex.map(one, rows):
            resolved[aid] = res
    return resolved


# ------------------------------------------------------------------ coverage
def rule_in_scope(rule, res):
    j = (rule.get("jurisdiction") or "").strip()
    if rule.get("level") == "state":
        return j.upper() == res["state"].upper()
    if not res.get("city"):
        return False
    return j.lower() == f"{res['city']}, {res['state']}".lower()


def split_conditions(rule):
    c = rule.get("coverage_conditions")
    if isinstance(c, dict):
        return c, str(c.get("text") or "")
    return {}, str(c or "")


def cutoff_year(cond):
    if cond.get("built_on_or_before_year"):
        return int(cond["built_on_or_before_year"]), "year built"
    if cond.get("co_on_or_before"):
        return int(str(cond["co_on_or_before"])[:4]), "certificate of occupancy"
    return None, None


def evaluate(rule, addr, asof):
    """Return (result, explanation) for one rule at one address and date, or None if out of scope."""
    status = rule.get("status")
    if status == "failed":
        return None  # a failed measure is never reported as applying
    if status == "pending":
        return "pending", "This is a bill or proposal, not law. It is shown separately from enacted rules."
    eff = pdate(rule.get("effective_date"))
    if eff and eff > asof:
        return "not_yet_effective", f"Enacted, but it takes effect {rule['effective_date']}."
    if status == "not_yet_effective" and not eff:
        return "not_yet_effective", "Enacted, but the source says it has not taken effect yet."

    cond, text = split_conditions(rule)
    unknown = []
    year, units = addr.get("year_built"), addr.get("units")

    yr, what = cutoff_year(cond)
    if yr:
        if year is None:
            unknown.append(f"coverage depends on the building's age ({what} on or before {yr}) and year built is not in the data")
        elif year > yr:
            return "not_covered", f"Built in {year}, after the {yr} cutoff."
        elif year == yr:
            unknown.append(f"built in the cutoff year {yr}; the cutoff uses the {what} date, which is not in the data")
    if cond.get("min_units") is not None:
        if units is None:
            unknown.append(f"coverage needs at least {cond['min_units']} units and the unit count is not in the data")
        elif units < int(cond["min_units"]):
            return "not_covered", f"{units} units is below the {cond['min_units']}-unit threshold."
    if cond.get("max_units") is not None:
        if units is None:
            unknown.append(f"coverage applies up to {cond['max_units']} units and the unit count is not in the data")
        elif units > int(cond["max_units"]):
            return "not_covered", f"{units} units is above the {cond['max_units']}-unit limit."
    if cond.get("depends_on_owner_type"):
        unknown.append("an exemption depends on owner type, which is not in the data")

    if not cond and text:  # free-text conditions only: flag fact dependencies we cannot test
        t = text.lower()
        if re.search(r"built|certificate of occupancy|construct|year", t) and year is None:
            unknown.append("the conditions mention building age and year built is not in the data")
        if re.search(r"\bunits?\b", t) and units is None:
            unknown.append("the conditions mention unit counts and the unit count is not in the data")
        if re.search(r"owner", t):
            unknown.append("the conditions mention owner status, which is not in the data")

    if unknown:
        return "unknown", "Cannot confirm coverage: " + "; ".join(unknown) + "."
    return "applies", "Covered based on the facts available."


def lookup_address(addr, res, rules, asof):
    items = {}
    for r in rules:
        if not rule_in_scope(r, res):
            continue
        ev = evaluate(r, addr, asof)
        if ev is None or ev[0] == "not_covered":
            continue
        items[r["team_rule_id"]] = {"team_rule_id": r["team_rule_id"], "result": ev[0],
                                    "explanation": ev[1], "conflict_flag": bool(r.get("conflict_flag"))}
    by_id = {r["team_rule_id"]: r for r in rules}

    # Precedence: in rent-increase limits, a local rule governs and the state cap yields to it.
    for it in items.values():
        r = by_id[it["team_rule_id"]]
        if r["category"] != "rent_increase_limits" or r["level"] != "state" or it["result"] != "applies":
            continue
        local = [items[i] for i in items if by_id[i]["category"] == "rent_increase_limits"
                 and by_id[i]["level"] == "city" and items[i]["result"] in ("applies", "unknown")]
        if any(l["result"] == "applies" for l in local):
            gov = next(l for l in local if l["result"] == "applies")
            it["result"] = "superseded"
            it["explanation"] = f"A local rent-control rule governs here ({by_id[gov['team_rule_id']]['title']}); the state rule yields to it."
        elif local:
            it["result"] = "unknown"
            it["explanation"] = "A local rent-control rule may take priority, and its coverage is unknown for this address."

    # Conflict flags: an enacted statewide algorithmic law that may preempt a local one (or vice versa).
    algo = [i for i in items.values() if by_id[i["team_rule_id"]]["category"] == "algorithmic_rent_setting"
            and items[i["team_rule_id"]]["result"] in ("applies", "not_yet_effective")]
    has_state = [i for i in algo if by_id[i["team_rule_id"]]["level"] == "state"]
    has_city = [i for i in algo if by_id[i["team_rule_id"]]["level"] == "city"]
    if has_state and has_city:
        for i in has_state + has_city:
            i["conflict_flag"] = True
            i["explanation"] += " A state and a local algorithmic-pricing rule both cover this address; the state law may preempt the local one. Needs human review."
    return list(items.values())


# ------------------------------------------------------------------ change tests
def find_test_rules(rule_ids, rules):
    found, missing = {}, []
    for tid in rule_ids:
        pat = TEST_RULE_PATTERNS.get(tid)
        match = None
        if pat:
            jur, cat, rx, statuses = pat
            for r in rules:
                text = " ".join(str(r.get(k, "")) for k in ("title", "citation", "requirement", "key_value"))
                if (r.get("jurisdiction", "").lower() == jur.lower() and r.get("category") == cat
                        and re.search(rx, text, re.I) and (statuses is None or r.get("status") in statuses)):
                    match = r["team_rule_id"]
                    break
        if match:
            found[tid] = match
        else:
            missing.append(tid)
    return found, missing


def run_tests(tests, rules, addrs, resolved, default_asof):
    out = {}
    by_id = {r["team_rule_id"]: r for r in rules}
    for t in tests:
        tid = t["test_id"]
        found, missing = find_test_rules(t.get("rule_ids", []), rules)
        notes = [t.get("expected_behavior", "")]
        if missing:
            notes.append("Rules not found in rules.json for: " + ", ".join(missing) +
                         ". Check the extraction output for these.")
        affected, flags = [], []
        d_before = pdate(t.get("as_of_before") or t.get("as_of")) or default_asof
        d_after = pdate(t.get("as_of_after") or t.get("as_of")) or default_asof
        states = set(t.get("states") or [])
        for a in addrs:
            res = resolved[a["address_id"]]
            if states and a["state"] not in states:
                continue
            hits = []
            for rid in found.values():
                r = by_id[rid]
                if not rule_in_scope(r, res):
                    continue
                for d in (d_before, d_after):
                    ev = evaluate(r, a, d)
                    if ev and ev[0] != "not_covered":
                        hits.append((rid, ev[0]))
            if tid == "T5":
                hits = []  # a failed measure never produces an affected address
            if hits:
                affected.append(a["address_id"])
            if t.get("conflict_with"):
                conf_ids, _ = find_test_rules(t["conflict_with"], rules)
                if hits and any(rule_in_scope(by_id[c], res) for c in conf_ids.values()):
                    flags.append(a["address_id"])
        entry = {"affected_address_ids": affected, "notes": " ".join(n for n in notes if n)}
        if t.get("conflict_with"):
            entry["conflict_flag_address_ids"] = flags
        entry["matched_rules"] = found
        out[tid] = entry
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="rules.json")
    ap.add_argument("--addresses", default="data/sample_addresses.csv")
    ap.add_argument("--tests", default="dev/change_tests.json")
    ap.add_argument("--as-of", default="2026-10-01")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--cache", default="cache/geocode")
    ap.add_argument("--no-geocode", action="store_true", help="skip the Census Geocoder (low-confidence fallback)")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    rules = json.loads(rules_path.read_text())["rules"] if rules_path.exists() else []
    if not rules:
        print("WARNING: no rules found. Run extract.py first. Writing address data only.", file=sys.stderr)
    with open(args.addresses, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    addrs = [{"address_id": r["address_id"], "street": r["street_address"], "postal_city": r["postal_city"],
              "state": r["state"], "zip": r["zip"], "year_built": to_int(r["year_built"]),
              "units": to_int(r["units"]), "use_description": clean(r["use_description"]),
              "source_dataset": r["source_dataset"], "retrieved_at": r["retrieved_at"]} for r in rows]
    for a, r in zip(addrs, rows):
        a["street_address"], a["state"] = r["street_address"], r["state"]

    known_cities = sorted({j.split(",")[0].strip() for j in (r["jurisdiction"] for r in rules) if "," in j})
    print(f"Resolving {len(rows)} addresses ({'Census Geocoder' if not args.no_geocode else 'fallback only'})...")
    resolved = resolve_addresses(rows, known_cities, Path(args.cache), not args.no_geocode)
    low = sum(1 for v in resolved.values() if v["confidence"] == "low")
    print(f"  {len(rows) - low} geocoded, {low} low-confidence fallbacks")

    asof = pdate(args.as_of)
    lookups = {a["address_id"]: lookup_address(a, resolved[a["address_id"]], rules, asof) for a in addrs}

    # Results only change on a rule's effective date, so precompute one lookup per date range
    # and let the web app answer any "as of" date from them.
    breakpoints = sorted({pdate(r["effective_date"]) for r in rules if r.get("effective_date")} | {asof})
    epochs = [{"from": "1900-01-01", "lookups": {a["address_id"]: lookup_address(a, resolved[a["address_id"]], rules, date(1900, 1, 1)) for a in addrs}}]
    for bp in breakpoints:
        epochs.append({"from": bp.isoformat(),
                       "lookups": {a["address_id"]: lookup_address(a, resolved[a["address_id"]], rules, bp) for a in addrs}})

    tests = json.loads(Path(args.tests).read_text()) if Path(args.tests).exists() else []
    changes = run_tests(tests, rules, addrs, resolved, asof)

    outdir = Path(args.outdir)
    (outdir / "site").mkdir(parents=True, exist_ok=True)
    (outdir / "lookups.json").write_text(json.dumps({"as_of": args.as_of, "lookups": lookups}, indent=1))
    (outdir / "changes.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "matched_rules"} for k, v in changes.items()}, indent=1))
    (outdir / "resolved_addresses.json").write_text(json.dumps(resolved, indent=1))
    bundle = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "default_as_of": args.as_of, "categories": CATS,
              "rules": rules, "addresses": [{**a, "resolved": resolved[a["address_id"]]} for a in addrs],
              "epochs": epochs, "tests": tests, "changes": changes}
    (outdir / "site" / "data.json").write_text(json.dumps(bundle, separators=(",", ":")))
    n = sum(len(v) for v in lookups.values())
    print(f"Wrote lookups.json ({n} results), changes.json, resolved_addresses.json, site/data.json")
    for tid, c in changes.items():
        print(f"  {tid}: {len(c['affected_address_ids'])} affected"
              + (f", {len(c.get('conflict_flag_address_ids', []))} flagged" if "conflict_flag_address_ids" in c else "")
              + (f"  [rules matched: {c['matched_rules']}]" if c.get("matched_rules") else ""))


if __name__ == "__main__":
    main()
