# main.py
# Purpose: Convert raw TXT -> one-line JSON records (.jsonl) in GCS.
# Compatible input layouts:
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/*.txt
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/txt/*.txt
# where <RUN> is either 20251026T170002Z or 20251026170002.
# Output:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<RUN>/jsonl/<post_id>.jsonl

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
BUCKET_NAME        = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX     = os.getenv("SCRAPES_PREFIX", "scrapes")         # input
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")   # output

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

storage_client = storage.Client()

# -------------------- SIMPLE REGEX EXTRACTORS --------------------
PRICE_RE      = re.compile(r"\$\s?([0-9,]+)")
YEAR_RE       = re.compile(r"\b(19|20)\d{2}\b")
MAKE_MODEL_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][A-Za-z0-9]+)")

VALID_MAKES = {
    "Acura", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chevy",
    "Chrysler", "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti",
    "Jaguar", "Jeep", "Kia", "Lexus", "Lincoln", "Mazda", "Mercedes",
    "Mercury", "Mini", "Mitsubishi", "Nissan", "Pontiac", "Ram",
    "Subaru", "Toyota", "Volkswagen", "Volvo"
}

BANNED_TOKENS = {
    "Contact", "Information", "New", "North", "South", "East", "West",
    "Buy", "Here", "River", "Haven", "Hartford", "Glastonbury",
    "Saybrook", "Milford", "Orange", "Deep", "Sport", "SUV",
    "Sedan", "Coupe", "Super", "Duty"
}

# -------------------- HELPERS --------------------
def _list_run_ids(bucket: str, scrapes_prefix: str) -> list[str]:
    """
    List run folders under gs://bucket/<scrapes_prefix>/ and return normalized run_ids.
    Accept:
      - <scrapes_prefix>/run_id=20251026T170002Z/
      - <scrapes_prefix>/20251026170002/
    """
    it = storage_client.list_blobs(bucket, prefix=f"{scrapes_prefix}/", delimiter="/")
    for _ in it:
        pass  # populate it.prefixes

    run_ids: list[str] = []
    for pref in getattr(it, "prefixes", []):
        # e.g., 'scrapes/run_id=20251026T170002Z/' OR 'scrapes/20251026170002/'
        tail = pref.rstrip("/").split("/")[-1]
        cand = tail.split("run_id=", 1)[1] if tail.startswith("run_id=") else tail
        if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
            run_ids.append(cand)
    return sorted(run_ids)

def _txt_objects_for_run(run_id: str) -> list[str]:
    """
    Return .txt object names for a given run_id.
    Tries (in order) and returns the first non-empty list:
      scrapes/run_id=<run_id>/txt/
      scrapes/run_id=<run_id>/
      scrapes/<run_id>/txt/
      scrapes/<run_id>/
    """
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/txt/",
        f"{SCRAPES_PREFIX}/run_id={run_id}/",
        f"{SCRAPES_PREFIX}/{run_id}/txt/",
        f"{SCRAPES_PREFIX}/{run_id}/",
    ]
    for pref in candidates:
        names = [b.name for b in bucket.list_blobs(prefix=pref) if b.name.endswith(".txt")]
        if names:
            return names
    return []

def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)

def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")

def _parse_run_id_as_iso(run_id: str) -> str:
    """Normalize either run_id style to ISO8601 Z (fallback = now UTC)."""
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# -------------------- PARSE A LISTING --------------------
def parse_listing(text: str) -> dict:
    d = {}

    m = PRICE_RE.search(text)
    if m:
        try:
            d["price"] = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    y = YEAR_RE.search(text)
    if y:
        try:
            d["year"] = int(y.group(0))
        except ValueError:
            pass

    # try to parse make/model from the first non-empty line, which is usually the title
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title_line = lines[0] if lines else ""

    found_make = None
    for make in sorted(VALID_MAKES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(make)}\b", title_line, re.I):
            found_make = make
            break

    if found_make:
        d["make"] = found_make

        model_part = re.split(rf"\b{re.escape(found_make)}\b", title_line, maxsplit=1, flags=re.I)
        if len(model_part) > 1:
            model_tokens = model_part[1].strip().split()
            if model_tokens:
                first_model_token = re.sub(r"[^A-Za-z0-9-]", "", model_tokens[0])
                if first_model_token and first_model_token not in BANNED_TOKENS:
                    d["model"] = first_model_token
    else:
        # fallback to the old regex-based parser if no valid make is found in the title
        mm = MAKE_MODEL_RE.search(text)
        if mm:
            make_candidate = mm.group(1)
            model_candidate = mm.group(2)

            if (
                make_candidate in VALID_MAKES
                and make_candidate not in BANNED_TOKENS
                and model_candidate not in BANNED_TOKENS
            ):
                d["make"] = make_candidate
                d["model"] = model_candidate

    # mileage variants
    mi = None
    m1 = re.search(r"(?:mileage|odometer)\s*[:\-]?\s*([\d,]+)", text, re.I)
    if m1:
        try:
            mi = int(m1.group(1).replace(",", ""))
        except ValueError:
            mi = None

    if mi is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*k\s*(?:mi|mile|miles)\b", text, re.I)
        if m2:
            try:
                mi = int(float(m2.group(1)) * 1000)
            except ValueError:
                mi = None

    if mi is None:
        m3 = re.search(r"(\d{1,3}(?:[,\d]{3})*)\s*(?:mi|mile|miles)\b", text, re.I)
        if m3:
            try:
                mi = int(re.sub(r"[^\d]", "", m3.group(1)))
            except ValueError:
                mi = None

    if mi is not None:
        d["mileage"] = mi

    # clean title flag
    if re.search(r"\bclean title\b|\btitle in hand\b", text, re.I):
        d["clean_title_flag"] = 1
    else:
        d["clean_title_flag"] = 0

    # vehicle age
    if "year" in d:
        d["vehicle_age"] = 2026 - d["year"]

    # miles per year
    if "mileage" in d and "vehicle_age" in d and d["vehicle_age"] > 0:
        d["miles_per_year"] = int(d["mileage"] / d["vehicle_age"])

    # price per 10k miles
    if "price" in d and "mileage" in d and d["mileage"] > 0:
        d["price_per_10k_miles"] = round(d["price"] / (d["mileage"] / 10000), 2)

    return d

# -------------------- HTTP ENTRY --------------------
def extract_http(request: Request):
    """
    Reads latest (or requested) run's TXT listings and writes ONE-LINE JSON records to:
      gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<run_id>/jsonl/<post_id>.jsonl
    Request JSON (optional):
      { "run_id": "<...>", "max_files": 0, "overwrite": false }
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or 0)        # 0 = unlimited
    overwrite = bool(body.get("overwrite") or False)

    # Pick newest run if not provided
    if not run_id:
        runs = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {SCRAPES_PREFIX}/"}), 200
        run_id = runs[-1]

    scraped_at_iso = _parse_run_id_as_iso(run_id)

    txt_blobs = _txt_objects_for_run(run_id)
    if not txt_blobs:
        return jsonify({"ok": False, "run_id": run_id, "error": "no .txt files found for run"}), 200
    if max_files > 0:
        txt_blobs = txt_blobs[:max_files]

    processed = written = skipped = errors = 0
    bucket = storage_client.bucket(BUCKET_NAME)

    for name in txt_blobs:
        try:
            text = _download_text(name)
            fields = parse_listing(text)

            post_id = os.path.splitext(os.path.basename(name))[0]
            record = {
                "post_id": post_id,
                "run_id": run_id,
                "scraped_at": scraped_at_iso,
                "source_txt": name,
                **fields,
            }

            out_key = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/{post_id}.jsonl"

            if not overwrite and bucket.blob(out_key).exists():
                skipped += 1
            else:
                _upload_jsonl_line(out_key, record)
                written += 1

        except Exception as e:
            errors += 1
            logging.error(f"Failed {name}: {e}\n{traceback.format_exc()}")

        processed += 1

    result = {
        "ok": True,
        "version": "extractor-v3-jsonl-flex",
        "run_id": run_id,
        "processed_txt": processed,
        "written_jsonl": written,
        "skipped_existing": skipped,
        "errors": errors
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
