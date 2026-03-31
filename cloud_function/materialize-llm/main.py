import csv
import io
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME = os.getenv("GCS_BUCKET")                         # REQUIRED
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")   # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")       # 20251026170002

# Stable CSV schema for students
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "color", "transmission", "condition",
    "vehicle_age", "miles_per_year", "price_per_10k_miles",
    "source_txt"
]


def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    """Lists all run_id= folders in the bucket prefix."""
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)


def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/jsonl_llm/."""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"

    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        try:
            line = blob.download_as_text().strip()
            if not line:
                continue
            rec = json.loads(line)
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue


def _run_id_to_dt(rid: str) -> datetime:
    """Parses run_id string into a UTC datetime object."""
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    return blob.open("w")


def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def _derive_fields(rec: dict) -> dict:
    rec = dict(rec)

    current_year = datetime.now(timezone.utc).year
    price = _safe_int(rec.get("price"))
    year = _safe_int(rec.get("year"))
    mileage = _safe_int(rec.get("mileage"))

    if year is not None and 1900 <= year <= current_year:
        rec["vehicle_age"] = current_year - year
    else:
        rec["vehicle_age"] = None

    if mileage is not None and rec["vehicle_age"] not in (None, 0):
        rec["miles_per_year"] = round(mileage / rec["vehicle_age"])
    else:
        rec["miles_per_year"] = None

    if price is not None and mileage not in (None, 0):
        rec["price_per_10k_miles"] = round(price / (mileage / 10000), 2)
    else:
        rec["price_per_10k_miles"] = None

    return rec


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    """Writes the provided records to a CSV in GCS."""
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            rec = _derive_fields(rec)
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n


def _get_existing_master_data(bucket_name: str, key: str) -> Dict[str, Dict]:
    """Downloads existing master CSV and returns a dict keyed by post_id."""
    b = storage_client.bucket(bucket_name)
    blob = b.blob(key)
    data = {}

    if not blob.exists():
        return data

    try:
        content = blob.download_as_text()
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            pid = row.get("post_id")
            if pid:
                data[pid] = row
    except Exception:
        pass

    return data


def materialize_http(request: Request):
    """
    HTTP POST:
    1. Filter runs for only the last ~75 minutes.
    2. Load existing master CSV.
    3. Merge recent run data into master by post_id.
    4. Overwrite the master CSV in GCS.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        all_run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        limit_time = datetime.now(timezone.utc) - timedelta(minutes=75)
        recent_runs = [r for r in all_run_ids if _run_id_to_dt(r) > limit_time]

        if not recent_runs:
            return jsonify({"ok": True, "message": "No new runs found in the last hour"}), 200

        final_key = f"{STRUCTURED_PREFIX}/datasets/listings_master_llm.csv"
        master_records = _get_existing_master_data(BUCKET_NAME, final_key)

        for rid in recent_runs:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue

                prev = master_records.get(pid)
                if (prev is None) or (_run_id_to_dt(rid) >= _run_id_to_dt(prev.get("run_id", ""))):
                    master_records[pid] = rec

        rows_written = _write_csv(master_records.values(), final_key)

        return jsonify({
            "ok": True,
            "recent_runs_scanned": len(recent_runs),
            "total_listings_in_master": rows_written,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
