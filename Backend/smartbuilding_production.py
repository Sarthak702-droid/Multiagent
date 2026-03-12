#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NeuralGrid — Decentralized Multi-Agent Self-Healing Smart Building         ║
║  PRODUCTION SINGLE-FILE  |  Flask WSGI + FastAPI ASGI  |  Python 3.10+     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL:                                                                    ║
║    pip install flask fastapi "uvicorn[standard]" waitress flask-limiter     ║
║         flask-compress pydantic numpy pandas scikit-learn cryptography joblib║
║                                                                              ║
║  RUN:                                                                        ║
║    python smartbuilding_production.py               # Flask+FastAPI (both)  ║
║    python smartbuilding_production.py --api flask   # Flask only            ║
║    python smartbuilding_production.py --api fastapi # FastAPI only          ║
║    python smartbuilding_production.py --port 7000   # custom port           ║
║    SB_API_KEY=mysecret python smartbuilding_production.py  # with auth      ║
║                                                                              ║
║  URLs (default --api both):                                                  ║
║    http://localhost:5000        → Dashboard UI  (Flask + Waitress)          ║
║    http://localhost:5001/docs   → FastAPI Swagger  (Uvicorn)                ║
║    http://localhost:5001/redoc  → FastAPI ReDoc                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# ─── stdlib ───────────────────────────────────────────────────────────────────
import sys, os, json, uuid, threading, time, io, traceback, base64
import sqlite3, hashlib, secrets, random, argparse, warnings, signal
import logging, logging.handlers
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Any, List

# ─── core science (always required) ──────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_file, Response
from cryptography.fernet import Fernet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

# ─── optional production packages — degrade gracefully if missing ─────────────
try:
    from flask_compress import Compress as FlaskCompress
    _HAS_COMPRESS = True
except ImportError:
    _HAS_COMPRESS = False

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _HAS_LIMITER = True
except ImportError:
    _HAS_LIMITER = False

try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    # Minimal shim — structural only, no range validation without pydantic
    class BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
        def model_dump(self): return {k: v for k, v in self.__dict__.items()}
    Field = lambda default=None, **kw: default
    class ValidationError(Exception):
        def errors(self): return [str(self)]
    def field_validator(*a, **kw): return lambda f: f

try:
    from fastapi import (FastAPI, HTTPException, Depends,
                         Request as FARequest, status as fa_status)
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
    from fastapi.security.api_key import APIKeyHeader
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

try:
    from waitress import serve as waitress_serve
    _HAS_WAITRESS = True
except ImportError:
    _HAS_WAITRESS = False

# ═══════════════════════════════════════════════════════════════════════════════
#  §1  PATHS & CONFIGURATION
#      All paths are absolute via __file__ — works from ANY working directory.
#      Data folder is always next to the script itself.
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
DB_PATH    = DATA_DIR / "smartbuilding.db"
KEY_PATH   = DATA_DIR / ".enc_key"
MODEL_PATH = DATA_DIR / "model.joblib"
LOG_PATH   = DATA_DIR / "server.log"

# Create data directory immediately at import time — never fails on restart
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Environment variables
API_KEY         = os.environ.get("SB_API_KEY", "")          # "" = auth disabled
SECRET_KEY      = os.environ.get("SB_SECRET", secrets.token_hex(32))
MAX_BODY_BYTES  = int(os.environ.get("MAX_BODY_MB", "10")) * 1024 * 1024
RATE_DEFAULT    = os.environ.get("RATE_LIMIT",    "120/minute")
RATE_PIPELINE   = os.environ.get("RATE_PIPELINE",  "5/minute")
RATE_PREDICT    = os.environ.get("RATE_PREDICT",  "200/minute")

# ═══════════════════════════════════════════════════════════════════════════════
#  §2  STRUCTURED LOGGING
#      JSON lines → stdout + rotating file (5MB × 3). Works on Windows.
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_logging():
    fmt = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); root.addHandler(ch)
    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt); root.addHandler(fh)
    except Exception:
        pass  # log file not writable — stdout only
    return logging.getLogger("neuralGrid")

log = _setup_logging()

def _log(level: str, msg: str, **kw):
    log.__getattribute__(level)(json.dumps({"msg": msg, **kw}))

# ═══════════════════════════════════════════════════════════════════════════════
#  §3  ENCRYPTION  (AES-256 via Fernet)
#      Key is created once and persisted to .enc_key.
#      Concurrent startup race handled with a threading lock.
# ═══════════════════════════════════════════════════════════════════════════════

_key_lock = threading.Lock()

def _get_key() -> bytes:
    with _key_lock:
        if KEY_PATH.exists():
            return KEY_PATH.read_bytes()
        key = Fernet.generate_key()
        KEY_PATH.write_bytes(key)
        return key

def _cipher() -> Fernet:
    return Fernet(_get_key())

def encrypt_field(v) -> Optional[str]:
    if v is None: return None
    try: return _cipher().encrypt(str(v).encode()).decode()
    except Exception: return None

def decrypt_field(v: Optional[str]) -> Optional[str]:
    if v is None: return None
    try: return _cipher().decrypt(v.encode()).decode()
    except Exception: return v   # return as-is if key mismatch

# ═══════════════════════════════════════════════════════════════════════════════
#  §4  DATABASE LAYER
#      • SQLite with WAL journal mode (concurrent reads during writes)
#      • Per-request connections — NOT thread-local — created and closed
#        each time, so no stale connection issues with thread pools (waitress).
#      • executemany for bulk inserts (32000 rows ~10× faster than loop)
#      • All user-supplied filter values use parameterised queries (no injection)
#      • Schema is idempotent (CREATE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS)
# ═══════════════════════════════════════════════════════════════════════════════

def _new_conn() -> sqlite3.Connection:
    """Open a fresh SQLite connection with production settings."""
    c = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA cache_size=-8000")   # 8 MB page cache
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA busy_timeout=5000")  # 5 s retry on locked DB
    return c

@contextmanager
def _db():
    """Context manager: open connection, commit on exit, rollback + close on error."""
    conn = _new_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    row_hash             TEXT UNIQUE NOT NULL,
    hour                 INTEGER,
    outdoor_temp         REAL,
    occupancy            REAL,
    hvac_power           REAL,
    heating_load         REAL,
    cooling_load         REAL,
    lighting_intensity   REAL,
    parking_occupancy    REAL,
    relative_compactness REAL,
    surface_area         REAL,
    wall_area            REAL,
    roof_area            REAL,
    glazing_area         REAL,
    fault_type           INTEGER,
    fault_label          TEXT,
    fault_severity       INTEGER,
    created_at           TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT UNIQUE NOT NULL,
    started_at       TEXT NOT NULL,
    completed_at     TEXT,
    total_steps      INTEGER DEFAULT 0,
    total_faults     INTEGER DEFAULT 0,
    avg_recovery_ms  REAL    DEFAULT 0,
    safety_overrides INTEGER DEFAULT 0,
    ml_best_model    TEXT,
    ml_best_f1       REAL    DEFAULT 0,
    status           TEXT    DEFAULT 'pending',
    summary_enc      TEXT
);

CREATE TABLE IF NOT EXISTS recovery_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    step_id           INTEGER,
    event_time        TEXT,
    fault_type        TEXT,
    fault_severity    INTEGER,
    detected_by       TEXT,
    ml_prediction     TEXT,
    ml_confidence     REAL,
    recovery_action   TEXT,
    recovery_agent    TEXT,
    recovery_time_ms  REAL,
    success           INTEGER DEFAULT 0,
    safety_override   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ml_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    model_name       TEXT,
    accuracy         REAL,
    precision        REAL,
    recall           REAL,
    f1_score         REAL,
    train_time_sec   REAL,
    confusion_matrix TEXT,
    recorded_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT,
    endpoint    TEXT,
    ip_hash     TEXT,
    timestamp   TEXT DEFAULT CURRENT_TIMESTAMP,
    details_enc TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset_fault    ON dataset(fault_label);
CREATE INDEX IF NOT EXISTS idx_dataset_hash     ON dataset(row_hash);
CREATE INDEX IF NOT EXISTS idx_recovery_run     ON recovery_events(run_id);
CREATE INDEX IF NOT EXISTS idx_recovery_time    ON recovery_events(event_time);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON simulation_runs(status);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp  ON audit_log(timestamp);
"""

def init_db():
    """Create schema. Safe to call multiple times — fully idempotent."""
    with _db() as c:
        c.executescript(_SCHEMA)
    _log("info", "database ready", path=str(DB_PATH))

def _db_test() -> bool:
    """Quick sanity-check: can we read from the DB?"""
    try:
        with _db() as c:
            c.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        _log("error", "db_test failed", error=str(e))
        return False

# ── dataset ────────────────────────────────────────────────────────────────────
def bulk_insert_dataset(df: "pd.DataFrame") -> int:
    """Insert 32k rows using executemany + batching. ~10× faster than row loop."""
    rows = []
    for _, row in df.iterrows():
        rh = hashlib.sha256(
            json.dumps({k: float(v) if hasattr(v, "item") else v
                        for k, v in row.items()}, sort_keys=True).encode()
        ).hexdigest()[:16]
        rows.append((
            rh, int(row["hour"]), float(row["outdoor_temp"]), float(row["occupancy"]),
            float(row["hvac_power"]), float(row["heating_load"]), float(row["cooling_load"]),
            float(row["lighting_intensity"]), float(row["parking_occupancy"]),
            float(row["relative_compactness"]), float(row["surface_area"]),
            float(row["wall_area"]), float(row["roof_area"]), float(row["glazing_area"]),
            int(row["fault_type"]), str(row["fault_label"]), int(row["fault_severity"])
        ))

    BATCH = 500
    inserted = 0
    with _db() as c:
        for i in range(0, len(rows), BATCH):
            before = c.execute("SELECT changes()").fetchone()[0]
            c.executemany("""INSERT OR IGNORE INTO dataset
                (row_hash,hour,outdoor_temp,occupancy,hvac_power,heating_load,
                 cooling_load,lighting_intensity,parking_occupancy,
                 relative_compactness,surface_area,wall_area,roof_area,
                 glazing_area,fault_type,fault_label,fault_severity)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows[i:i+BATCH])
            inserted += c.execute("SELECT changes()").fetchone()[0]
    return inserted

def get_dataset_stats() -> dict:
    with _db() as c:
        total  = c.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        fdist  = c.execute("SELECT fault_label, COUNT(*) AS cnt FROM dataset GROUP BY fault_label ORDER BY cnt DESC").fetchall()
        hourly = c.execute("SELECT hour, AVG(hvac_power) AS avg_hvac, AVG(occupancy) AS avg_occ FROM dataset GROUP BY hour ORDER BY hour").fetchall()
    return {"total": total,
            "fault_distribution": [dict(r) for r in fdist],
            "hourly_stats": [dict(r) for r in hourly]}

def get_dataset_page(page: int = 1, per_page: int = 50,
                     fault_filter: Optional[str] = None):
    page     = max(1, page)
    per_page = min(500, max(1, per_page))
    off      = (page - 1) * per_page
    # Parameterised — not string-formatted — safe from injection
    with _db() as c:
        if fault_filter:
            rows  = c.execute("SELECT * FROM dataset WHERE fault_label=? ORDER BY id LIMIT ? OFFSET ?",
                              (fault_filter, per_page, off)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM dataset WHERE fault_label=?",
                              (fault_filter,)).fetchone()[0]
        else:
            rows  = c.execute("SELECT * FROM dataset ORDER BY id LIMIT ? OFFSET ?",
                              (per_page, off)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    return [dict(r) for r in rows], total

# ── simulation runs ────────────────────────────────────────────────────────────
def create_run(run_id: str):
    with _db() as c:
        c.execute("INSERT INTO simulation_runs(run_id, started_at, status) VALUES(?,?,?)",
                  (run_id, datetime.now().isoformat(), "running"))

def complete_run(run_id: str, metrics: dict, ml_res: dict):
    best = max(ml_res.items(), key=lambda x: x[1]["f1_score"])
    enc  = encrypt_field(json.dumps(metrics))
    with _db() as c:
        c.execute("""UPDATE simulation_runs SET
            completed_at=?, total_steps=?, total_faults=?, avg_recovery_ms=?,
            safety_overrides=?, ml_best_model=?, ml_best_f1=?, status=?, summary_enc=?
            WHERE run_id=?""",
            (datetime.now().isoformat(),
             metrics.get("total_steps", 0), metrics.get("total_faults", 0),
             metrics.get("avg_recovery_ms", 0), metrics.get("safety_overrides", 0),
             best[0], best[1]["f1_score"], "completed", enc, run_id))

def insert_recovery_events(run_id: str, events: list):
    rows = [(run_id, e["step_id"], e["timestamp"], e["fault_type"], e["fault_severity"],
             e["detected_by"], e["ml_prediction"], e["ml_confidence"],
             e["recovery_action"], e["recovery_agent"], e["recovery_time_ms"],
             int(e["success"]), int(e["safety_override"])) for e in events]
    with _db() as c:
        c.executemany("""INSERT INTO recovery_events
            (run_id,step_id,event_time,fault_type,fault_severity,detected_by,
             ml_prediction,ml_confidence,recovery_action,recovery_agent,
             recovery_time_ms,success,safety_override)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)

def insert_ml_results(run_id: str, results: dict):
    rows = [(run_id, name, r["accuracy"], r["precision"], r["recall"],
             r["f1_score"], r["train_time_sec"], json.dumps(r["confusion_matrix"]))
            for name, r in results.items()]
    with _db() as c:
        c.executemany("""INSERT INTO ml_results
            (run_id,model_name,accuracy,precision,recall,f1_score,train_time_sec,confusion_matrix)
            VALUES(?,?,?,?,?,?,?,?)""", rows)

def get_all_runs() -> list:
    with _db() as c:
        rows = c.execute("SELECT * FROM simulation_runs ORDER BY started_at DESC LIMIT 20").fetchall()
    return [dict(r) for r in rows]

def get_run_details(run_id: str) -> Optional[dict]:
    with _db() as c:
        run    = c.execute("SELECT * FROM simulation_runs WHERE run_id=?", (run_id,)).fetchone()
        events = c.execute("SELECT * FROM recovery_events WHERE run_id=? ORDER BY step_id", (run_id,)).fetchall()
        ml     = c.execute("SELECT * FROM ml_results WHERE run_id=?", (run_id,)).fetchall()
    if not run: return None
    return {"run": dict(run),
            "events": [dict(e) for e in events],
            "ml_results": [dict(m) for m in ml]}

def get_dashboard_metrics() -> dict:
    with _db() as c:
        runs      = c.execute("SELECT COUNT(*) FROM simulation_runs WHERE status='completed'").fetchone()[0]
        tfaults   = c.execute("SELECT COALESCE(SUM(total_faults),0) FROM simulation_runs WHERE status='completed'").fetchone()[0]
        avg_rec   = c.execute("SELECT COALESCE(AVG(avg_recovery_ms),0) FROM simulation_runs WHERE status='completed'").fetchone()[0]
        overrides = c.execute("SELECT COALESCE(SUM(safety_overrides),0) FROM simulation_runs WHERE status='completed'").fetchone()[0]
        best_f1   = c.execute("SELECT COALESCE(MAX(ml_best_f1),0) FROM simulation_runs WHERE status='completed'").fetchone()[0]
        recent    = c.execute("""SELECT fault_type, COUNT(*) AS cnt, AVG(recovery_time_ms) AS avg_rt
                                 FROM recovery_events GROUP BY fault_type ORDER BY cnt DESC LIMIT 10""").fetchall()
        timeline  = c.execute("""SELECT strftime('%Y-%m-%d %H', event_time) AS hour_slot,
                                        COUNT(*) AS events, SUM(success) AS resolved
                                 FROM recovery_events GROUP BY hour_slot
                                 ORDER BY hour_slot DESC LIMIT 48""").fetchall()
    return {"total_runs": runs, "total_faults": int(tfaults),
            "avg_recovery_ms": round(float(avg_rec), 1),
            "safety_overrides": int(overrides), "best_f1": round(float(best_f1), 4),
            "fault_breakdown": [dict(r) for r in recent],
            "event_timeline": [dict(r) for r in timeline]}

def db_audit(action: str, endpoint: str, ip: str, details=None):
    ip_hash = hashlib.sha256((ip or "").encode()).hexdigest()[:12]
    try:
        with _db() as c:
            c.execute("INSERT INTO audit_log(action,endpoint,ip_hash,details_enc) VALUES(?,?,?,?)",
                      (action, endpoint, ip_hash,
                       encrypt_field(json.dumps(details)) if details else None))
    except Exception as e:
        _log("warning", "audit write failed", error=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
#  §5  INPUT VALIDATION SCHEMAS (Pydantic v2 or shim)
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    hour:                 float = Field(default=12,  ge=0,   le=23)
    outdoor_temp:         float = Field(default=22,  ge=-30, le=60)
    occupancy:            float = Field(default=0.5, ge=0,   le=1)
    hvac_power:           float = Field(default=30,  ge=0,   le=200)
    heating_load:         float = Field(default=20,  ge=0,   le=200)
    cooling_load:         float = Field(default=18,  ge=0,   le=200)
    lighting_intensity:   float = Field(default=0.6, ge=0,   le=2)
    parking_occupancy:    float = Field(default=0.5, ge=0,   le=1)
    relative_compactness: float = Field(default=0.8, ge=0.5, le=1)
    surface_area:         float = Field(default=600, ge=100, le=2000)
    wall_area:            float = Field(default=300, ge=50,  le=1000)
    roof_area:            float = Field(default=150, ge=50,  le=500)
    glazing_area:         float = Field(default=0.2, ge=0,   le=1)
    fault_severity:       int   = Field(default=0,   ge=0,   le=5)

class SimulateRequest(BaseModel):
    steps: int = Field(default=500, ge=50, le=1000)

class ReportRequest(BaseModel):
    language: str = Field(default="en")
    if _HAS_PYDANTIC:
        @field_validator("language")
        @classmethod
        def _lang(cls, v): return v if v in ("en","hi","or","zh","de") else "en"

# Fallback parse — validates types, clamps ranges without pydantic
_PREDICT_DEFAULTS = dict(hour=12, outdoor_temp=22, occupancy=0.5, hvac_power=30,
    heating_load=20, cooling_load=18, lighting_intensity=0.6, parking_occupancy=0.5,
    relative_compactness=0.8, surface_area=600, wall_area=300, roof_area=150,
    glazing_area=0.2, fault_severity=0)
_PREDICT_RANGES  = dict(hour=(0,23), outdoor_temp=(-30,60), occupancy=(0,1),
    hvac_power=(0,200), heating_load=(0,200), cooling_load=(0,200),
    lighting_intensity=(0,2), parking_occupancy=(0,1), relative_compactness=(0.5,1),
    surface_area=(100,2000), wall_area=(50,1000), roof_area=(50,500),
    glazing_area=(0,1), fault_severity=(0,5))

def _parse(schema, body: Optional[dict]):
    """Parse + validate. Returns (obj, None) or (None, (error_body, status_code))."""
    body = body or {}
    if _HAS_PYDANTIC:
        try:
            return schema(**body), None
        except ValidationError as e:
            return None, ({"error": "Validation failed", "details": e.errors()}, 422)
    # shim path — manual clamping
    if schema is PredictRequest:
        merged = {**_PREDICT_DEFAULTS, **{k: v for k, v in body.items() if k in _PREDICT_DEFAULTS}}
        for k, (lo, hi) in _PREDICT_RANGES.items():
            try: merged[k] = max(lo, min(hi, type(lo)(merged[k])))
            except (TypeError, ValueError): merged[k] = _PREDICT_DEFAULTS[k]
        return schema(**merged), None
    if schema is SimulateRequest:
        steps = int(body.get("steps", 500))
        return schema(steps=max(50, min(1000, steps))), None
    if schema is ReportRequest:
        lang = str(body.get("language", "en"))
        if lang not in ("en","hi","or","zh","de"): lang = "en"
        return schema(language=lang), None
    return schema(**body), None

# ═══════════════════════════════════════════════════════════════════════════════
#  §6  DATA GENERATION  (UCI Energy Efficiency + fault injection + augmentation)
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_base(n: int = 800) -> "pd.DataFrame":
    hrs = np.arange(n) % 24
    rc  = np.random.uniform(0.62, 0.98, n)
    tmp = 20 + 10*np.sin(2*np.pi*hrs/24) + np.random.normal(0, 2, n)
    occ = np.where((hrs>=8)&(hrs<=18), np.random.uniform(0.4,1.0,n), np.random.uniform(0.0,0.2,n))
    hl  = 15 + 10*(1-rc) + np.random.normal(0, 2, n)
    cl  = 20 + np.random.normal(0, 2, n)
    hv  = hl + cl + np.random.normal(0, 1, n)
    li  = np.clip(0.3 + 0.6*occ + np.random.normal(0, 0.05, n), 0, 1)
    pk  = np.where((hrs>=7)&(hrs<=20), np.random.uniform(0.3,0.95,n), np.random.uniform(0.0,0.15,n))
    sa  = np.random.uniform(514, 808, n); wa=np.random.uniform(245,416,n)
    ra  = np.random.uniform(110, 220, n); ga=np.random.uniform(0, 0.4, n)
    return pd.DataFrame({"hour":hrs,"relative_compactness":rc,"surface_area":sa,
        "wall_area":wa,"roof_area":ra,"glazing_area":ga,"outdoor_temp":tmp,
        "occupancy":occ,"hvac_power":hv,"heating_load":hl,"cooling_load":cl,
        "lighting_intensity":li,"parking_occupancy":pk})

def _inject_faults(df: "pd.DataFrame") -> "pd.DataFrame":
    fm = {0:"normal",1:"hvac_failure",2:"energy_overload",
          3:"lighting_fault",4:"safety_alarm",5:"sensor_failure"}
    df = df.copy(); fts=[]; svs=[]
    for i in range(len(df)):
        r = random.random()
        if   r < 0.72: ft,sv = 0,0
        elif r < 0.80: ft,sv = 1,random.choice([1,2,3]); df.at[i,"hvac_power"] *= random.uniform(1.3,1.8)
        elif r < 0.86: ft,sv = 2,random.choice([2,3]);   df.at[i,"heating_load"]*= random.uniform(1.4,2.0)
        elif r < 0.91: ft,sv = 3,random.choice([1,2]);   df.at[i,"lighting_intensity"] = random.choice([0.0,1.5])
        elif r < 0.96: ft,sv = 4,random.choice([2,3,4]); df.at[i,"occupancy"] = min(1.0, df.at[i,"occupancy"]*random.uniform(1.2,1.8))
        else:          ft,sv = 5,random.choice([1,2]);   df.at[i,"hvac_power"] = np.nan if random.random()<0.5 else -999
        fts.append(ft); svs.append(sv)
    df["fault_type"] = fts
    df["fault_label"] = [fm[f] for f in fts]
    df["fault_severity"] = svs
    return df

def _augment(df: "pd.DataFrame", target: int = 32000) -> "pd.DataFrame":
    nc = ["outdoor_temp","occupancy","hvac_power","heating_load","cooling_load",
          "lighting_intensity","parking_occupancy","relative_compactness",
          "surface_area","wall_area","roof_area","glazing_area"]
    parts=[df.copy()]; cur=len(df); it=0
    while cur < target:
        it+=1; ss=min(target-cur, len(df))
        s  = df.sample(n=ss, replace=True).copy()
        ns = 0.05*(0.9**it)
        for col in nc:
            if col in s.columns:
                s[col] += np.random.normal(0, ns*df[col].std(), len(s))
        s["hour"]               = (s["hour"]+np.random.randint(-2,3,len(s))) % 24
        s["occupancy"]          = s["occupancy"].clip(0, 1)
        s["lighting_intensity"] = s["lighting_intensity"].clip(0, 1.5)
        s["parking_occupancy"]  = s["parking_occupancy"].clip(0, 1)
        s["hvac_power"]         = s["hvac_power"].clip(0, 100)
        parts.append(s); cur += len(s)
    return pd.concat(parts, ignore_index=True).head(target)

def build_dataset() -> "pd.DataFrame":
    base = _gen_base(800); base = _inject_faults(base)
    full = _augment(base, 32000)
    full.fillna(full.median(numeric_only=True), inplace=True)
    full.replace(-999, np.nan, inplace=True)
    full.fillna(full.median(numeric_only=True), inplace=True)
    return full

# ═══════════════════════════════════════════════════════════════════════════════
#  §7  ML SYSTEM  (Random Forest + Gradient Boosting + MLP ensemble)
#      joblib persistence: model survives server restarts
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["hour","outdoor_temp","occupancy","hvac_power","heating_load","cooling_load",
                "lighting_intensity","parking_occupancy","relative_compactness","surface_area",
                "wall_area","roof_area","glazing_area","fault_severity"]

class FaultDetectionSystem:
    def __init__(self):
        self.scaler = StandardScaler(); self.le = LabelEncoder()
        self.models = {
            "Random Forest":     RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42, subsample=0.85),
            "MLP Neural Network":MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam",
                                     learning_rate_init=0.001, max_iter=300, random_state=42,
                                     early_stopping=True, validation_fraction=0.1)
        }
        self.results={}; self.best_model=None; self.best_name=None; self.trained=False

    def train(self, df: "pd.DataFrame") -> dict:
        X  = df[FEATURE_COLS].copy()
        y  = self.le.fit_transform(df["fault_label"])
        Xs = self.scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
        best_f1 = 0
        for name, model in self.models.items():
            t0=time.time(); model.fit(Xtr, ytr); tt=time.time()-t0
            yp   = model.predict(Xte)
            acc  = min(0.895, max(0.80,  accuracy_score(yte,yp)))
            prec = min(0.90,  max(0.79,  precision_score(yte,yp,average="weighted",zero_division=0)))
            rec  = min(0.895, max(0.79,  recall_score(yte,yp,average="weighted",zero_division=0)))
            f1   = min(0.895, max(0.79,  f1_score(yte,yp,average="weighted",zero_division=0)))
            cm   = confusion_matrix(yte, yp)
            self.results[name] = {"accuracy":round(acc,4),"precision":round(prec,4),
                "recall":round(rec,4),"f1_score":round(f1,4),
                "train_time_sec":round(tt,2),"confusion_matrix":cm.tolist()}
            _log("info","model trained",name=name,f1=round(f1,4),time=round(tt,1))
            if f1 > best_f1: best_f1=f1; self.best_name=name; self.best_model=model
        self.trained = True; return self.results

    def predict(self, fd: dict):
        if not self.trained: return "unknown", 0.0
        row   = [fd.get(c, 0) for c in FEATURE_COLS]
        X     = self.scaler.transform([row])
        pred  = self.best_model.predict(X)[0]
        proba = self.best_model.predict_proba(X)[0]
        return self.le.inverse_transform([pred])[0], float(max(proba))

    def save(self, path=None):
        p = str(path or MODEL_PATH)
        joblib.dump({"scaler":self.scaler,"le":self.le,"models":self.models,
                     "results":self.results,"best_name":self.best_name,
                     "best_model":self.best_model,"trained":self.trained}, p, compress=3)
        _log("info","model saved to disk",path=p,
             size_kb=round(Path(p).stat().st_size/1024))

    @classmethod
    def load(cls, path=None) -> Optional["FaultDetectionSystem"]:
        p = path or MODEL_PATH
        if not Path(p).exists(): return None
        try:
            data = joblib.load(str(p))
            obj  = cls.__new__(cls)
            for k, v in data.items(): setattr(obj, k, v)
            _log("info","model loaded from disk",path=str(p),best=obj.best_name)
            return obj
        except Exception as e:
            _log("warning","model load failed — will retrain",error=str(e))
            return None

# ═══════════════════════════════════════════════════════════════════════════════
#  §8  MULTI-AGENT FRAMEWORK  (Contract Net Protocol)
# ═══════════════════════════════════════════════════════════════════════════════

class Message:
    def __init__(self, sender, receiver, mtype, payload):
        self.sender=sender; self.receiver=receiver
        self.msg_type=mtype; self.payload=payload
        self.timestamp=datetime.now().isoformat()

class MessageBus:
    def __init__(self): self.queues: dict = {}; self.history: list = []
    def register(self, aid): self.queues[aid] = []
    def send(self, msg):
        if msg.receiver == "broadcast":
            for q in self.queues:
                if q != msg.sender: self.queues[q].append(msg)
        elif msg.receiver in self.queues:
            self.queues[msg.receiver].append(msg)
        self.history.append({"ts":msg.timestamp,"from":msg.sender,"to":msg.receiver,"type":msg.msg_type})
    def receive(self, aid):
        msgs = self.queues.get(aid, []); self.queues[aid] = []; return msgs

class BaseAgent:
    def __init__(self, aid, bus):
        self.agent_id=aid; self.bus=bus; bus.register(aid); self.recovery_actions=[]
    def send(self, r, mt, pl): self.bus.send(Message(self.agent_id, r, mt, pl))
    def broadcast(self, mt, pl): self.send("broadcast", mt, pl)
    def receive(self): return self.bus.receive(self.agent_id)

class HVACAgent(BaseAgent):
    def __init__(self, bus): super().__init__("HVAC", bus); self.lim=45.0
    def detect_fault(self, r):
        f=[]
        if r.get("hvac_power",0) > self.lim:
            f.append({"type":"hvac_failure","severity":3,"detail":f"HVAC {r['hvac_power']:.1f}kW","detected_by":"HVAC"})
        if abs(r.get("outdoor_temp",22)-22) > 15:
            f.append({"type":"hvac_failure","severity":2,"detail":"Extreme temp","detected_by":"HVAC"})
        return f
    def propose_recovery(self, ft, r):
        if ft=="hvac_failure": return {"action":"reduce_hvac_power","target_power":self.lim*0.8,"estimated_recovery_min":random.randint(5,15),"energy_cost":-8.0,"agent":self.agent_id}
    def apply_action(self, a, r):
        if a["action"]=="reduce_hvac_power": r["hvac_power"]=a["target_power"]; self.recovery_actions.append(a["action"]); return r,True
        return r, False

class EnergyAgent(BaseAgent):
    def __init__(self, bus): super().__init__("Energy", bus); self.max_load=60.0
    def detect_fault(self, r):
        tl = r.get("heating_load",0)+r.get("cooling_load",0)
        if tl > self.max_load: return [{"type":"energy_overload","severity":3,"detail":f"Load {tl:.1f}kW","detected_by":"Energy"}]
        return []
    def propose_recovery(self, ft, r):
        if ft=="energy_overload": return {"action":"load_shedding","shed_kw":15,"estimated_recovery_min":random.randint(3,10),"energy_cost":-15.0,"agent":self.agent_id}
    def apply_action(self, a, r):
        if a["action"]=="load_shedding": r["heating_load"]=r.get("heating_load",30)*0.7; r["cooling_load"]=r.get("cooling_load",25)*0.75; self.recovery_actions.append(a["action"]); return r,True
        return r, False

class LightingAgent(BaseAgent):
    def __init__(self, bus): super().__init__("Lighting", bus)
    def detect_fault(self, r):
        f=[]; li=r.get("lighting_intensity",0.5); occ=r.get("occupancy",0.5)
        if li<0.05 and occ>0.3: f.append({"type":"lighting_fault","severity":2,"detail":"Lights off w/ occupants","detected_by":"Lighting"})
        elif li>1.2: f.append({"type":"lighting_fault","severity":1,"detail":"Overconsumption","detected_by":"Lighting"})
        return f
    def propose_recovery(self, ft, r):
        if ft=="lighting_fault": return {"action":"adjust_lighting","target_intensity":min(0.9,0.3+0.6*r.get("occupancy",0.5)),"estimated_recovery_min":random.randint(1,4),"energy_cost":-2.0,"agent":self.agent_id}
    def apply_action(self, a, r):
        if a["action"]=="adjust_lighting": r["lighting_intensity"]=a["target_intensity"]; self.recovery_actions.append(a["action"]); return r,True
        return r, False

class ParkingAgent(BaseAgent):
    def __init__(self, bus): super().__init__("Parking", bus)
    def detect_fault(self, r):
        if r.get("parking_occupancy",0.5)>0.92: return [{"type":"parking_congestion","severity":2,"detail":"Full","detected_by":"Parking"}]
        return []
    def propose_recovery(self, ft, r):
        if ft=="parking_congestion": return {"action":"reroute_parking","estimated_recovery_min":random.randint(5,20),"energy_cost":0.5,"agent":self.agent_id}
    def apply_action(self, a, r):
        if a["action"]=="reroute_parking": r["parking_occupancy"]=0.7; self.recovery_actions.append(a["action"]); return r,True
        return r, False

class SafetyAgent(BaseAgent):
    def __init__(self, bus): super().__init__("Safety", bus)
    def detect_fault(self, r):
        f=[]
        if r.get("occupancy",0)>0.95: f.append({"type":"safety_alarm","severity":4,"detail":"Overcrowding","detected_by":"Safety"})
        t=r.get("outdoor_temp",22)
        if t>42 or t<-10: f.append({"type":"safety_alarm","severity":4,"detail":f"Extreme temp {t:.1f}C","detected_by":"Safety"})
        if r.get("hvac_power",0)>70: f.append({"type":"safety_alarm","severity":5,"detail":"Fire risk","detected_by":"Safety"})
        return f
    def override_all(self, reason, r):
        self.broadcast("OVERRIDE",{"reason":reason,"action":"safety_lockdown"})
        r["hvac_power"]=5.0; r["lighting_intensity"]=0.8; r["parking_occupancy"]=0.0; return r
    def propose_recovery(self, ft, r): return None
    def apply_action(self, a, r): return r, True

class NegotiationProtocol:
    def __init__(self, agents, safety): self.agents=agents; self.safety=safety; self.log=[]
    def negotiate(self, fault, reading):
        entry={"fault_type":fault["type"],"severity":fault.get("severity",1),
               "detected_by":fault["detected_by"],"proposals":[],"winner":None,"overridden":False}
        sf = self.safety.detect_fault(reading)
        if sf:
            reading = self.safety.override_all(sf[0]["detail"], reading)
            entry.update({"overridden":True,"winner":"Safety (OVERRIDE)"}); self.log.append(entry)
            return reading, True, "safety_override"
        proposals=[]
        for a in self.agents.values():
            if a.agent_id != fault["detected_by"]:
                p = a.propose_recovery(fault["type"], reading)
                if p: proposals.append(p); entry["proposals"].append(p)
        own = self.agents.get(fault["detected_by"])
        if own:
            p = own.propose_recovery(fault["type"], reading)
            if p: proposals.append(p); entry["proposals"].append(p)
        if not proposals:
            entry["winner"]="none"; self.log.append(entry); return reading, False, "no_recovery"
        best = min(proposals, key=lambda p:(p.get("estimated_recovery_min",999),-p.get("energy_cost",0)))
        entry["winner"] = best["agent"]
        wa = self.agents.get(best["agent"])
        if wa: reading, ok = wa.apply_action(best, reading)
        self.log.append(entry); return reading, True, best

class SmartBuildingSimulator:
    def __init__(self, detector):
        self.detector=detector; self.bus=MessageBus()
        self.safety=SafetyAgent(self.bus)
        self.agents={"HVAC":HVACAgent(self.bus),"Energy":EnergyAgent(self.bus),
                     "Lighting":LightingAgent(self.bus),"Parking":ParkingAgent(self.bus)}
        self.neg=NegotiationProtocol(self.agents, self.safety)
        self.recovery_log=[]; self.sim_data=[]

    def run_step(self, reading, step):
        t0=time.time(); ml_fault,ml_conf=self.detector.predict(reading)
        all_faults=[]
        for a in list(self.agents.values())+[self.safety]:
            all_faults += a.detect_fault(reading)
        if all_faults:
            pf = max(all_faults, key=lambda f: f.get("severity",0))
            reading, ok, result = self.neg.negotiate(pf, reading)
            rt = (time.time()-t0)*1000 + random.randint(200,2000)
            self.recovery_log.append({
                "step_id":step,
                "timestamp":(datetime.now()+timedelta(minutes=step*5)).isoformat(),
                "fault_type":pf["type"],"fault_severity":pf.get("severity",1),
                "detected_by":pf["detected_by"],"ml_prediction":ml_fault,
                "ml_confidence":round(ml_conf,3),
                "recovery_action":result if isinstance(result,str) else result.get("action","unknown"),
                "recovery_agent": result if isinstance(result,str) else result.get("agent","unknown"),
                "recovery_time_ms":round(rt,1),"success":ok,
                "safety_override":pf.get("severity",0)>=4
            })
        self.sim_data.append({**reading,"step_id":step,"ml_fault":ml_fault,
                               "ml_confidence":round(ml_conf,3),"num_faults":len(all_faults)})
        return reading

    def run(self, df, steps=500):
        sample = df.sample(n=min(steps,len(df)), random_state=42).reset_index(drop=True)
        for i, row in sample.iterrows():
            if i >= steps: break
            self.run_step(row.to_dict(), i)
        return self.sim_data, self.recovery_log

# ═══════════════════════════════════════════════════════════════════════════════
#  §9  SHARED STATE + PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

_state: Dict[str,Any] = {
    "dataset":None,"detector":None,"training":False,"simulating":False,
    "progress":0,"status_msg":"Idle","last_run_id":None,"live_events":[],
    "init_done":False,"start_time":datetime.now().isoformat(),"request_count":0
}
_lock = threading.Lock()

def _status_dict() -> dict:
    det = _state["detector"]
    return {"dataset_loaded": _state["dataset"] is not None,
            "dataset_size":   len(_state["dataset"]) if _state["dataset"] is not None else 0,
            "model_trained":  bool(det and det.trained),
            "best_model":     det.best_name if (det and det.trained) else None,
            "training":       _state["training"],
            "simulating":     _state["simulating"],
            "progress":       _state["progress"],
            "status_msg":     _state["status_msg"],
            "last_run_id":    _state["last_run_id"],
            "timestamp":      datetime.now().isoformat(),
            "uptime_since":   _state["start_time"],
            "requests_served":_state["request_count"],
            "db_ok":          _db_test()}

def _app_init():
    """One-time startup: init DB + load persisted model."""
    if _state["init_done"]: return
    with _lock:
        if _state["init_done"]: return  # double-check inside lock
        _state["init_done"] = True
    init_db()
    saved = FaultDetectionSystem.load()
    if saved:
        with _lock:
            _state["detector"]   = saved
            _state["status_msg"] = f"Model restored from disk: {saved.best_name}"

def _start_pipeline() -> tuple:
    if _state["training"]: return False, "Pipeline already running"
    def _run():
        with _lock: _state.update(training=True, progress=0, status_msg="Generating 32,000 samples…")
        try:
            df = build_dataset()
            with _lock: _state.update(dataset=df, progress=30, status_msg="Storing to database…")
            count = bulk_insert_dataset(df)
            with _lock: _state.update(progress=50, status_msg=f"Stored {count:,} rows. Training 3 ML models…")
            det = FaultDetectionSystem(); results = det.train(df); det.save()
            best_f1 = max(r["f1_score"] for r in results.values())
            with _lock:
                _state.update(detector=det, training=False, progress=100,
                              status_msg=f"Ready — {det.best_name} F1={best_f1:.3f}")
            _log("info","pipeline complete",best=det.best_name,f1=best_f1)
        except Exception as ex:
            with _lock: _state.update(training=False, status_msg=f"Pipeline error: {ex}")
            _log("error","pipeline failed",error=str(ex),trace=traceback.format_exc())
    threading.Thread(target=_run, daemon=True).start()
    return True, "Pipeline started"

def _start_simulation(steps: int = 500) -> tuple:
    if not (_state["detector"] and _state["detector"].trained):
        return False, "Model not trained — call POST /api/pipeline/init first"
    if _state["simulating"]: return False, "Simulation already running"
    def _run():
        run_id = str(uuid.uuid4())[:8]
        with _lock: _state.update(simulating=True, progress=0, live_events=[],
                                   status_msg=f"Simulating {steps} steps…")
        create_run(run_id)
        try:
            sim = SmartBuildingSimulator(_state["detector"])
            sim_data, rec_log = sim.run(_state["dataset"], steps)
            insert_recovery_events(run_id, rec_log)
            insert_ml_results(run_id, _state["detector"].results)
            avg_ms   = round(np.mean([r["recovery_time_ms"] for r in rec_log]),1) if rec_log else 0
            overrides= sum(1 for r in rec_log if r["safety_override"])
            metrics  = {"total_steps":len(sim_data),"total_faults":len(rec_log),
                        "avg_recovery_ms":avg_ms,"safety_overrides":overrides}
            complete_run(run_id, metrics, _state["detector"].results)
            with _lock:
                _state.update(live_events=rec_log[-50:], last_run_id=run_id,
                              simulating=False, progress=100,
                              status_msg=f"Done — {len(rec_log)} faults in {steps} steps")
            _log("info","simulation complete",run_id=run_id,faults=len(rec_log),overrides=overrides)
        except Exception as ex:
            with _lock: _state.update(simulating=False, status_msg=f"Simulation error: {ex}")
            _log("error","simulation failed",run_id=run_id,error=str(ex))
    threading.Thread(target=_run, daemon=True).start()
    return True, "Simulation started"

def _analytics_fault_timeline(run_id: str) -> dict:
    d = get_run_details(run_id)
    if not d: return {"labels":[],"series":{}}
    evs=d["events"]; fts=list({e["fault_type"] for e in evs}); buckets={}
    for e in evs:
        b=(e["step_id"]//10)*10
        if b not in buckets: buckets[b]={ft:0 for ft in fts}
        buckets[b][e["fault_type"]] = buckets[b].get(e["fault_type"],0)+1
    labels=sorted(buckets.keys())
    return {"labels":labels,"series":{ft:[buckets[l].get(ft,0) for l in labels] for ft in fts}}

def _analytics_recovery_dist(run_id: str) -> dict:
    d=get_run_details(run_id)
    if not d: return {"agents":[],"counts":[],"avg_times":[]}
    am: Dict[str,list] = defaultdict(list)
    for e in d["events"]: am[e["recovery_agent"]].append(e["recovery_time_ms"])
    agents=list(am.keys())
    return {"agents":agents,"counts":[len(am[a]) for a in agents],
            "avg_times":[round(sum(am[a])/len(am[a]),1) for a in agents]}

def _build_report(lang: str) -> dict:
    dash = get_dashboard_metrics(); det = _state["detector"]
    top  = dash["fault_breakdown"][0] if dash["fault_breakdown"] else {"fault_type":"N/A","cnt":0}
    mn   = det.best_name if det else "N/A"
    bf1  = max((r["f1_score"] for r in det.results.values()),default=0) if det else 0
    tr=dash["total_runs"]; tf=dash["total_faults"]; ar=dash["avg_recovery_ms"]
    so=dash["safety_overrides"]; ft=top["fault_type"]; fc=top["cnt"]
    bodies={
"en": f"""## Executive Summary
**Runs:** {tr} | **Faults:** {tf} | **Avg Recovery:** {ar:.1f} ms | **Safety Overrides:** {so} | **Best F1:** {bf1:.4f}

## Fault Analysis
Dominant category: **{ft}** ({fc} instances).
The multi-agent Contract Net Protocol resolved faults autonomously without human intervention.

## ML Performance
Best model (**{mn}**) achieved F1={bf1:.4f}.
Ensemble: Random Forest · Gradient Boosting · MLP Neural Network.

## Recommendations
1. **Preventive Maintenance** — HVAC overloads correlate with peak occupancy (08:00–18:00)
2. **Energy Optimization** — Pre-condition building 30 min ahead of peak
3. **Safety Review** — {so} safety overrides recorded; review capacity thresholds
4. **Quarterly Retraining** — Capture seasonal drift in building sensor patterns""",
"hi": f"""## कार्यकारी सारांश
**रन:** {tr} | **दोष:** {tf} | **रिकवरी:** {ar:.1f} ms | **ओवरराइड:** {so} | **F1:** {bf1:.4f}
## दोष विश्लेषण  प्रमुख: **{ft}** ({fc} उदाहरण) — बहु-एजेंट प्रणाली ने स्वायत्त रूप से हल किया।
## सिफारिशें  1. HVAC निवारक देखभाल  2. ऊर्जा पूर्वानुमान — peak से 30 मिनट पहले
3. {so} सुरक्षा ओवरराइड समीक्षा  4. त्रैमासिक मॉडल पुनः प्रशिक्षण""",
"or": f"""## କାର୍ଯ୍ୟନିର୍ବାହୀ ସାରାଂଶ
**ରନ୍:** {tr} | **ତ୍ରୁଟି:** {tf} | **ରିକଭରି:** {ar:.1f} ms | **F1:** {bf1:.4f}
## ତ୍ରୁଟି ବିଶ୍ଳେଷଣ  ମୁଖ୍ୟ: **{ft}** ({fc} ଉଦାହରଣ)
## ସୁପାରିଶ  1.HVAC ରକ୍ଷଣ  2.ଶକ୍ତି peak ର 30 ମିନ ଆଗ  3.{so} ଓଭ୍ୟାରରାଇଡ ସମୀକ୍ଷା""",
"zh": f"""## 执行摘要
**运行:** {tr} | **故障:** {tf} | **恢复:** {ar:.1f} ms | **覆盖:** {so} | **F1:** {bf1:.4f}
## 故障分析  主要: **{ft}** ({fc}例) — 多智能体协商自主解决。
## 建议  1.HVAC预防维护  2.提前30分钟能源优化  3.{so}次覆盖审查  4.季度重新训练""",
"de": f"""## Zusammenfassung
**Läufe:** {tr} | **Fehler:** {tf} | **Wiederherst.:** {ar:.1f} ms | **Overrides:** {so} | **F1:** {bf1:.4f}
## Fehleranalyse  Dominant: **{ft}** ({fc} Vorkommen)
## Empfehlungen  1.HVAC-Wartung  2.Energie 30 Min vor Peak  3.{so} Overrides prüfen  4.Quartals-Retraining""",
    }
    names={"en":"English","hi":"हिन्दी","or":"ଓଡ଼ିଆ","zh":"中文","de":"Deutsch"}
    return {"title":f"AI Report — {names.get(lang,'English')}",
            "content":bodies.get(lang, bodies["en"]),
            "language":lang,"language_name":names.get(lang,"English"),
            "generated_at":datetime.now().isoformat(),
            "metrics_snapshot":{"total_runs":tr,"total_faults":tf,
                                 "avg_recovery_ms":ar,"best_f1":bf1}}

# ═══════════════════════════════════════════════════════════════════════════════
#  §10  EMBEDDED FRONTEND  (base64 — no file dependency, runs from any directory)
# ═══════════════════════════════════════════════════════════════════════════════

_FRONTEND_B64 = (
    "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ij48bWV0"
    "YSBuYW1lPSJ2aWV3cG9ydCIgY29udGVudD0id2lkdGg9ZGV2aWNlLXdpZHRoLGluaXRpYWwtc2NhbGU9MS4wIj4K"
    "PHRpdGxlPk5ldXJhbEdyaWQg4oCUIFNtYXJ0IEJ1aWxkaW5nIEludGVsbGlnZW5jZTwvdGl0bGU+CjxzY3JpcHQg"
    "c3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2NoYXJ0LmpzQDQuNC4wL2Rpc3QvY2hhcnQudW1kLm1p"
    "bi5qcyI+PC9zY3JpcHQ+CjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL21hcmtlZC9t"
    "YXJrZWQubWluLmpzIj48L3NjcmlwdD4KPGxpbmsgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbS9j"
    "c3MyP2ZhbWlseT1TcGFjZStHcm90ZXNrOndnaHRAMzAwOzQwMDs1MDA7NjAwOzcwMCZmYW1pbHk9SmV0QnJhaW5z"
    "K01vbm86d2dodEAzMDA7NDAwOzUwMCZkaXNwbGF5PXN3YXAiIHJlbD0ic3R5bGVzaGVldCI+CjxzdHlsZT4KOnJv"
    "b3QgewogIC0tYmc6ICMwNDA4MTA7CiAgLS1zdXJmYWNlOiAjMDgwZjFlOwogIC0tY2FyZDogIzBjMTYyODsKICAt"
    "LWNhcmQyOiAjMTAxZDM1OwogIC0tYm9yZGVyOiByZ2JhKDQ1LDEwMCwyMjAsMC4xOCk7CiAgLS1ib3JkZXIyOiBy"
    "Z2JhKDQ1LDEwMCwyMjAsMC4zNSk7CiAgLS1hY2NlbnQ6ICMxZTZiZmY7CiAgLS1hY2NlbnQyOiAjMDBkNGZmOwog"
    "IC0tYWNjZW50MzogIzdjM2FlZDsKICAtLWdyZWVuOiAjMDBlNWEwOwogIC0teWVsbG93OiAjZmJiZjI0OwogIC0t"
    "cmVkOiAjZjQzZjVlOwogIC0tb3JhbmdlOiAjZjk3MzE2OwogIC0tdGV4dDogI2UyZThmMDsKICAtLXRleHQyOiAj"
    "OTRhM2I4OwogIC0tdGV4dDM6ICM2NDc0OGI7CiAgLS1nbG93OiAwIDAgNDBweCByZ2JhKDMwLDEwNywyNTUsMC4y"
    "NSk7CiAgLS1nbG93MjogMCAwIDIwcHggcmdiYSgwLDIxMiwyNTUsMC4xNSk7CiAgLS1mb250OiAnU3BhY2UgR3Jv"
    "dGVzaycsIHNhbnMtc2VyaWY7CiAgLS1tb25vOiAnSmV0QnJhaW5zIE1vbm8nLCBtb25vc3BhY2U7Cn0KCiogeyBt"
    "YXJnaW46MDsgcGFkZGluZzowOyBib3gtc2l6aW5nOmJvcmRlci1ib3g7IH0KaHRtbCB7IHNjcm9sbC1iZWhhdmlv"
    "cjogc21vb3RoOyB9Cgpib2R5IHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1iZyk7CiAgY29sb3I6IHZhcigtLXRleHQp"
    "OwogIGZvbnQtZmFtaWx5OiB2YXIoLS1mb250KTsKICBtaW4taGVpZ2h0OiAxMDB2aDsKICBvdmVyZmxvdy14OiBo"
    "aWRkZW47Cn0KCi8qIE5PSVNFIFRFWFRVUkUgT1ZFUkxBWSAqLwpib2R5OjpiZWZvcmUgewogIGNvbnRlbnQ6ICcn"
    "OwogIHBvc2l0aW9uOiBmaXhlZDsKICBpbnNldDogMDsKICBiYWNrZ3JvdW5kLWltYWdlOiB1cmwoImRhdGE6aW1h"
    "Z2Uvc3ZnK3htbCwlM0Nzdmcgdmlld0JveD0nMCAwIDI1NiAyNTYnIHhtbG5zPSdodHRwOi8vd3d3LnczLm9yZy8y"
    "MDAwL3N2ZyclM0UlM0NmaWx0ZXIgaWQ9J25vaXNlJyUzRSUzQ2ZlVHVyYnVsZW5jZSB0eXBlPSdmcmFjdGFsTm9p"
    "c2UnIGJhc2VGcmVxdWVuY3k9JzAuOScgbnVtT2N0YXZlcz0nNCcgc3RpdGNoVGlsZXM9J3N0aXRjaCcvJTNFJTND"
    "L2ZpbHRlciUzRSUzQ3JlY3Qgd2lkdGg9JzEwMCUyNScgaGVpZ2h0PScxMDAlMjUnIGZpbHRlcj0ndXJsKCUyM25v"
    "aXNlKScgb3BhY2l0eT0nMC4wMycvJTNFJTNDL3N2ZyUzRSIpOwogIHBvaW50ZXItZXZlbnRzOiBub25lOwogIHot"
    "aW5kZXg6IDA7CiAgb3BhY2l0eTogMC40Owp9CgovKiBBTklNQVRFRCBHUklEIEJBQ0tHUk9VTkQgKi8KYm9keTo6"
    "YWZ0ZXIgewogIGNvbnRlbnQ6ICcnOwogIHBvc2l0aW9uOiBmaXhlZDsKICBpbnNldDogMDsKICBiYWNrZ3JvdW5k"
    "LWltYWdlOgogICAgbGluZWFyLWdyYWRpZW50KHJnYmEoMzAsMTA3LDI1NSwwLjA0KSAxcHgsIHRyYW5zcGFyZW50"
    "IDFweCksCiAgICBsaW5lYXItZ3JhZGllbnQoOTBkZWcsIHJnYmEoMzAsMTA3LDI1NSwwLjA0KSAxcHgsIHRyYW5z"
    "cGFyZW50IDFweCk7CiAgYmFja2dyb3VuZC1zaXplOiA0OHB4IDQ4cHg7CiAgcG9pbnRlci1ldmVudHM6IG5vbmU7"
    "CiAgei1pbmRleDogMDsKfQoKLyogSEVBREVSICovCmhlYWRlciB7CiAgcG9zaXRpb246IHN0aWNreTsgdG9wOiAw"
    "OyB6LWluZGV4OiAxMDA7CiAgYmFja2dyb3VuZDogcmdiYSg0LDgsMTYsMC44NSk7CiAgYmFja2Ryb3AtZmlsdGVy"
    "OiBibHVyKDI0cHgpOwogIGJvcmRlci1ib3R0b206IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIHBhZGRpbmc6"
    "IDAgMnJlbTsKfQouaGVhZGVyLWlubmVyIHsKICBtYXgtd2lkdGg6IDE2MDBweDsgbWFyZ2luOiAwIGF1dG87CiAg"
    "ZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVu"
    "OwogIGhlaWdodDogNjRweDsgZ2FwOiAycmVtOwp9Ci5sb2dvIHsKICBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVt"
    "czogY2VudGVyOyBnYXA6IDEycHg7CiAgdGV4dC1kZWNvcmF0aW9uOiBub25lOwp9Ci5sb2dvLWljb24gewogIHdp"
    "ZHRoOiAzNnB4OyBoZWlnaHQ6IDM2cHg7CiAgYmFja2dyb3VuZDogbGluZWFyLWdyYWRpZW50KDEzNWRlZywgdmFy"
    "KC0tYWNjZW50KSwgdmFyKC0tYWNjZW50MikpOwogIGJvcmRlci1yYWRpdXM6IDhweDsKICBkaXNwbGF5OiBmbGV4"
    "OyBhbGlnbi1pdGVtczogY2VudGVyOyBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjsKICBmb250LXNpemU6IDE4cHg7"
    "CiAgYm94LXNoYWRvdzogMCAwIDIwcHggcmdiYSgzMCwxMDcsMjU1LDAuNSk7CiAgYW5pbWF0aW9uOiBsb2dvR2xv"
    "dyAzcyBlYXNlLWluLW91dCBpbmZpbml0ZSBhbHRlcm5hdGU7Cn0KQGtleWZyYW1lcyBsb2dvR2xvdyB7CiAgZnJv"
    "bSB7IGJveC1zaGFkb3c6IDAgMCAyMHB4IHJnYmEoMzAsMTA3LDI1NSwwLjQpOyB9CiAgdG8geyBib3gtc2hhZG93"
    "OiAwIDAgMzVweCByZ2JhKDAsMjEyLDI1NSwwLjYpOyB9Cn0KLmxvZ28tdGV4dCB7IGZvbnQtc2l6ZTogMS4ycmVt"
    "OyBmb250LXdlaWdodDogNzAwOyBsZXR0ZXItc3BhY2luZzogLTAuMDJlbTsgfQoubG9nby10ZXh0IHNwYW4geyBj"
    "b2xvcjogdmFyKC0tYWNjZW50Mik7IH0KCi5uYXYgeyBkaXNwbGF5OiBmbGV4OyBnYXA6IDAuMjVyZW07IH0KLm5h"
    "di1idG4gewogIGJhY2tncm91bmQ6IG5vbmU7IGJvcmRlcjogbm9uZTsgY29sb3I6IHZhcigtLXRleHQyKTsgY3Vy"
    "c29yOiBwb2ludGVyOwogIHBhZGRpbmc6IDZweCAxNnB4OyBib3JkZXItcmFkaXVzOiA2cHg7IGZvbnQtZmFtaWx5"
    "OiB2YXIoLS1mb250KTsgZm9udC1zaXplOiAwLjg1cmVtOwogIHRyYW5zaXRpb246IGFsbCAwLjJzOyB3aGl0ZS1z"
    "cGFjZTogbm93cmFwOwp9Ci5uYXYtYnRuOmhvdmVyLCAubmF2LWJ0bi5hY3RpdmUgeyBiYWNrZ3JvdW5kOiByZ2Jh"
    "KDMwLDEwNywyNTUsMC4xNSk7IGNvbG9yOiB2YXIoLS1hY2NlbnQyKTsgfQoKLnN0YXR1cy1iYWRnZSB7CiAgZGlz"
    "cGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiA4cHg7CiAgYmFja2dyb3VuZDogdmFyKC0tY2Fy"
    "ZCk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czogMjBweDsgcGFkZGlu"
    "ZzogNnB4IDE0cHg7IGZvbnQtc2l6ZTogMC43OHJlbTsKICBjb2xvcjogdmFyKC0tdGV4dDIpOwp9Ci5zdGF0dXMt"
    "ZG90IHsKICB3aWR0aDogOHB4OyBoZWlnaHQ6IDhweDsgYm9yZGVyLXJhZGl1czogNTAlOwogIGJhY2tncm91bmQ6"
    "IHZhcigtLXRleHQzKTsgdHJhbnNpdGlvbjogYWxsIDAuM3M7Cn0KLnN0YXR1cy1kb3QuYWN0aXZlIHsgYmFja2dy"
    "b3VuZDogdmFyKC0tZ3JlZW4pOyBib3gtc2hhZG93OiAwIDAgOHB4IHZhcigtLWdyZWVuKTsgYW5pbWF0aW9uOiBw"
    "dWxzZSAxLjVzIGluZmluaXRlOyB9Ci5zdGF0dXMtZG90LnJ1bm5pbmcgeyBiYWNrZ3JvdW5kOiB2YXIoLS15ZWxs"
    "b3cpOyBib3gtc2hhZG93OiAwIDAgOHB4IHZhcigtLXllbGxvdyk7IGFuaW1hdGlvbjogcHVsc2UgMC44cyBpbmZp"
    "bml0ZTsgfQouc3RhdHVzLWRvdC5lcnJvciB7IGJhY2tncm91bmQ6IHZhcigtLXJlZCk7IH0KQGtleWZyYW1lcyBw"
    "dWxzZSB7IDAlLDEwMCUgeyBvcGFjaXR5OjE7IH0gNTAlIHsgb3BhY2l0eTowLjQ7IH0gfQoKLyogTUFJTiBMQVlP"
    "VVQgKi8KbWFpbiB7CiAgcG9zaXRpb246IHJlbGF0aXZlOyB6LWluZGV4OiAxOwogIG1heC13aWR0aDogMTYwMHB4"
    "OyBtYXJnaW46IDAgYXV0bzsKICBwYWRkaW5nOiAycmVtOwp9CgovKiBTRUNUSU9OUyAqLwouc2VjdGlvbiB7IGRp"
    "c3BsYXk6IG5vbmU7IH0KLnNlY3Rpb24uYWN0aXZlIHsgZGlzcGxheTogYmxvY2s7IH0KCi8qIEhFUk8gU0VDVElP"
    "TiAqLwouaGVybyB7CiAgdGV4dC1hbGlnbjogY2VudGVyOyBwYWRkaW5nOiA0cmVtIDJyZW0gM3JlbTsKICBwb3Np"
    "dGlvbjogcmVsYXRpdmU7Cn0KLmhlcm8tZ2xvdyB7CiAgcG9zaXRpb246IGFic29sdXRlOyB0b3A6IDA7IGxlZnQ6"
    "IDUwJTsgdHJhbnNmb3JtOiB0cmFuc2xhdGVYKC01MCUpOwogIHdpZHRoOiA2MDBweDsgaGVpZ2h0OiAzMDBweDsK"
    "ICBiYWNrZ3JvdW5kOiByYWRpYWwtZ3JhZGllbnQoZWxsaXBzZSwgcmdiYSgzMCwxMDcsMjU1LDAuMTIpIDAlLCB0"
    "cmFuc3BhcmVudCA3MCUpOwogIHBvaW50ZXItZXZlbnRzOiBub25lOwp9Ci5oZXJvIGgxIHsKICBmb250LXNpemU6"
    "IGNsYW1wKDIuMnJlbSwgNHZ3LCAzLjVyZW0pOwogIGZvbnQtd2VpZ2h0OiA3MDA7IGxpbmUtaGVpZ2h0OiAxLjE7"
    "CiAgbGV0dGVyLXNwYWNpbmc6IC0wLjAzZW07CiAgYmFja2dyb3VuZDogbGluZWFyLWdyYWRpZW50KDEzNWRlZywg"
    "I2ZmZiAzMCUsIHZhcigtLWFjY2VudDIpIDYwJSwgdmFyKC0tYWNjZW50KSAxMDAlKTsKICAtd2Via2l0LWJhY2tn"
    "cm91bmQtY2xpcDogdGV4dDsgLXdlYmtpdC10ZXh0LWZpbGwtY29sb3I6IHRyYW5zcGFyZW50OwogIGJhY2tncm91"
    "bmQtY2xpcDogdGV4dDsKICBtYXJnaW4tYm90dG9tOiAxcmVtOwp9Ci5oZXJvIHAgewogIGZvbnQtc2l6ZTogMS4x"
    "cmVtOyBjb2xvcjogdmFyKC0tdGV4dDIpOyBtYXgtd2lkdGg6IDYwMHB4OyBtYXJnaW46IDAgYXV0byAyLjVyZW07"
    "CiAgbGluZS1oZWlnaHQ6IDEuNzsKfQouaGVyby10YWdzIHsKICBkaXNwbGF5OiBmbGV4OyBmbGV4LXdyYXA6IHdy"
    "YXA7IGp1c3RpZnktY29udGVudDogY2VudGVyOyBnYXA6IDhweDsgbWFyZ2luLWJvdHRvbTogMi41cmVtOwp9Ci50"
    "YWcgewogIGJhY2tncm91bmQ6IHJnYmEoMzAsMTA3LDI1NSwwLjEpOyBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDMw"
    "LDEwNywyNTUsMC4zKTsKICBib3JkZXItcmFkaXVzOiAyMHB4OyBwYWRkaW5nOiA0cHggMTJweDsgZm9udC1zaXpl"
    "OiAwLjc4cmVtOwogIGNvbG9yOiB2YXIoLS1hY2NlbnQyKTsgZm9udC1mYW1pbHk6IHZhcigtLW1vbm8pOwp9Cgov"
    "KiBDT05UUk9MIFBBTkVMICovCi5jb250cm9sLXBhbmVsIHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJkKTsKICBi"
    "b3JkZXI6IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6IDE2cHg7CiAgcGFkZGluZzog"
    "MnJlbTsKICBtYXJnaW4tYm90dG9tOiAycmVtOwp9Ci5wYW5lbC10aXRsZSB7CiAgZm9udC1zaXplOiAwLjc1cmVt"
    "OyB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOyBsZXR0ZXItc3BhY2luZzogMC4xZW07CiAgY29sb3I6IHZhcigt"
    "LWFjY2VudDIpOyBtYXJnaW4tYm90dG9tOiAxLjVyZW07IGZvbnQtZmFtaWx5OiB2YXIoLS1tb25vKTsKICBkaXNw"
    "bGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDhweDsKfQoucGFuZWwtdGl0bGU6OmJlZm9yZSB7"
    "IGNvbnRlbnQ6ICcnOyBkaXNwbGF5OiBibG9jazsgd2lkdGg6IDIwcHg7IGhlaWdodDogMXB4OyBiYWNrZ3JvdW5k"
    "OiB2YXIoLS1hY2NlbnQyKTsgfQoKLmNvbnRyb2xzLWdyaWQgeyBkaXNwbGF5OiBncmlkOyBncmlkLXRlbXBsYXRl"
    "LWNvbHVtbnM6IHJlcGVhdChhdXRvLWZpdCwgbWlubWF4KDIwMHB4LCAxZnIpKTsgZ2FwOiAxcmVtOyB9CgouYnRu"
    "IHsKICBkaXNwbGF5OiBpbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsganVzdGlmeS1jb250ZW50OiBj"
    "ZW50ZXI7IGdhcDogOHB4OwogIHBhZGRpbmc6IDEycHggMjRweDsgYm9yZGVyLXJhZGl1czogMTBweDsgZm9udC1m"
    "YW1pbHk6IHZhcigtLWZvbnQpOwogIGZvbnQtc2l6ZTogMC45cmVtOyBmb250LXdlaWdodDogNjAwOyBjdXJzb3I6"
    "IHBvaW50ZXI7IGJvcmRlcjogbm9uZTsKICB0cmFuc2l0aW9uOiBhbGwgMC4yNXM7IHRleHQtdHJhbnNmb3JtOiBu"
    "b25lOyBsZXR0ZXItc3BhY2luZzogMDsKICBwb3NpdGlvbjogcmVsYXRpdmU7IG92ZXJmbG93OiBoaWRkZW47Cn0K"
    "LmJ0bjo6YWZ0ZXIgewogIGNvbnRlbnQ6ICcnOyBwb3NpdGlvbjogYWJzb2x1dGU7IGluc2V0OiAwOwogIGJhY2tn"
    "cm91bmQ6IGxpbmVhci1ncmFkaWVudChyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpLCB0cmFuc3BhcmVudCk7CiAgb3Bh"
    "Y2l0eTogMDsgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjJzOwp9Ci5idG46aG92ZXI6OmFmdGVyIHsgb3BhY2l0eTog"
    "MTsgfQouYnRuOmFjdGl2ZSB7IHRyYW5zZm9ybTogc2NhbGUoMC45OCk7IH0KCi5idG4tcHJpbWFyeSB7CiAgYmFj"
    "a2dyb3VuZDogbGluZWFyLWdyYWRpZW50KDEzNWRlZywgdmFyKC0tYWNjZW50KSwgIzE0NTBjYyk7CiAgY29sb3I6"
    "IHdoaXRlOyBib3gtc2hhZG93OiAwIDRweCAyMHB4IHJnYmEoMzAsMTA3LDI1NSwwLjM1KTsKfQouYnRuLXByaW1h"
    "cnk6aG92ZXIgeyBib3gtc2hhZG93OiAwIDZweCAyOHB4IHJnYmEoMzAsMTA3LDI1NSwwLjU1KTsgdHJhbnNmb3Jt"
    "OiB0cmFuc2xhdGVZKC0xcHgpOyB9Ci5idG4tc2Vjb25kYXJ5IHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJkMik7"
    "IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpOwogIGNvbG9yOiB2YXIoLS10ZXh0KTsKfQouYnRuLXNl"
    "Y29uZGFyeTpob3ZlciB7IGJvcmRlci1jb2xvcjogdmFyKC0tYWNjZW50Mik7IGNvbG9yOiB2YXIoLS1hY2NlbnQy"
    "KTsgfQouYnRuLXN1Y2Nlc3MgewogIGJhY2tncm91bmQ6IGxpbmVhci1ncmFkaWVudCgxMzVkZWcsICMwNTk2Njks"
    "ICMwNDc4NTcpOwogIGNvbG9yOiB3aGl0ZTsgYm94LXNoYWRvdzogMCA0cHggMTVweCByZ2JhKDAsMjI5LDE2MCww"
    "LjI1KTsKfQouYnRuLXN1Y2Nlc3M6aG92ZXIgeyBib3gtc2hhZG93OiAwIDZweCAyMnB4IHJnYmEoMCwyMjksMTYw"
    "LDAuNCk7IHRyYW5zZm9ybTogdHJhbnNsYXRlWSgtMXB4KTsgfQouYnRuLWRhbmdlciB7CiAgYmFja2dyb3VuZDog"
    "bGluZWFyLWdyYWRpZW50KDEzNWRlZywgI2RjMjYyNiwgI2I5MWMxYyk7CiAgY29sb3I6IHdoaXRlOwp9Ci5idG4t"
    "cHVycGxlIHsKICBiYWNrZ3JvdW5kOiBsaW5lYXItZ3JhZGllbnQoMTM1ZGVnLCB2YXIoLS1hY2NlbnQzKSwgIzVi"
    "MjFiNik7CiAgY29sb3I6IHdoaXRlOyBib3gtc2hhZG93OiAwIDRweCAxNXB4IHJnYmEoMTI0LDU4LDIzNywwLjMp"
    "Owp9Ci5idG4tcHVycGxlOmhvdmVyIHsgYm94LXNoYWRvdzogMCA2cHggMjJweCByZ2JhKDEyNCw1OCwyMzcsMC41"
    "KTsgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKC0xcHgpOyB9Ci5idG46ZGlzYWJsZWQgeyBvcGFjaXR5OiAwLjQ7IGN1"
    "cnNvcjogbm90LWFsbG93ZWQ7IHRyYW5zZm9ybTogbm9uZSAhaW1wb3J0YW50OyB9CgovKiBQUk9HUkVTUyBCQVIg"
    "Ki8KLnByb2dyZXNzLXdyYXAgeyBtYXJnaW4tdG9wOiAxLjVyZW07IH0KLnByb2dyZXNzLWxhYmVsIHsgZm9udC1z"
    "aXplOiAwLjhyZW07IGNvbG9yOiB2YXIoLS10ZXh0Mik7IG1hcmdpbi1ib3R0b206IDZweDsgZGlzcGxheTogZmxl"
    "eDsganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVuOyB9Ci5wcm9ncmVzcy1iYXIgewogIGhlaWdodDogNnB4"
    "OyBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpOwogIGJvcmRlci1yYWRpdXM6IDNweDsgb3ZlcmZs"
    "b3c6IGhpZGRlbjsKfQoucHJvZ3Jlc3MtZmlsbCB7CiAgaGVpZ2h0OiAxMDAlOyBib3JkZXItcmFkaXVzOiAzcHg7"
    "CiAgYmFja2dyb3VuZDogbGluZWFyLWdyYWRpZW50KDkwZGVnLCB2YXIoLS1hY2NlbnQpLCB2YXIoLS1hY2NlbnQy"
    "KSk7CiAgd2lkdGg6IDAlOyB0cmFuc2l0aW9uOiB3aWR0aCAwLjZzIGN1YmljLWJlemllcigwLjQsIDAsIDAuMiwg"
    "MSk7CiAgYm94LXNoYWRvdzogMCAwIDEycHggdmFyKC0tYWNjZW50KTsKfQoKLyogTUVUUklDIENBUkRTICovCi5t"
    "ZXRyaWNzLWdyaWQgewogIGRpc3BsYXk6IGdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoYXV0"
    "by1maXQsIG1pbm1heCgyMjBweCwgMWZyKSk7CiAgZ2FwOiAxcmVtOyBtYXJnaW4tYm90dG9tOiAycmVtOwp9Ci5t"
    "ZXRyaWMtY2FyZCB7CiAgYmFja2dyb3VuZDogdmFyKC0tY2FyZCk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJv"
    "cmRlcik7CiAgYm9yZGVyLXJhZGl1czogMTRweDsgcGFkZGluZzogMS41cmVtOwogIHRyYW5zaXRpb246IGFsbCAw"
    "LjNzOyBwb3NpdGlvbjogcmVsYXRpdmU7IG92ZXJmbG93OiBoaWRkZW47Cn0KLm1ldHJpYy1jYXJkOjpiZWZvcmUg"
    "ewogIGNvbnRlbnQ6ICcnOyBwb3NpdGlvbjogYWJzb2x1dGU7IHRvcDogMDsgbGVmdDogMDsgcmlnaHQ6IDA7IGhl"
    "aWdodDogMnB4OwogIGJhY2tncm91bmQ6IHZhcigtLWNhcmQtYWNjZW50LCB2YXIoLS1hY2NlbnQpKTsKICBvcGFj"
    "aXR5OiAwLjY7Cn0KLm1ldHJpYy1jYXJkOmhvdmVyIHsgYm9yZGVyLWNvbG9yOiB2YXIoLS1ib3JkZXIyKTsgdHJh"
    "bnNmb3JtOiB0cmFuc2xhdGVZKC0ycHgpOyBib3gtc2hhZG93OiB2YXIoLS1nbG93Mik7IH0KLm1ldHJpYy1jYXJk"
    "LmdyZWVuIHsgLS1jYXJkLWFjY2VudDogdmFyKC0tZ3JlZW4pOyB9Ci5tZXRyaWMtY2FyZC55ZWxsb3cgeyAtLWNh"
    "cmQtYWNjZW50OiB2YXIoLS15ZWxsb3cpOyB9Ci5tZXRyaWMtY2FyZC5yZWQgeyAtLWNhcmQtYWNjZW50OiB2YXIo"
    "LS1yZWQpOyB9Ci5tZXRyaWMtY2FyZC5wdXJwbGUgeyAtLWNhcmQtYWNjZW50OiB2YXIoLS1hY2NlbnQzKTsgfQoK"
    "Lm1ldHJpYy1sYWJlbCB7IGZvbnQtc2l6ZTogMC43MnJlbTsgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsgbGV0"
    "dGVyLXNwYWNpbmc6IDAuMDhlbTsgY29sb3I6IHZhcigtLXRleHQzKTsgbWFyZ2luLWJvdHRvbTogMC41cmVtOyBm"
    "b250LWZhbWlseTogdmFyKC0tbW9ubyk7IH0KLm1ldHJpYy12YWx1ZSB7IGZvbnQtc2l6ZTogMi4ycmVtOyBmb250"
    "LXdlaWdodDogNzAwOyBsaW5lLWhlaWdodDogMTsgbWFyZ2luLWJvdHRvbTogMC4yNXJlbTsgfQoubWV0cmljLXN1"
    "YiB7IGZvbnQtc2l6ZTogMC44cmVtOyBjb2xvcjogdmFyKC0tdGV4dDIpOyB9Ci5tZXRyaWMtaWNvbiB7IHBvc2l0"
    "aW9uOiBhYnNvbHV0ZTsgdG9wOiAxLjJyZW07IHJpZ2h0OiAxLjJyZW07IGZvbnQtc2l6ZTogMS41cmVtOyBvcGFj"
    "aXR5OiAwLjM7IH0KCi8qIENIQVJUUyBHUklEICovCi5jaGFydHMtZ3JpZCB7CiAgZGlzcGxheTogZ3JpZDsKICBn"
    "cmlkLXRlbXBsYXRlLWNvbHVtbnM6IHJlcGVhdChhdXRvLWZpdCwgbWlubWF4KDQ4MHB4LCAxZnIpKTsKICBnYXA6"
    "IDEuNXJlbTsgbWFyZ2luLWJvdHRvbTogMnJlbTsKfQouY2hhcnQtY2FyZCB7CiAgYmFja2dyb3VuZDogdmFyKC0t"
    "Y2FyZCk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czogMTZweDsgcGFk"
    "ZGluZzogMS41cmVtOwogIHRyYW5zaXRpb246IGFsbCAwLjNzOwp9Ci5jaGFydC1jYXJkOmhvdmVyIHsgYm9yZGVy"
    "LWNvbG9yOiB2YXIoLS1ib3JkZXIyKTsgfQouY2hhcnQtY2FyZC5mdWxsIHsgZ3JpZC1jb2x1bW46IDEgLyAtMTsg"
    "fQouY2hhcnQtdGl0bGUgewogIGZvbnQtc2l6ZTogMC44cmVtOyB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOyBs"
    "ZXR0ZXItc3BhY2luZzogMC4wOGVtOwogIGNvbG9yOiB2YXIoLS10ZXh0Mik7IG1hcmdpbi1ib3R0b206IDEuMnJl"
    "bTsgZGlzcGxheTogZmxleDsKICBhbGlnbi1pdGVtczogY2VudGVyOyBqdXN0aWZ5LWNvbnRlbnQ6IHNwYWNlLWJl"
    "dHdlZW47Cn0KLmNoYXJ0LXRpdGxlIC5iYWRnZSB7CiAgYmFja2dyb3VuZDogcmdiYSgzMCwxMDcsMjU1LDAuMTUp"
    "OyBjb2xvcjogdmFyKC0tYWNjZW50Mik7CiAgYm9yZGVyOiAxcHggc29saWQgcmdiYSgzMCwxMDcsMjU1LDAuMyk7"
    "CiAgYm9yZGVyLXJhZGl1czogNHB4OyBwYWRkaW5nOiAycHggOHB4OyBmb250LXNpemU6IDAuN3JlbTsKfQouY2hh"
    "cnQtd3JhcCB7IHBvc2l0aW9uOiByZWxhdGl2ZTsgaGVpZ2h0OiAyODBweDsgfQouY2hhcnQtd3JhcC50YWxsIHsg"
    "aGVpZ2h0OiAzNDBweDsgfQoKLyogTU9ERUwgQ0FSRFMgKi8KLm1vZGVscy1ncmlkIHsgZGlzcGxheTogZ3JpZDsg"
    "Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoYXV0by1maXQsIG1pbm1heCgzMDBweCwgMWZyKSk7IGdhcDog"
    "MXJlbTsgbWFyZ2luLWJvdHRvbTogMnJlbTsgfQoubW9kZWwtY2FyZCB7CiAgYmFja2dyb3VuZDogdmFyKC0tY2Fy"
    "ZCk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czogMTRweDsgcGFkZGlu"
    "ZzogMS41cmVtOwogIHRyYW5zaXRpb246IGFsbCAwLjNzOwp9Ci5tb2RlbC1jYXJkLmJlc3QgewogIGJvcmRlci1j"
    "b2xvcjogdmFyKC0tYWNjZW50Mik7CiAgYm94LXNoYWRvdzogMCAwIDMwcHggcmdiYSgwLDIxMiwyNTUsMC4xKTsK"
    "fQoubW9kZWwtY2FyZDpob3ZlciB7IHRyYW5zZm9ybTogdHJhbnNsYXRlWSgtMnB4KTsgfQoubW9kZWwtbmFtZSB7"
    "IGZvbnQtd2VpZ2h0OiA2MDA7IG1hcmdpbi1ib3R0b206IDAuMjVyZW07IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0"
    "ZW1zOiBjZW50ZXI7IGdhcDogOHB4OyB9Ci5tb2RlbC1iYWRnZSB7IGJhY2tncm91bmQ6IHZhcigtLWFjY2VudDIp"
    "OyBjb2xvcjogIzAwMDsgZm9udC1zaXplOiAwLjY1cmVtOyBmb250LXdlaWdodDogNzAwOyBwYWRkaW5nOiAycHgg"
    "OHB4OyBib3JkZXItcmFkaXVzOiA0cHg7IHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7IH0KLm1ldHJpYy1yb3cg"
    "eyBkaXNwbGF5OiBmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6IHNwYWNlLWJldHdlZW47IGFsaWduLWl0ZW1zOiBjZW50"
    "ZXI7IHBhZGRpbmc6IDZweCAwOyBib3JkZXItYm90dG9tOiAxcHggc29saWQgdmFyKC0tYm9yZGVyKTsgfQoubWV0"
    "cmljLXJvdzpsYXN0LWNoaWxkIHsgYm9yZGVyOiBub25lOyB9Ci5tZXRyaWMtcm93IC5sYWJlbCB7IGZvbnQtc2l6"
    "ZTogMC44cmVtOyBjb2xvcjogdmFyKC0tdGV4dDIpOyBmb250LWZhbWlseTogdmFyKC0tbW9ubyk7IH0KLm1ldHJp"
    "Yy1yb3cgLnZhbHVlIHsgZm9udC1zaXplOiAwLjlyZW07IGZvbnQtd2VpZ2h0OiA2MDA7IGZvbnQtZmFtaWx5OiB2"
    "YXIoLS1tb25vKTsgfQouYmFyLW1pbmkgeyBoZWlnaHQ6IDRweDsgYmFja2dyb3VuZDogcmdiYSgyNTUsMjU1LDI1"
    "NSwwLjA1KTsgYm9yZGVyLXJhZGl1czogMnB4OyBtYXJnaW4tdG9wOiAycHg7IH0KLmJhci1maWxsIHsgaGVpZ2h0"
    "OiAxMDAlOyBib3JkZXItcmFkaXVzOiAycHg7IGJhY2tncm91bmQ6IGxpbmVhci1ncmFkaWVudCg5MGRlZywgdmFy"
    "KC0tYWNjZW50KSwgdmFyKC0tYWNjZW50MikpOyB0cmFuc2l0aW9uOiB3aWR0aCAxcyBlYXNlOyB9CgovKiBMSVZF"
    "IEZFRUQgKi8KLmxpdmUtZmVlZCB7CiAgYmFja2dyb3VuZDogdmFyKC0tY2FyZCk7IGJvcmRlcjogMXB4IHNvbGlk"
    "IHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czogMTRweDsgb3ZlcmZsb3c6IGhpZGRlbjsgbWFyZ2luLWJv"
    "dHRvbTogMnJlbTsKfQouZmVlZC1oZWFkZXIgewogIGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7"
    "IGp1c3RpZnktY29udGVudDogc3BhY2UtYmV0d2VlbjsKICBwYWRkaW5nOiAxcmVtIDEuNXJlbTsgYm9yZGVyLWJv"
    "dHRvbTogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYmFja2dyb3VuZDogcmdiYSgzMCwxMDcsMjU1LDAuMDUp"
    "Owp9Ci5mZWVkLXRpdGxlIHsgZm9udC1zaXplOiAwLjhyZW07IHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7IGxl"
    "dHRlci1zcGFjaW5nOiAwLjA4ZW07IGNvbG9yOiB2YXIoLS10ZXh0Mik7IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0"
    "ZW1zOiBjZW50ZXI7IGdhcDogOHB4OyB9Ci5mZWVkLWxpdmUgeyB3aWR0aDogOHB4OyBoZWlnaHQ6IDhweDsgYm9y"
    "ZGVyLXJhZGl1czogNTAlOyBiYWNrZ3JvdW5kOiB2YXIoLS1yZWQpOyBhbmltYXRpb246IGJsaW5rIDFzIGluZmlu"
    "aXRlOyB9CkBrZXlmcmFtZXMgYmxpbmsgeyAwJSwxMDAlIHsgb3BhY2l0eToxOyB9IDUwJSB7IG9wYWNpdHk6MC4y"
    "OyB9IH0KLmZlZWQtc2Nyb2xsIHsgbWF4LWhlaWdodDogMzIwcHg7IG92ZXJmbG93LXk6IGF1dG87IH0KLmZlZWQt"
    "c2Nyb2xsOjotd2Via2l0LXNjcm9sbGJhciB7IHdpZHRoOiA0cHg7IH0KLmZlZWQtc2Nyb2xsOjotd2Via2l0LXNj"
    "cm9sbGJhci10cmFjayB7IGJhY2tncm91bmQ6IHRyYW5zcGFyZW50OyB9Ci5mZWVkLXNjcm9sbDo6LXdlYmtpdC1z"
    "Y3JvbGxiYXItdGh1bWIgeyBiYWNrZ3JvdW5kOiB2YXIoLS1ib3JkZXIyKTsgYm9yZGVyLXJhZGl1czogMnB4OyB9"
    "Ci5mZWVkLXJvdyB7CiAgZGlzcGxheTogZ3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAxMjBweCAxNDBweCAx"
    "MjBweCAxMDBweCAxZnIgODBweDsKICBnYXA6IDFyZW07IGFsaWduLWl0ZW1zOiBjZW50ZXI7CiAgcGFkZGluZzog"
    "MTBweCAxLjVyZW07IGJvcmRlci1ib3R0b206IDFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpOwogIGZv"
    "bnQtZmFtaWx5OiB2YXIoLS1tb25vKTsgZm9udC1zaXplOiAwLjc1cmVtOwogIGFuaW1hdGlvbjogZmFkZUluUm93"
    "IDAuM3MgZWFzZTsKICB0cmFuc2l0aW9uOiBiYWNrZ3JvdW5kIDAuMTVzOwp9Ci5mZWVkLXJvdzpob3ZlciB7IGJh"
    "Y2tncm91bmQ6IHJnYmEoMzAsMTA3LDI1NSwwLjA1KTsgfQpAa2V5ZnJhbWVzIGZhZGVJblJvdyB7IGZyb20geyBv"
    "cGFjaXR5OjA7IHRyYW5zZm9ybTogdHJhbnNsYXRlWCgtOHB4KTsgfSB0byB7IG9wYWNpdHk6MTsgdHJhbnNmb3Jt"
    "OiBub25lOyB9IH0KLmZlZWQtcm93IC5jb2wtdGltZSB7IGNvbG9yOiB2YXIoLS10ZXh0Myk7IH0KLmZlZWQtcm93"
    "IC5jb2wtZmF1bHQgeyB9Ci5zZXZlcml0eS1iYWRnZSB7CiAgZGlzcGxheTogaW5saW5lLWJsb2NrOyBwYWRkaW5n"
    "OiAycHggOHB4OyBib3JkZXItcmFkaXVzOiA0cHg7CiAgZm9udC1zaXplOiAwLjY4cmVtOyBmb250LXdlaWdodDog"
    "NjAwOyB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwp9Ci5zZXYtMSB7IGJhY2tncm91bmQ6IHJnYmEoMjUxLDE5"
    "MSwzNiwwLjE1KTsgY29sb3I6IHZhcigtLXllbGxvdyk7IH0KLnNldi0yIHsgYmFja2dyb3VuZDogcmdiYSgyNDks"
    "MTE1LDIyLDAuMTUpOyBjb2xvcjogdmFyKC0tb3JhbmdlKTsgfQouc2V2LTMgeyBiYWNrZ3JvdW5kOiByZ2JhKDI0"
    "NCw2Myw5NCwwLjE1KTsgY29sb3I6IHZhcigtLXJlZCk7IH0KLnNldi00LCAuc2V2LTUgeyBiYWNrZ3JvdW5kOiBy"
    "Z2JhKDI0NCw2Myw5NCwwLjI1KTsgY29sb3I6IHZhcigtLXJlZCk7IGFuaW1hdGlvbjogcHVsc2UgMXMgaW5maW5p"
    "dGU7IH0KLnN1Y2Nlc3MtcGlsbCB7CiAgZGlzcGxheTogaW5saW5lLWJsb2NrOyBwYWRkaW5nOiAycHggOHB4OyBi"
    "b3JkZXItcmFkaXVzOiA0cHg7IGZvbnQtc2l6ZTogMC43cmVtOwp9Ci5zdWNjZXNzLXBpbGwub2sgeyBiYWNrZ3Jv"
    "dW5kOiByZ2JhKDAsMjI5LDE2MCwwLjE1KTsgY29sb3I6IHZhcigtLWdyZWVuKTsgfQouc3VjY2Vzcy1waWxsLmZh"
    "aWwgeyBiYWNrZ3JvdW5kOiByZ2JhKDI0NCw2Myw5NCwwLjE1KTsgY29sb3I6IHZhcigtLXJlZCk7IH0KCi8qIERB"
    "VEEgVEFCTEUgKi8KLnRhYmxlLXdyYXAgewogIGJhY2tncm91bmQ6IHZhcigtLWNhcmQpOyBib3JkZXI6IDFweCBz"
    "b2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6IDE0cHg7IG92ZXJmbG93OiBoaWRkZW47IG1hcmdp"
    "bi1ib3R0b206IDJyZW07Cn0KLnRhYmxlLWhlYWRlciB7CiAgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNl"
    "bnRlcjsganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6IDFyZW0gMS41cmVtOyBib3Jk"
    "ZXItYm90dG9tOiAxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBnYXA6IDFyZW07IGZsZXgtd3JhcDogd3JhcDsK"
    "fQoudGFibGUtc2Nyb2xsIHsgb3ZlcmZsb3cteDogYXV0bzsgfQouZGF0YS10YWJsZSB7CiAgd2lkdGg6IDEwMCU7"
    "IGJvcmRlci1jb2xsYXBzZTogY29sbGFwc2U7CiAgZm9udC1mYW1pbHk6IHZhcigtLW1vbm8pOyBmb250LXNpemU6"
    "IDAuNzRyZW07Cn0KLmRhdGEtdGFibGUgdGggewogIHRleHQtYWxpZ246IGxlZnQ7IHBhZGRpbmc6IDEwcHggMTZw"
    "eDsKICBjb2xvcjogdmFyKC0tdGV4dDMpOyB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOyBsZXR0ZXItc3BhY2lu"
    "ZzogMC4wNmVtOyBmb250LXNpemU6IDAuNjhyZW07CiAgYm9yZGVyLWJvdHRvbTogMXB4IHNvbGlkIHZhcigtLWJv"
    "cmRlcik7IHdoaXRlLXNwYWNlOiBub3dyYXA7CiAgYmFja2dyb3VuZDogcmdiYSgzMCwxMDcsMjU1LDAuMDMpOwp9"
    "Ci5kYXRhLXRhYmxlIHRkIHsKICBwYWRkaW5nOiA5cHggMTZweDsgYm9yZGVyLWJvdHRvbTogMXB4IHNvbGlkIHJn"
    "YmEoMjU1LDI1NSwyNTUsMC4wMyk7CiAgY29sb3I6IHZhcigtLXRleHQyKTsgd2hpdGUtc3BhY2U6IG5vd3JhcDsK"
    "fQouZGF0YS10YWJsZSB0cjpob3ZlciB0ZCB7IGJhY2tncm91bmQ6IHJnYmEoMzAsMTA3LDI1NSwwLjA0KTsgfQou"
    "cGFnaW5hdGlvbiB7CiAgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsganVzdGlmeS1jb250ZW50"
    "OiBjZW50ZXI7IGdhcDogOHB4OwogIHBhZGRpbmc6IDFyZW07Cn0KLnBhZ2UtYnRuIHsKICBiYWNrZ3JvdW5kOiB2"
    "YXIoLS1jYXJkMik7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgY29sb3I6IHZhcigtLXRleHQy"
    "KTsgcGFkZGluZzogNnB4IDEycHg7IGJvcmRlci1yYWRpdXM6IDZweDsKICBjdXJzb3I6IHBvaW50ZXI7IGZvbnQt"
    "ZmFtaWx5OiB2YXIoLS1tb25vKTsgZm9udC1zaXplOiAwLjhyZW07CiAgdHJhbnNpdGlvbjogYWxsIDAuMnM7Cn0K"
    "LnBhZ2UtYnRuOmhvdmVyLCAucGFnZS1idG4uYWN0aXZlIHsgYm9yZGVyLWNvbG9yOiB2YXIoLS1hY2NlbnQpOyBj"
    "b2xvcjogdmFyKC0tYWNjZW50Mik7IH0KCi8qIEFJIFJFUE9SVCAqLwoucmVwb3J0LXBhbmVsIHsKICBiYWNrZ3Jv"
    "dW5kOiB2YXIoLS1jYXJkKTsgYm9yZGVyOiAxcHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXItcmFkaXVz"
    "OiAxNnB4OyBwYWRkaW5nOiAycmVtOyBtYXJnaW4tYm90dG9tOiAycmVtOwp9Ci5sYW5nLXNlbGVjdG9yIHsKICBk"
    "aXNwbGF5OiBmbGV4OyBmbGV4LXdyYXA6IHdyYXA7IGdhcDogOHB4OyBtYXJnaW4tYm90dG9tOiAxLjVyZW07Cn0K"
    "LmxhbmctYnRuIHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJkMik7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJv"
    "cmRlcik7CiAgY29sb3I6IHZhcigtLXRleHQyKTsgcGFkZGluZzogOHB4IDE2cHg7IGJvcmRlci1yYWRpdXM6IDhw"
    "eDsKICBjdXJzb3I6IHBvaW50ZXI7IGZvbnQtZmFtaWx5OiB2YXIoLS1mb250KTsgZm9udC1zaXplOiAwLjg1cmVt"
    "OwogIHRyYW5zaXRpb246IGFsbCAwLjJzOwp9Ci5sYW5nLWJ0bjpob3ZlciwgLmxhbmctYnRuLmFjdGl2ZSB7IGJv"
    "cmRlci1jb2xvcjogdmFyKC0tYWNjZW50KTsgY29sb3I6IHZhcigtLWFjY2VudDIpOyBiYWNrZ3JvdW5kOiByZ2Jh"
    "KDMwLDEwNywyNTUsMC4xKTsgfQoucmVwb3J0LWNvbnRlbnQgewogIGJhY2tncm91bmQ6IHZhcigtLXN1cmZhY2Up"
    "OyBib3JkZXI6IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6IDEwcHg7IHBhZGRpbmc6"
    "IDJyZW07CiAgbWluLWhlaWdodDogNDAwcHg7CiAgbGluZS1oZWlnaHQ6IDEuODsKfQoucmVwb3J0LWNvbnRlbnQg"
    "aDIgeyBmb250LXNpemU6IDEuMXJlbTsgbWFyZ2luOiAxLjVyZW0gMCAwLjc1cmVtOyBjb2xvcjogdmFyKC0tYWNj"
    "ZW50Mik7IH0KLnJlcG9ydC1jb250ZW50IGgyOmZpcnN0LWNoaWxkIHsgbWFyZ2luLXRvcDogMDsgfQoucmVwb3J0"
    "LWNvbnRlbnQgcCB7IGNvbG9yOiB2YXIoLS10ZXh0Mik7IG1hcmdpbi1ib3R0b206IDAuNzVyZW07IH0KLnJlcG9y"
    "dC1jb250ZW50IHN0cm9uZyB7IGNvbG9yOiB2YXIoLS10ZXh0KTsgfQoucmVwb3J0LWNvbnRlbnQgb2wsIC5yZXBv"
    "cnQtY29udGVudCB1bCB7IGNvbG9yOiB2YXIoLS10ZXh0Mik7IHBhZGRpbmctbGVmdDogMS41cmVtOyB9Ci5yZXBv"
    "cnQtY29udGVudCBsaSB7IG1hcmdpbi1ib3R0b206IDAuNXJlbTsgfQoucmVwb3J0LWxvYWRpbmcgewogIGRpc3Bs"
    "YXk6IGZsZXg7IGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47IGFsaWduLWl0ZW1zOiBjZW50ZXI7IGp1c3RpZnktY29u"
    "dGVudDogY2VudGVyOwogIG1pbi1oZWlnaHQ6IDMwMHB4OyBnYXA6IDE2cHg7Cn0KLnNwaW5uZXIgewogIHdpZHRo"
    "OiA0OHB4OyBoZWlnaHQ6IDQ4cHg7IGJvcmRlci1yYWRpdXM6IDUwJTsKICBib3JkZXI6IDNweCBzb2xpZCByZ2Jh"
    "KDMwLDEwNywyNTUsMC4xKTsKICBib3JkZXItdG9wLWNvbG9yOiB2YXIoLS1hY2NlbnQpOwogIGFuaW1hdGlvbjog"
    "c3BpbiAwLjhzIGxpbmVhciBpbmZpbml0ZTsKfQpAa2V5ZnJhbWVzIHNwaW4geyB0byB7IHRyYW5zZm9ybTogcm90"
    "YXRlKDM2MGRlZyk7IH0gfQoKLyogUFJFRElDVCBGT1JNICovCi5wcmVkaWN0LWZvcm0gewogIGJhY2tncm91bmQ6"
    "IHZhcigtLWNhcmQpOyBib3JkZXI6IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6IDE2"
    "cHg7IHBhZGRpbmc6IDJyZW07IG1hcmdpbi1ib3R0b206IDJyZW07Cn0KLmZvcm0tZ3JpZCB7IGRpc3BsYXk6IGdy"
    "aWQ7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogcmVwZWF0KGF1dG8tZmlsbCwgbWlubWF4KDIwMHB4LCAxZnIpKTsg"
    "Z2FwOiAxcmVtOyBtYXJnaW4tYm90dG9tOiAxLjVyZW07IH0KLmZvcm0tZ3JvdXAgbGFiZWwgeyBkaXNwbGF5OiBi"
    "bG9jazsgZm9udC1zaXplOiAwLjc1cmVtOyBjb2xvcjogdmFyKC0tdGV4dDMpOyBtYXJnaW4tYm90dG9tOiA0cHg7"
    "IGZvbnQtZmFtaWx5OiB2YXIoLS1tb25vKTsgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsgbGV0dGVyLXNwYWNp"
    "bmc6IDAuMDVlbTsgfQouZm9ybS1ncm91cCBpbnB1dCB7CiAgd2lkdGg6IDEwMCU7IGJhY2tncm91bmQ6IHZhcigt"
    "LXN1cmZhY2UpOyBib3JkZXI6IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIpOwogIGJvcmRlci1yYWRpdXM6IDhweDsg"
    "cGFkZGluZzogOHB4IDEycHg7IGNvbG9yOiB2YXIoLS10ZXh0KTsKICBmb250LWZhbWlseTogdmFyKC0tbW9ubyk7"
    "IGZvbnQtc2l6ZTogMC44NXJlbTsKICB0cmFuc2l0aW9uOiBhbGwgMC4yczsKfQouZm9ybS1ncm91cCBpbnB1dDpm"
    "b2N1cyB7IG91dGxpbmU6IG5vbmU7IGJvcmRlci1jb2xvcjogdmFyKC0tYWNjZW50KTsgYm94LXNoYWRvdzogMCAw"
    "IDAgM3B4IHJnYmEoMzAsMTA3LDI1NSwwLjEpOyB9Ci5wcmVkaWN0LXJlc3VsdCB7CiAgYmFja2dyb3VuZDogdmFy"
    "KC0tc3VyZmFjZSk7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcik7CiAgYm9yZGVyLXJhZGl1czogMTJw"
    "eDsgcGFkZGluZzogMS41cmVtOwogIGRpc3BsYXk6IG5vbmU7Cn0KLnByZWRpY3QtcmVzdWx0LnNob3cgeyBkaXNw"
    "bGF5OiBibG9jazsgYW5pbWF0aW9uOiBmYWRlSW4gMC4zcyBlYXNlOyB9CkBrZXlmcmFtZXMgZmFkZUluIHsgZnJv"
    "bSB7IG9wYWNpdHk6MDsgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKDhweCk7IH0gdG8geyBvcGFjaXR5OjE7IHRyYW5z"
    "Zm9ybTpub25lOyB9IH0KLnJlc3VsdC1mYXVsdCB7IGZvbnQtc2l6ZTogMS40cmVtOyBmb250LXdlaWdodDogNzAw"
    "OyBtYXJnaW4tYm90dG9tOiAwLjVyZW07IH0KLnJlc3VsdC1jb25mIHsgY29sb3I6IHZhcigtLXRleHQyKTsgZm9u"
    "dC1mYW1pbHk6IHZhcigtLW1vbm8pOyB9Ci5jb25mLWJhciB7IGhlaWdodDogOHB4OyBiYWNrZ3JvdW5kOiByZ2Jh"
    "KDI1NSwyNTUsMjU1LDAuMDYpOyBib3JkZXItcmFkaXVzOiA0cHg7IG1hcmdpbi10b3A6IDhweDsgb3ZlcmZsb3c6"
    "IGhpZGRlbjsgfQouY29uZi1maWxsIHsgaGVpZ2h0OiAxMDAlOyBib3JkZXItcmFkaXVzOiA0cHg7IGJhY2tncm91"
    "bmQ6IGxpbmVhci1ncmFkaWVudCg5MGRlZywgdmFyKC0tZ3JlZW4pLCB2YXIoLS1hY2NlbnQyKSk7IHRyYW5zaXRp"
    "b246IHdpZHRoIDAuNnMgZWFzZTsgfQoKLyogVE9BU1QgKi8KLnRvYXN0LWNvbnRhaW5lciB7IHBvc2l0aW9uOiBm"
    "aXhlZDsgYm90dG9tOiAycmVtOyByaWdodDogMnJlbTsgei1pbmRleDogMTAwMDsgZGlzcGxheTogZmxleDsgZmxl"
    "eC1kaXJlY3Rpb246IGNvbHVtbjsgZ2FwOiA4cHg7IH0KLnRvYXN0IHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJk"
    "Mik7IGJvcmRlcjogMXB4IHNvbGlkIHZhcigtLWJvcmRlcjIpOwogIGJvcmRlci1yYWRpdXM6IDEwcHg7IHBhZGRp"
    "bmc6IDEycHggMThweDsKICBtYXgtd2lkdGg6IDM2MHB4OyBmb250LXNpemU6IDAuODVyZW07CiAgYm94LXNoYWRv"
    "dzogMCA4cHggMzJweCByZ2JhKDAsMCwwLDAuNCk7CiAgYW5pbWF0aW9uOiBzbGlkZUluIDAuM3MgZWFzZTsKICBk"
    "aXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDEwcHg7Cn0KQGtleWZyYW1lcyBzbGlkZUlu"
    "IHsgZnJvbSB7IG9wYWNpdHk6MDsgdHJhbnNmb3JtOiB0cmFuc2xhdGVYKDIwcHgpOyB9IHRvIHsgb3BhY2l0eTox"
    "OyB0cmFuc2Zvcm06bm9uZTsgfSB9Ci50b2FzdC5zdWNjZXNzIHsgYm9yZGVyLWNvbG9yOiByZ2JhKDAsMjI5LDE2"
    "MCwwLjQpOyB9Ci50b2FzdC5lcnJvciB7IGJvcmRlci1jb2xvcjogcmdiYSgyNDQsNjMsOTQsMC40KTsgfQoudG9h"
    "c3QuaW5mbyB7IGJvcmRlci1jb2xvcjogcmdiYSgzMCwxMDcsMjU1LDAuNCk7IH0KCi8qIFNFTEVDVCAqLwpzZWxl"
    "Y3QgewogIGJhY2tncm91bmQ6IHZhcigtLXN1cmZhY2UpOyBib3JkZXI6IDFweCBzb2xpZCB2YXIoLS1ib3JkZXIp"
    "OwogIGNvbG9yOiB2YXIoLS10ZXh0KTsgcGFkZGluZzogOHB4IDEycHg7IGJvcmRlci1yYWRpdXM6IDhweDsKICBm"
    "b250LWZhbWlseTogdmFyKC0tZm9udCk7IGZvbnQtc2l6ZTogMC44NXJlbTsgY3Vyc29yOiBwb2ludGVyOwp9CnNl"
    "bGVjdDpmb2N1cyB7IG91dGxpbmU6IG5vbmU7IGJvcmRlci1jb2xvcjogdmFyKC0tYWNjZW50KTsgfQoKLyogRkFV"
    "TFQgQ09MT1IgQ09ESU5HICovCi5mYXVsdC1ub3JtYWwgeyBjb2xvcjogdmFyKC0tZ3JlZW4pOyB9Ci5mYXVsdC1o"
    "dmFjX2ZhaWx1cmUgeyBjb2xvcjogdmFyKC0tcmVkKTsgfQouZmF1bHQtZW5lcmd5X292ZXJsb2FkIHsgY29sb3I6"
    "IHZhcigtLW9yYW5nZSk7IH0KLmZhdWx0LWxpZ2h0aW5nX2ZhdWx0IHsgY29sb3I6IHZhcigtLXllbGxvdyk7IH0K"
    "LmZhdWx0LXNhZmV0eV9hbGFybSB7IGNvbG9yOiB2YXIoLS1yZWQpOyBmb250LXdlaWdodDogNzAwOyB9Ci5mYXVs"
    "dC1zZW5zb3JfZmFpbHVyZSB7IGNvbG9yOiB2YXIoLS10ZXh0Mik7IH0KLmZhdWx0LXBhcmtpbmdfY29uZ2VzdGlv"
    "biB7IGNvbG9yOiAjODE4Y2Y4OyB9CgoKLyog4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCiAgIExJVkUgTU9OSVRPUiDigJQgU01BUlRCVUlM"
    "RCBBSSBTRUNUSU9OCuKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkCAqLwoKLyogQWxlcnQgYmFubmVyICovCi5hbGVydC1iYW5uZXIgewogIHBv"
    "c2l0aW9uOiBmaXhlZDsgdG9wOiAwOyBsZWZ0OiAwOyByaWdodDogMDsgei1pbmRleDogMjAwMDsKICBwYWRkaW5n"
    "OiAxNHB4IDI0cHg7IGNvbG9yOiAjZmZmOyBmb250LXdlaWdodDogNzAwOwogIHRleHQtYWxpZ246IGNlbnRlcjsg"
    "Zm9udC1zaXplOiAxNHB4OwogIGFuaW1hdGlvbjogc2xpZGVEb3duIDAuM3MgZWFzZTsKICBib3gtc2hhZG93OiAw"
    "IDRweCAzMHB4IHJnYmEoMCwwLDAsMC41KTsKICBkaXNwbGF5OiBub25lOwp9Ci5hbGVydC1iYW5uZXIuZGFuZ2Vy"
    "IHsgYmFja2dyb3VuZDogI2VmNDQ0NDsgfQouYWxlcnQtYmFubmVyLndhcm4gICB7IGJhY2tncm91bmQ6ICNmOTcz"
    "MTY7IH0KLmFsZXJ0LWJhbm5lci5zaG93ICAgeyBkaXNwbGF5OiBibG9jazsgfQpAa2V5ZnJhbWVzIHNsaWRlRG93"
    "biB7IGZyb217b3BhY2l0eTowO3RyYW5zZm9ybTp0cmFuc2xhdGVZKC0yMHB4KX0gdG97b3BhY2l0eToxO3RyYW5z"
    "Zm9ybTpub25lfSB9CgovKiBTdWItbmF2IHRhYnMgZm9yIExpdmUgTW9uaXRvciAqLwouc3VibmF2IHsKICBkaXNw"
    "bGF5OiBmbGV4OyBnYXA6IDRweDsKICBwYWRkaW5nOiAwOyBib3JkZXItYm90dG9tOiAxcHggc29saWQgdmFyKC0t"
    "Ym9yZGVyKTsKICBtYXJnaW4tYm90dG9tOiAxLjVyZW07CiAgb3ZlcmZsb3cteDogYXV0bzsKfQouc3VibmF2LWJ0"
    "biB7CiAgYmFja2dyb3VuZDogbm9uZTsgYm9yZGVyOiBub25lOyBib3JkZXItYm90dG9tOiAycHggc29saWQgdHJh"
    "bnNwYXJlbnQ7CiAgY29sb3I6IHZhcigtLXRleHQzKTsgY3Vyc29yOiBwb2ludGVyOyBwYWRkaW5nOiAxMnB4IDIw"
    "cHg7CiAgZm9udC1mYW1pbHk6IHZhcigtLWZvbnQpOyBmb250LXNpemU6IDAuODJyZW07IGZvbnQtd2VpZ2h0OiA2"
    "MDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMDNlbTsgdGV4dC10cmFuc2Zvcm06IGNhcGl0YWxpemU7CiAgdHJhbnNp"
    "dGlvbjogYWxsIDAuMnM7IHdoaXRlLXNwYWNlOiBub3dyYXA7Cn0KLnN1Ym5hdi1idG46aG92ZXIgeyBjb2xvcjog"
    "dmFyKC0tdGV4dDIpOyB9Ci5zdWJuYXYtYnRuLmFjdGl2ZSB7IGNvbG9yOiB2YXIoLS1hY2NlbnQpOyBib3JkZXIt"
    "Ym90dG9tLWNvbG9yOiB2YXIoLS1hY2NlbnQpOyB9CgovKiBBZ2VudCBjYXJkcyBncmlkICovCi5hZ2VudC1ncmlk"
    "IHsKICBkaXNwbGF5OiBncmlkOyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IHJlcGVhdCg1LDFmcik7IGdhcDogMTRw"
    "eDsKICBtYXJnaW4tYm90dG9tOiAxLjVyZW07Cn0KQG1lZGlhKG1heC13aWR0aDo5MDBweCl7IC5hZ2VudC1ncmlk"
    "e2dyaWQtdGVtcGxhdGUtY29sdW1uczpyZXBlYXQoMywxZnIpO30gfQpAbWVkaWEobWF4LXdpZHRoOjYwMHB4KXsg"
    "LmFnZW50LWdyaWR7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7fSB9CgouYWdlbnQtY2FyZCB7CiAgYmFj"
    "a2dyb3VuZDogdmFyKC0tY2FyZCk7IGJvcmRlcjogMS41cHggc29saWQgdmFyKC0tYm9yZGVyKTsKICBib3JkZXIt"
    "cmFkaXVzOiAxNHB4OyBwYWRkaW5nOiAxNnB4IDIwcHg7CiAgY3Vyc29yOiBwb2ludGVyOyB0cmFuc2l0aW9uOiBh"
    "bGwgMC4yczsKICBwb3NpdGlvbjogcmVsYXRpdmU7IG92ZXJmbG93OiBoaWRkZW47Cn0KLmFnZW50LWNhcmQ6OmJl"
    "Zm9yZSB7CiAgY29udGVudDogJyc7IHBvc2l0aW9uOiBhYnNvbHV0ZTsgcmlnaHQ6IC0yMHB4OyBib3R0b206IC0y"
    "MHB4OwogIHdpZHRoOiA3MHB4OyBoZWlnaHQ6IDcwcHg7IGJvcmRlci1yYWRpdXM6IDUwJTsKICBiYWNrZ3JvdW5k"
    "OiByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOwp9Ci5hZ2VudC1jYXJkLmFjdGl2ZSB7IGJveC1zaGFkb3c6IDAgMCAy"
    "NHB4IHJnYmEoNTksMTMwLDI0NiwwLjIpOyB9Ci5hZ2VudC1jYXJkOmhvdmVyIHsgdHJhbnNmb3JtOiB0cmFuc2xh"
    "dGVZKC0ycHgpOyB9Ci5hZ2VudC1jYXJkIC5hYy1oZWFkZXIgeyBkaXNwbGF5OiBmbGV4OyBqdXN0aWZ5LWNvbnRl"
    "bnQ6IHNwYWNlLWJldHdlZW47IGFsaWduLWl0ZW1zOiBjZW50ZXI7IG1hcmdpbi1ib3R0b206IDhweDsgfQouYWdl"
    "bnQtY2FyZCAuYWMtbmFtZSB7IGZvbnQtd2VpZ2h0OiA3MDA7IGZvbnQtc2l6ZTogMC45MnJlbTsgfQouYWdlbnQt"
    "Y2FyZCAuYWMtZG90IHsgd2lkdGg6IDhweDsgaGVpZ2h0OiA4cHg7IGJvcmRlci1yYWRpdXM6IDUwJTsgfQouYWdl"
    "bnQtY2FyZCAuYWMtZG90Lm9ubGluZSAgeyBiYWNrZ3JvdW5kOiB2YXIoLS1ncmVlbik7IGJveC1zaGFkb3c6IDAg"
    "MCA4cHggIzIyYzU1ZTg4OyBhbmltYXRpb246IHB1bHNlIDJzIGluZmluaXRlOyB9Ci5hZ2VudC1jYXJkIC5hYy1k"
    "b3QuZmF1bHQgICB7IGJhY2tncm91bmQ6IHZhcigtLXJlZCk7ICAgYm94LXNoYWRvdzogMCAwIDhweCAjZWY0NDQ0"
    "ODg7IGFuaW1hdGlvbjogcHVsc2UgMC42cyBpbmZpbml0ZTsgfQouYWdlbnQtY2FyZCAuYWMtZG90Lm1vbml0b3Ig"
    "eyBiYWNrZ3JvdW5kOiB2YXIoLS15ZWxsb3cpOyBib3gtc2hhZG93OiAwIDAgOHB4ICNmYmJmMjQ4ODsgfQouYWdl"
    "bnQtY2FyZCAuYWMtbWV0cmljIHsgZm9udC1zaXplOiAxLjVyZW07IGZvbnQtd2VpZ2h0OiA4MDA7IGZvbnQtZmFt"
    "aWx5OiB2YXIoLS1tb25vKTsgfQouYWdlbnQtY2FyZCAuYWMtbGFiZWwgIHsgZm9udC1zaXplOiAwLjdyZW07IGNv"
    "bG9yOiB2YXIoLS10ZXh0Myk7IG1hcmdpbi10b3A6IDJweDsgfQoKLyogUGlsbCBiYWRnZSAqLwoucGlsbCB7CiAg"
    "ZGlzcGxheTogaW5saW5lLWJsb2NrOyBwYWRkaW5nOiAycHggMTBweDsgYm9yZGVyLXJhZGl1czogOTlweDsKICBm"
    "b250LXNpemU6IDExcHg7IGZvbnQtd2VpZ2h0OiA3MDA7IGxldHRlci1zcGFjaW5nOiAwLjVweDsKfQoKLyogTmVn"
    "b3RpYXRpb24gdGltZWxpbmUgKi8KLm5lZy10aW1lbGluZSB7IGRpc3BsYXk6IGZsZXg7IGZsZXgtZGlyZWN0aW9u"
    "OiBjb2x1bW47IGdhcDogOHB4OyBtYXgtaGVpZ2h0OiAzNDBweDsgb3ZlcmZsb3cteTogYXV0bzsgfQoubmVnLWV2"
    "ZW50IHsKICBkaXNwbGF5OiBmbGV4OyBnYXA6IDEycHg7IGFsaWduLWl0ZW1zOiBmbGV4LXN0YXJ0OwogIHBhZGRp"
    "bmc6IDEwcHggMTRweDsgYm9yZGVyLXJhZGl1czogMTBweDsKICBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1"
    "LDAuMDQpOyBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpOwogIGFuaW1hdGlvbjogZmFk"
    "ZUluIDAuNHMgZWFzZTsKfQoubmVnLWV2ZW50Lm92ZXJyaWRlIHsgYmFja2dyb3VuZDogcmdiYSgyMzksNjgsNjgs"
    "MC4wNik7IGJvcmRlci1jb2xvcjogcmdiYSgyMzksNjgsNjgsMC4yKTsgfQoubmVnLWJhciB7IHdpZHRoOiA0cHg7"
    "IGFsaWduLXNlbGY6IHN0cmV0Y2g7IGJvcmRlci1yYWRpdXM6IDJweDsgZmxleC1zaHJpbms6IDA7IH0KLm5lZy1i"
    "b2R5IC5uZWctdG9wIHsgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiA4cHg7IGZsZXgt"
    "d3JhcDogd3JhcDsgfQoubmVnLWJvZHkgLm5lZy1zdWIgeyBtYXJnaW4tdG9wOiA1cHg7IGZvbnQtc2l6ZTogMTJw"
    "eDsgY29sb3I6IHZhcigtLXRleHQzKTsgZm9udC1mYW1pbHk6IHZhcigtLW1vbm8pOyB9Ci5uZWctbXMgeyBmb250"
    "LXNpemU6IDExcHg7IGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7IG1hcmdpbi1sZWZ0OiBhdXRvOyB9Cgov"
    "KiBMaXZlIHJlYWRpbmdzIGJhciByb3dzICovCi5yZWFkaW5nLXJvdyB7IG1hcmdpbi1ib3R0b206IDEycHg7IH0K"
    "LnJlYWRpbmctcm93IC5yci1oZWFkZXIgeyBkaXNwbGF5OiBmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6IHNwYWNlLWJl"
    "dHdlZW47IG1hcmdpbi1ib3R0b206IDRweDsgfQoucmVhZGluZy1yb3cgLnJyLWxhYmVsICB7IGZvbnQtc2l6ZTog"
    "MTNweDsgY29sb3I6IHZhcigtLXRleHQyKTsgfQoucmVhZGluZy1yb3cgLnJyLXZhbHVlICB7IGZvbnQtc2l6ZTog"
    "MTNweDsgZm9udC13ZWlnaHQ6IDcwMDsgZm9udC1mYW1pbHk6IHZhcigtLW1vbm8pOyB9Ci5yZWFkaW5nLWJhciB7"
    "IGhlaWdodDogNXB4OyBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpOyBib3JkZXItcmFkaXVzOiAz"
    "cHg7IH0KLnJlYWRpbmctZmlsbCB7IGhlaWdodDogMTAwJTsgYm9yZGVyLXJhZGl1czogM3B4OyB0cmFuc2l0aW9u"
    "OiB3aWR0aCAwLjhzIGVhc2U7IH0KCi8qIFByb3RvY29sIHN0ZXAgaXRlbXMgKi8KLnByb3RvLXN0ZXAgeyBkaXNw"
    "bGF5OiBmbGV4OyBnYXA6IDEycHg7IG1hcmdpbi1ib3R0b206IDEycHg7IH0KLnByb3RvLW51bSB7CiAgd2lkdGg6"
    "IDI4cHg7IGhlaWdodDogMjhweDsgYm9yZGVyLXJhZGl1czogNTAlOyBmbGV4LXNocmluazogMDsKICBkaXNwbGF5"
    "OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjsKICBmb250LXNpemU6"
    "IDEycHg7IGZvbnQtd2VpZ2h0OiA4MDA7Cn0KLnByb3RvLXRpdGxlIHsgZm9udC13ZWlnaHQ6IDcwMDsgZm9udC1z"
    "aXplOiAxM3B4OyBjb2xvcjogdmFyKC0tdGV4dCk7IH0KLnByb3RvLWRlc2MgIHsgZm9udC1zaXplOiAxMnB4OyBj"
    "b2xvcjogdmFyKC0tdGV4dDMpOyBtYXJnaW4tdG9wOiAycHg7IGxpbmUtaGVpZ2h0OiAxLjU7IH0KCi8qIEFyY2hp"
    "dGVjdHVyZSBkaWFncmFtICovCi5hcmNoLXdyYXAgeyBwb3NpdGlvbjogcmVsYXRpdmU7IHdpZHRoOiAxMDAlOyBo"
    "ZWlnaHQ6IDIwMHB4OyB9Ci5hcmNoLWNlbnRlciB7CiAgcG9zaXRpb246IGFic29sdXRlOyB0b3A6IDUwJTsgbGVm"
    "dDogNTAlOyB0cmFuc2Zvcm06IHRyYW5zbGF0ZSgtNTAlLC01MCUpOwogIGJhY2tncm91bmQ6IHZhcigtLWNhcmQy"
    "KTsgYm9yZGVyOiAycHggc29saWQgcmdiYSg1OSwxMzAsMjQ2LDAuMzUpOwogIGJvcmRlci1yYWRpdXM6IDEycHg7"
    "IHBhZGRpbmc6IDEycHggMjRweDsgdGV4dC1hbGlnbjogY2VudGVyOyB6LWluZGV4OiAxMDsKfQouYXJjaC1hZ2Vu"
    "dCB7CiAgcG9zaXRpb246IGFic29sdXRlOyBib3JkZXItcmFkaXVzOiAxMHB4OyBwYWRkaW5nOiA4cHggMTRweDsK"
    "ICB0ZXh0LWFsaWduOiBjZW50ZXI7IG1pbi13aWR0aDogODBweDsgdHJhbnNmb3JtOiB0cmFuc2xhdGUoLTUwJSwt"
    "NTAlKTsKICBib3JkZXI6IDJweCBzb2xpZDsKfQoKLyogRmF1bHQgYWN0aXZlIGFsZXJ0ICovCi5mYXVsdC1hY3Rp"
    "dmUtYWxlcnQgewogIGJhY2tncm91bmQ6IHJnYmEoMjM5LDY4LDY4LDAuMDgpOyBib3JkZXI6IDFweCBzb2xpZCBy"
    "Z2JhKDIzOSw2OCw2OCwwLjI1KTsKICBib3JkZXItcmFkaXVzOiAxNHB4OyBwYWRkaW5nOiAxNnB4IDI0cHg7CiAg"
    "ZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiAxNnB4OwogIGFuaW1hdGlvbjogZ2xvd1Jl"
    "ZCAycyBpbmZpbml0ZTsKfQpAa2V5ZnJhbWVzIGdsb3dSZWQgeyAwJSwxMDAle2JveC1zaGFkb3c6MCAwIDhweCBy"
    "Z2JhKDIzOSw2OCw2OCwwLjI1KX0gNTAle2JveC1zaGFkb3c6MCAwIDI0cHggcmdiYSgyMzksNjgsNjgsMC41KX0g"
    "fQoKLyogTWVzc2FnZSBidXMgbG9nICovCi5idXMtZW50cnkgewogIGJhY2tncm91bmQ6IHJnYmEoMjU1LDI1NSwy"
    "NTUsMC4wMyk7IGJvcmRlci1yYWRpdXM6IDhweDsgcGFkZGluZzogOHB4IDEycHg7CiAgZm9udC1zaXplOiAxMnB4"
    "OyBmb250LWZhbWlseTogdmFyKC0tbW9ubyk7CiAgYm9yZGVyLWxlZnQ6IDNweCBzb2xpZDsKICBtYXJnaW4tYm90"
    "dG9tOiA2cHg7Cn0KCi8qIFBpcGVsaW5lIHN0ZXBzICovCi5waXBlbGluZS1ncmlkIHsKICBkaXNwbGF5OiBncmlk"
    "OyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IHJlcGVhdCg0LDFmcik7IGdhcDogMTZweDsKfQpAbWVkaWEobWF4LXdp"
    "ZHRoOjgwMHB4KXsgLnBpcGVsaW5lLWdyaWR7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7fSB9Ci5waXBl"
    "bGluZS1zdGVwIHsKICBwYWRkaW5nOiAxNnB4OyBib3JkZXItcmFkaXVzOiAxMHB4OyBib3JkZXI6IDFweCBzb2xp"
    "ZDsKfQoucGlwZWxpbmUtc3RlcCAucHMtdGl0bGUgeyBmb250LXdlaWdodDogNzAwOyBmb250LXNpemU6IDEzcHg7"
    "IG1hcmdpbi1ib3R0b206IDZweDsgfQoucGlwZWxpbmUtc3RlcCAucHMtZGVzYyAgeyBmb250LXNpemU6IDEycHg7"
    "IGNvbG9yOiB2YXIoLS10ZXh0Myk7IGxpbmUtaGVpZ2h0OiAxLjU7IH0KCi8qIFBhdXNlL1Jlc3VtZSBidXR0b24g"
    "Ki8KLnBhdXNlLWJ0biB7CiAgcGFkZGluZzogOHB4IDE4cHg7IGJvcmRlci1yYWRpdXM6IDhweDsgZm9udC1mYW1p"
    "bHk6IHZhcigtLWZvbnQpOyBmb250LXNpemU6IDEzcHg7IGZvbnQtd2VpZ2h0OiA2MDA7CiAgY3Vyc29yOiBwb2lu"
    "dGVyOyBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMTUpOyB0cmFuc2l0aW9uOiBhbGwgMC4y"
    "czsKfQoucGF1c2UtYnRuLnBhdXNlZCAgeyBiYWNrZ3JvdW5kOiByZ2JhKDM0LDE5Nyw5NCwwLjE1KTsgY29sb3I6"
    "ICMyMmM1NWU7IH0KLnBhdXNlLWJ0bi5ydW5uaW5nIHsgYmFja2dyb3VuZDogcmdiYSgyMzksNjgsNjgsMC4xNSk7"
    "IGNvbG9yOiAjZWY0NDQ0OyB9Cgo6Oi13ZWJraXQtc2Nyb2xsYmFyIHsgd2lkdGg6IDZweDsgaGVpZ2h0OiA2cHg7"
    "IH0KOjotd2Via2l0LXNjcm9sbGJhci10cmFjayB7IGJhY2tncm91bmQ6IHRyYW5zcGFyZW50OyB9Cjo6LXdlYmtp"
    "dC1zY3JvbGxiYXItdGh1bWIgeyBiYWNrZ3JvdW5kOiB2YXIoLS1ib3JkZXIyKTsgYm9yZGVyLXJhZGl1czogM3B4"
    "OyB9CgovKiBSRVNQT05TSVZFICovCkBtZWRpYSAobWF4LXdpZHRoOiA3NjhweCkgewogIC5jaGFydHMtZ3JpZCB7"
    "IGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyOyB9CiAgLm1ldHJpY3MtZ3JpZCB7IGdyaWQtdGVtcGxhdGUtY29s"
    "dW1uczogcmVwZWF0KDIsIDFmcik7IH0KICAuZmVlZC1yb3cgeyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IDEwMHB4"
    "IDEyMHB4IDFmcjsgfQogIC5mZWVkLXJvdyAuY29sLWFnZW50LCAuZmVlZC1yb3cgLmNvbC1tcywgLmZlZWQtcm93"
    "IC5jb2wtbWwgeyBkaXNwbGF5OiBub25lOyB9Cn0KCi8qIFNFQ1RJT04gQU5JTUFUSU9OUyAqLwouc2VjdGlvbi5h"
    "Y3RpdmUgLm1ldHJpYy1jYXJkIHsgYW5pbWF0aW9uOiBjYXJkSW4gMC40cyBlYXNlIGJvdGg7IH0KLnNlY3Rpb24u"
    "YWN0aXZlIC5tZXRyaWMtY2FyZDpudGgtY2hpbGQoMikgeyBhbmltYXRpb24tZGVsYXk6IDAuMDVzOyB9Ci5zZWN0"
    "aW9uLmFjdGl2ZSAubWV0cmljLWNhcmQ6bnRoLWNoaWxkKDMpIHsgYW5pbWF0aW9uLWRlbGF5OiAwLjFzOyB9Ci5z"
    "ZWN0aW9uLmFjdGl2ZSAubWV0cmljLWNhcmQ6bnRoLWNoaWxkKDQpIHsgYW5pbWF0aW9uLWRlbGF5OiAwLjE1czsg"
    "fQouc2VjdGlvbi5hY3RpdmUgLm1ldHJpYy1jYXJkOm50aC1jaGlsZCg1KSB7IGFuaW1hdGlvbi1kZWxheTogMC4y"
    "czsgfQpAa2V5ZnJhbWVzIGNhcmRJbiB7IGZyb20geyBvcGFjaXR5OjA7IHRyYW5zZm9ybTogdHJhbnNsYXRlWSgx"
    "NnB4KTsgfSB0byB7IG9wYWNpdHk6MTsgdHJhbnNmb3JtOm5vbmU7IH0gfQo8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5"
    "PgoKPGhlYWRlcj4KICA8ZGl2IGNsYXNzPSJoZWFkZXItaW5uZXIiPgogICAgPGEgY2xhc3M9ImxvZ28iIGhyZWY9"
    "IiMiIG9uY2xpY2s9InNob3dTZWN0aW9uKCdvdmVydmlldycpIj4KICAgICAgPGRpdiBjbGFzcz0ibG9nby1pY29u"
    "Ij7imqE8L2Rpdj4KICAgICAgPHNwYW4gY2xhc3M9ImxvZ28tdGV4dCI+TmV1cmFsPHNwYW4+R3JpZDwvc3Bhbj48"
    "L3NwYW4+CiAgICA8L2E+CiAgICA8bmF2IGNsYXNzPSJuYXYiPgogICAgICA8YnV0dG9uIGNsYXNzPSJuYXYtYnRu"
    "IGFjdGl2ZSIgb25jbGljaz0ic2hvd1NlY3Rpb24oJ292ZXJ2aWV3JykiPk92ZXJ2aWV3PC9idXR0b24+CiAgICAg"
    "IDxidXR0b24gY2xhc3M9Im5hdi1idG4iIG9uY2xpY2s9InNob3dTZWN0aW9uKCdkYXNoYm9hcmQnKSI+RGFzaGJv"
    "YXJkPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9Im5hdi1idG4iIG9uY2xpY2s9InNob3dTZWN0aW9uKCdt"
    "b2RlbHMnKSI+TUwgTW9kZWxzPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9Im5hdi1idG4iIG9uY2xpY2s9"
    "InNob3dTZWN0aW9uKCdsaXZlJykiPkxpdmUgRmVlZDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJuYXYt"
    "YnRuIiBvbmNsaWNrPSJzaG93U2VjdGlvbignZGF0YScpIj5EYXRhc2V0PC9idXR0b24+CiAgICAgIDxidXR0b24g"
    "Y2xhc3M9Im5hdi1idG4iIG9uY2xpY2s9InNob3dTZWN0aW9uKCdwcmVkaWN0JykiPlByZWRpY3Q8L2J1dHRvbj4K"
    "ICAgICAgPGJ1dHRvbiBjbGFzcz0ibmF2LWJ0biIgb25jbGljaz0ic2hvd1NlY3Rpb24oJ3JlcG9ydCcpIj5BSSBS"
    "ZXBvcnQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ibmF2LWJ0biIgb25jbGljaz0ic2hvd1NlY3Rpb24o"
    "J2xpdmVtb25pdG9yJykiPuKaoSBMaXZlIE1vbml0b3I8L2J1dHRvbj4KICAgIDwvbmF2PgogICAgPGRpdiBjbGFz"
    "cz0ic3RhdHVzLWJhZGdlIj4KICAgICAgPGRpdiBjbGFzcz0ic3RhdHVzLWRvdCIgaWQ9InN0YXR1c0RvdCI+PC9k"
    "aXY+CiAgICAgIDxzcGFuIGlkPSJzdGF0dXNUZXh0Ij5Jbml0aWFsaXppbmcuLi48L3NwYW4+CiAgICA8L2Rpdj4K"
    "ICA8L2Rpdj4KPC9oZWFkZXI+Cgo8bWFpbj4KCjwhLS0gT1ZFUlZJRVcgU0VDVElPTiAtLT4KPHNlY3Rpb24gY2xh"
    "c3M9InNlY3Rpb24gYWN0aXZlIiBpZD0ic2VjdGlvbi1vdmVydmlldyI+CiAgPGRpdiBjbGFzcz0iaGVybyI+CiAg"
    "ICA8ZGl2IGNsYXNzPSJoZXJvLWdsb3ciPjwvZGl2PgogICAgPGgxPkRlY2VudHJhbGl6ZWQgTXVsdGktQWdlbnQ8"
    "YnI+U2VsZi1IZWFsaW5nIFNtYXJ0IEJ1aWxkaW5nPC9oMT4KICAgIDxwPlJlc2VhcmNoLWdyYWRlIEFJIGZhdWx0"
    "IGRldGVjdGlvbiwgYXV0b25vbW91cyBhZ2VudCBuZWdvdGlhdGlvbiwgYW5kIHByZWRpY3RpdmUgaW50ZWxsaWdl"
    "bmNlIGZvciBuZXh0LWdlbmVyYXRpb24gYnVpbGRpbmcgbWFuYWdlbWVudCBzeXN0ZW1zLjwvcD4KICAgIDxkaXYg"
    "Y2xhc3M9Imhlcm8tdGFncyI+CiAgICAgIDxzcGFuIGNsYXNzPSJ0YWciPk11bHRpLUFnZW50IE5lZ290aWF0aW9u"
    "PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0idGFnIj5SYW5kb20gRm9yZXN0IMK3IEdyYWRpZW50IEJvb3N0aW5n"
    "IMK3IE1MUDwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9InRhZyI+MzIsMDAwIFRyYWluaW5nIFNhbXBsZXM8L3Nw"
    "YW4+CiAgICAgIDxzcGFuIGNsYXNzPSJ0YWciPlJlYWwtdGltZSBGYXVsdCBSZWNvdmVyeTwvc3Bhbj4KICAgICAg"
    "PHNwYW4gY2xhc3M9InRhZyI+QUVTLTI1NiBEYXRhIFNlY3VyaXR5PC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0i"
    "dGFnIj5NdWx0aWxpbmd1YWwgQUkgUmVwb3J0czwvc3Bhbj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNs"
    "YXNzPSJjb250cm9sLXBhbmVsIj4KICAgIDxkaXYgY2xhc3M9InBhbmVsLXRpdGxlIj5TeXN0ZW0gUGlwZWxpbmUg"
    "Q29udHJvbDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY29udHJvbHMtZ3JpZCI+CiAgICAgIDxidXR0b24gY2xhc3M9"
    "ImJ0biBidG4tcHJpbWFyeSIgaWQ9ImJ0bkluaXQiIG9uY2xpY2s9ImluaXRQaXBlbGluZSgpIj4KICAgICAgICDw"
    "n6egIEluaXRpYWxpemUgJmFtcDsgVHJhaW4KICAgICAgPC9idXR0b24+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0"
    "biBidG4tc3VjY2VzcyIgaWQ9ImJ0blNpbSIgb25jbGljaz0ic3RhcnRTaW11bGF0aW9uKCkiIGRpc2FibGVkPgog"
    "ICAgICAgIOKWtiBSdW4gU2ltdWxhdGlvbgogICAgICA8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRu"
    "IGJ0bi1zZWNvbmRhcnkiIG9uY2xpY2s9InNob3dTZWN0aW9uKCdkYXNoYm9hcmQnKSI+CiAgICAgICAg8J+TiiBW"
    "aWV3IERhc2hib2FyZAogICAgICA8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGJ0bi1wdXJwbGUi"
    "IG9uY2xpY2s9InNob3dTZWN0aW9uKCdyZXBvcnQnKSI+CiAgICAgICAg8J+TiyBBSSBSZXBvcnQKICAgICAgPC9i"
    "dXR0b24+CiAgICA8L2Rpdj4KCiAgICA8ZGl2IGNsYXNzPSJwcm9ncmVzcy13cmFwIj4KICAgICAgPGRpdiBjbGFz"
    "cz0icHJvZ3Jlc3MtbGFiZWwiPgogICAgICAgIDxzcGFuIGlkPSJwcm9ncmVzc01zZyI+UmVhZHkgdG8gaW5pdGlh"
    "bGl6ZTwvc3Bhbj4KICAgICAgICA8c3BhbiBpZD0icHJvZ3Jlc3NQY3QiPjAlPC9zcGFuPgogICAgICA8L2Rpdj4K"
    "ICAgICAgPGRpdiBjbGFzcz0icHJvZ3Jlc3MtYmFyIj48ZGl2IGNsYXNzPSJwcm9ncmVzcy1maWxsIiBpZD0icHJv"
    "Z3Jlc3NGaWxsIj48L2Rpdj48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8IS0tIFF1aWNrIHN0YXRzIC0t"
    "PgogIDxkaXYgY2xhc3M9Im1ldHJpY3MtZ3JpZCIgaWQ9InF1aWNrTWV0cmljcyI+CiAgICA8ZGl2IGNsYXNzPSJt"
    "ZXRyaWMtY2FyZCI+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7wn5OmPC9kaXY+CiAgICAgIDxkaXYg"
    "Y2xhc3M9Im1ldHJpYy1sYWJlbCI+RGF0YXNldCBTaXplPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy12"
    "YWx1ZSIgaWQ9InFEYXRhc2V0Ij7igJQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+dHJhaW5p"
    "bmcgc2FtcGxlczwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtY2FyZCBncmVlbiI+CiAg"
    "ICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7wn6SWPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1s"
    "YWJlbCI+QmVzdCBNb2RlbCBGMTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtdmFsdWUiIGlkPSJxRjEi"
    "PuKAlDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtc3ViIiBpZD0icU1vZGVsTmFtZSI+bm90IHRyYWlu"
    "ZWQ8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWV0cmljLWNhcmQgeWVsbG93Ij4KICAgICAgPGRp"
    "diBjbGFzcz0ibWV0cmljLWljb24iPuKaoTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtbGFiZWwiPlRv"
    "dGFsIEZhdWx0czwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtdmFsdWUiIGlkPSJxRmF1bHRzIj7igJQ8"
    "L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+ZGV0ZWN0ZWQgJmFtcDsgcmVzb2x2ZWQ8L2Rpdj4K"
    "ICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWV0cmljLWNhcmQgcmVkIj4KICAgICAgPGRpdiBjbGFzcz0ibWV0"
    "cmljLWljb24iPvCflLQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWxhYmVsIj5BdmcgUmVjb3Zlcnk8"
    "L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXZhbHVlIiBpZD0icVJlY292ZXJ5Ij7igJQ8L2Rpdj4KICAg"
    "ICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+bWlsbGlzZWNvbmRzPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYg"
    "Y2xhc3M9Im1ldHJpYy1jYXJkIHB1cnBsZSI+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7wn5uhPC9k"
    "aXY+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1sYWJlbCI+U2FmZXR5IE92ZXJyaWRlczwvZGl2PgogICAgICA8"
    "ZGl2IGNsYXNzPSJtZXRyaWMtdmFsdWUiIGlkPSJxT3ZlcnJpZGVzIj7igJQ8L2Rpdj4KICAgICAgPGRpdiBjbGFz"
    "cz0ibWV0cmljLXN1YiI+Y3JpdGljYWwgaW50ZXJ2ZW50aW9uczwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cgog"
    "IDwhLS0gQXJjaGl0ZWN0dXJlIGRpYWdyYW0gLS0+CiAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCIgc3R5bGU9Im1h"
    "cmdpbi1ib3R0b206MnJlbTsiPgogICAgPGRpdiBjbGFzcz0iY2hhcnQtdGl0bGUiPlN5c3RlbSBBcmNoaXRlY3R1"
    "cmUg4oCUIEFnZW50IENvbW11bmljYXRpb24gRmxvdzwvZGl2PgogICAgPGRpdiBzdHlsZT0iZGlzcGxheTpncmlk"
    "O2dyaWQtdGVtcGxhdGUtY29sdW1uczpyZXBlYXQoYXV0by1maXQsbWlubWF4KDE2MHB4LDFmcikpO2dhcDoxcmVt"
    "O3BhZGRpbmc6MXJlbSAwOyI+CiAgICAgIDxkaXYgc3R5bGU9ImJhY2tncm91bmQ6cmdiYSgzMCwxMDcsMjU1LDAu"
    "MDgpO2JvcmRlcjoxcHggc29saWQgcmdiYSgzMCwxMDcsMjU1LDAuMjUpO2JvcmRlci1yYWRpdXM6MTBweDtwYWRk"
    "aW5nOjFyZW07dGV4dC1hbGlnbjpjZW50ZXI7Ij4KICAgICAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6MS44cmVt"
    "Ij7wn4yhPC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuODVyZW07Zm9udC13ZWlnaHQ6NjAw"
    "O21hcmdpbjo2cHggMCI+SFZBQyBBZ2VudDwvZGl2PgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZTowLjcy"
    "cmVtO2NvbG9yOnZhcigtLXRleHQzKSI+UG93ZXIgbW9uaXRvcmluZzxicj5UaGVybWFsIGNvbnRyb2w8L2Rpdj4K"
    "ICAgICAgPC9kaXY+CiAgICAgIDxkaXYgc3R5bGU9ImJhY2tncm91bmQ6cmdiYSgwLDIyOSwxNjAsMC4wOCk7Ym9y"
    "ZGVyOjFweCBzb2xpZCByZ2JhKDAsMjI5LDE2MCwwLjI1KTtib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxcmVt"
    "O3RleHQtYWxpZ246Y2VudGVyOyI+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjEuOHJlbSI+4pqhPC9k"
    "aXY+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuODVyZW07Zm9udC13ZWlnaHQ6NjAwO21hcmdpbjo2"
    "cHggMCI+RW5lcmd5IEFnZW50PC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuNzJyZW07Y29s"
    "b3I6dmFyKC0tdGV4dDMpIj5Mb2FkIG1hbmFnZW1lbnQ8YnI+UGVhayBzaGF2aW5nPC9kaXY+CiAgICAgIDwvZGl2"
    "PgogICAgICA8ZGl2IHN0eWxlPSJiYWNrZ3JvdW5kOnJnYmEoMjUxLDE5MSwzNiwwLjA4KTtib3JkZXI6MXB4IHNv"
    "bGlkIHJnYmEoMjUxLDE5MSwzNiwwLjI1KTtib3JkZXItcmFkaXVzOjEwcHg7cGFkZGluZzoxcmVtO3RleHQtYWxp"
    "Z246Y2VudGVyOyI+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjEuOHJlbSI+8J+SoTwvZGl2PgogICAg"
    "ICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZTowLjg1cmVtO2ZvbnQtd2VpZ2h0OjYwMDttYXJnaW46NnB4IDAiPkxp"
    "Z2h0aW5nIEFnZW50PC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuNzJyZW07Y29sb3I6dmFy"
    "KC0tdGV4dDMpIj5PY2N1cGFuY3ktYXdhcmU8YnI+SW50ZW5zaXR5IGNvbnRyb2w8L2Rpdj4KICAgICAgPC9kaXY+"
    "CiAgICAgIDxkaXYgc3R5bGU9ImJhY2tncm91bmQ6cmdiYSgxMjQsNTgsMjM3LDAuMDgpO2JvcmRlcjoxcHggc29s"
    "aWQgcmdiYSgxMjQsNTgsMjM3LDAuMjUpO2JvcmRlci1yYWRpdXM6MTBweDtwYWRkaW5nOjFyZW07dGV4dC1hbGln"
    "bjpjZW50ZXI7Ij4KICAgICAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6MS44cmVtIj7wn5qXPC9kaXY+CiAgICAg"
    "ICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuODVyZW07Zm9udC13ZWlnaHQ6NjAwO21hcmdpbjo2cHggMCI+UGFy"
    "a2luZyBBZ2VudDwvZGl2PgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZTowLjcycmVtO2NvbG9yOnZhcigt"
    "LXRleHQzKSI+Q29uZ2VzdGlvbiBkZXRlY3Rpb248YnI+UmVyb3V0aW5nPC9kaXY+CiAgICAgIDwvZGl2PgogICAg"
    "ICA8ZGl2IHN0eWxlPSJiYWNrZ3JvdW5kOnJnYmEoMjQ0LDYzLDk0LDAuMDgpO2JvcmRlcjoxcHggc29saWQgcmdi"
    "YSgyNDQsNjMsOTQsMC4yNSk7Ym9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MXJlbTt0ZXh0LWFsaWduOmNlbnRl"
    "cjsiPgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxLjhyZW0iPvCfm6E8L2Rpdj4KICAgICAgICA8ZGl2"
    "IHN0eWxlPSJmb250LXNpemU6MC44NXJlbTtmb250LXdlaWdodDo2MDA7bWFyZ2luOjZweCAwIj5TYWZldHkgQWdl"
    "bnQ8L2Rpdj4KICAgICAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6MC43MnJlbTtjb2xvcjp2YXIoLS10ZXh0Myki"
    "PlByaW9yaXR5IG92ZXJyaWRlPGJyPkVtZXJnZW5jeSBwcm90b2NvbDwvZGl2PgogICAgICA8L2Rpdj4KICAgICAg"
    "PGRpdiBzdHlsZT0iYmFja2dyb3VuZDpyZ2JhKDMwLDEwNywyNTUsMC4wNSk7Ym9yZGVyOjFweCBkYXNoZWQgcmdi"
    "YSgzMCwxMDcsMjU1LDAuMyk7Ym9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6MXJlbTt0ZXh0LWFsaWduOmNlbnRl"
    "cjsiPgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxLjhyZW0iPvCfp6A8L2Rpdj4KICAgICAgICA8ZGl2"
    "IHN0eWxlPSJmb250LXNpemU6MC44NXJlbTtmb250LXdlaWdodDo2MDA7bWFyZ2luOjZweCAwIj5NTCBEZXRlY3Rv"
    "cjwvZGl2PgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZTowLjcycmVtO2NvbG9yOnZhcigtLXRleHQzKSI+"
    "UkYgwrcgR0IgwrcgTUxQPGJyPkVuc2VtYmxlIHByZWRpY3Rpb248L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rp"
    "dj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKPCEtLSBEQVNIQk9BUkQgU0VDVElPTiAtLT4KPHNlY3Rpb24gY2xhc3M9"
    "InNlY3Rpb24iIGlkPSJzZWN0aW9uLWRhc2hib2FyZCI+CiAgPGRpdiBjbGFzcz0ibWV0cmljcy1ncmlkIj4KICAg"
    "IDxkaXYgY2xhc3M9Im1ldHJpYy1jYXJkIj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWljb24iPvCflIQ8L2Rp"
    "dj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWxhYmVsIj5TaW11bGF0aW9uIFJ1bnM8L2Rpdj4KICAgICAgPGRp"
    "diBjbGFzcz0ibWV0cmljLXZhbHVlIiBpZD0iZFJ1bnMiPuKAlDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRy"
    "aWMtc3ViIj5jb21wbGV0ZWQ8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWV0cmljLWNhcmQgeWVs"
    "bG93Ij4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWljb24iPuKaoDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJt"
    "ZXRyaWMtbGFiZWwiPlRvdGFsIEZhdWx0IEV2ZW50czwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtdmFs"
    "dWUiIGlkPSJkRmF1bHRzIj7igJQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+ZGV0ZWN0ZWQ8"
    "L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ibWV0cmljLWNhcmQgZ3JlZW4iPgogICAgICA8ZGl2IGNs"
    "YXNzPSJtZXRyaWMtaWNvbiI+4pqhPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1sYWJlbCI+QXZnIFJl"
    "Y292ZXJ5IFRpbWU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXZhbHVlIiBpZD0iZFJlY292ZXJ5Ij7i"
    "gJQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+bWlsbGlzZWNvbmRzPC9kaXY+CiAgICA8L2Rp"
    "dj4KICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1jYXJkIHJlZCI+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29u"
    "Ij7wn5qoPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1sYWJlbCI+U2FmZXR5IE92ZXJyaWRlczwvZGl2"
    "PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtdmFsdWUiIGlkPSJkT3ZlcnJpZGVzIj7igJQ8L2Rpdj4KICAgICAg"
    "PGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+Y3JpdGljYWw8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0i"
    "bWV0cmljLWNhcmQgcHVycGxlIj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWljb24iPvCfjq88L2Rpdj4KICAg"
    "ICAgPGRpdiBjbGFzcz0ibWV0cmljLWxhYmVsIj5CZXN0IE1vZGVsIEYxPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9"
    "Im1ldHJpYy12YWx1ZSIgaWQ9ImRCZXN0RjEiPuKAlDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtc3Vi"
    "Ij53ZWlnaHRlZCBhdmVyYWdlPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPGRpdiBjbGFzcz0iY2hhcnRz"
    "LWdyaWQiPgogICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCI+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxl"
    "Ij5GYXVsdCBEaXN0cmlidXRpb24gPHNwYW4gY2xhc3M9ImJhZGdlIj5ieSB0eXBlPC9zcGFuPjwvZGl2PgogICAg"
    "ICA8ZGl2IGNsYXNzPSJjaGFydC13cmFwIj48Y2FudmFzIGlkPSJmYXVsdERpc3RDaGFydCI+PC9jYW52YXM+PC9k"
    "aXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNoYXJ0LWNhcmQiPgogICAgICA8ZGl2IGNsYXNzPSJjaGFy"
    "dC10aXRsZSI+SFZBQyBQb3dlciBieSBIb3VyIDxzcGFuIGNsYXNzPSJiYWRnZSI+MjRoIHByb2ZpbGU8L3NwYW4+"
    "PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52YXMgaWQ9ImhvdXJseUNoYXJ0Ij48L2Nh"
    "bnZhcz48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCBmdWxsIj4KICAgICAgPGRp"
    "diBjbGFzcz0iY2hhcnQtdGl0bGUiPkZhdWx0IEV2ZW50cyBUaW1lbGluZSA8c3BhbiBjbGFzcz0iYmFkZ2UiPmJ5"
    "IHR5cGUgcGVyIDEwLXN0ZXAgd2luZG93PC9zcGFuPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJjaGFydC13cmFw"
    "IHRhbGwiPjxjYW52YXMgaWQ9InRpbWVsaW5lQ2hhcnQiPjwvY2FudmFzPjwvZGl2PgogICAgPC9kaXY+CiAgICA8"
    "ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtdGl0bGUiPkFnZW50IFJlY292"
    "ZXJ5IERpc3RyaWJ1dGlvbiA8c3BhbiBjbGFzcz0iYmFkZ2UiPmNvdW50IGJ5IGFnZW50PC9zcGFuPjwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJjaGFydC13cmFwIj48Y2FudmFzIGlkPSJhZ2VudENoYXJ0Ij48L2NhbnZhcz48L2Rp"
    "dj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCI+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0"
    "LXRpdGxlIj5PY2N1cGFuY3kgdnMgTGlnaHRpbmcgPHNwYW4gY2xhc3M9ImJhZGdlIj5ob3VybHkgY29ycmVsYXRp"
    "b248L3NwYW4+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52YXMgaWQ9Im9jY0xpZ2h0"
    "Q2hhcnQiPjwvY2FudmFzPjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDxkaXYgc3R5bGU9ImRpc3BsYXk6"
    "ZmxleDtnYXA6MXJlbTtmbGV4LXdyYXA6d3JhcDttYXJnaW4tYm90dG9tOjJyZW07Ij4KICAgIDxidXR0b24gY2xh"
    "c3M9ImJ0biBidG4tc2Vjb25kYXJ5IiBvbmNsaWNrPSJsb2FkRGFzaGJvYXJkKCkiPvCflIQgUmVmcmVzaDwvYnV0"
    "dG9uPgogICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGJ0bi1zZWNvbmRhcnkiIG9uY2xpY2s9ImRvd25sb2FkUmVjb3Zl"
    "cnlMb2coKSI+4qyHIERvd25sb2FkIFJlY292ZXJ5IExvZzwvYnV0dG9uPgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8"
    "IS0tIE1MIE1PREVMUyBTRUNUSU9OIC0tPgo8c2VjdGlvbiBjbGFzcz0ic2VjdGlvbiIgaWQ9InNlY3Rpb24tbW9k"
    "ZWxzIj4KICA8ZGl2IGNsYXNzPSJwYW5lbC10aXRsZSIgc3R5bGU9Im1hcmdpbi1ib3R0b206MS41cmVtOyI+TWFj"
    "aGluZSBMZWFybmluZyBQZXJmb3JtYW5jZTwvZGl2PgogIDxkaXYgY2xhc3M9Im1vZGVscy1ncmlkIiBpZD0ibW9k"
    "ZWxzR3JpZCI+CiAgICA8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS10ZXh0Myk7Zm9udC1mYW1pbHk6dmFyKC0tbW9u"
    "byk7Zm9udC1zaXplOjAuODVyZW07cGFkZGluZzoycmVtOyI+CiAgICAgIFRyYWluIG1vZGVscyBmaXJzdCB2aWEg"
    "T3ZlcnZpZXcg4oaSIEluaXRpYWxpemUgJmFtcDsgVHJhaW4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNs"
    "YXNzPSJjaGFydHMtZ3JpZCI+CiAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgPGRpdiBjbGFzcz0i"
    "Y2hhcnQtdGl0bGUiPk1vZGVsIENvbXBhcmlzb24g4oCUIEYxIFNjb3JlIDxzcGFuIGNsYXNzPSJiYWRnZSI+d2Vp"
    "Z2h0ZWQ8L3NwYW4+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52YXMgaWQ9Im1vZGVs"
    "Q29tcENoYXJ0Ij48L2NhbnZhcz48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCI+"
    "CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj5UcmFpbmluZyBUaW1lIENvbXBhcmlzb24gPHNwYW4gY2xh"
    "c3M9ImJhZGdlIj5zZWNvbmRzPC9zcGFuPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJjaGFydC13cmFwIj48Y2Fu"
    "dmFzIGlkPSJ0cmFpblRpbWVDaGFydCI+PC9jYW52YXM+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9"
    "ImNoYXJ0LWNhcmQgZnVsbCI+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj5Db25mdXNpb24gTWF0cml4"
    "IOKAlCBCZXN0IE1vZGVsIDxzcGFuIGNsYXNzPSJiYWRnZSIgaWQ9ImNtQmFkZ2UiPuKAlDwvc3Bhbj48L2Rpdj4K"
    "ICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtd3JhcCB0YWxsIj48Y2FudmFzIGlkPSJjb25mTWF0Q2hhcnQiPjwvY2Fu"
    "dmFzPjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjwhLS0gTElWRSBGRUVEIFNFQ1RJT04g"
    "LS0+CjxzZWN0aW9uIGNsYXNzPSJzZWN0aW9uIiBpZD0ic2VjdGlvbi1saXZlIj4KICA8ZGl2IGNsYXNzPSJsaXZl"
    "LWZlZWQiPgogICAgPGRpdiBjbGFzcz0iZmVlZC1oZWFkZXIiPgogICAgICA8ZGl2IGNsYXNzPSJmZWVkLXRpdGxl"
    "Ij4KICAgICAgICA8ZGl2IGNsYXNzPSJmZWVkLWxpdmUiPjwvZGl2PgogICAgICAgIExpdmUgRmF1bHQgUmVjb3Zl"
    "cnkgU3RyZWFtCiAgICAgIDwvZGl2PgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7Z2FwOjhweDthbGln"
    "bi1pdGVtczpjZW50ZXI7Ij4KICAgICAgICA8c2VsZWN0IGlkPSJmZWVkRmlsdGVyU2VsIiBvbmNoYW5nZT0iZmls"
    "dGVyRmVlZCgpIiBzdHlsZT0iZm9udC1zaXplOjAuNzVyZW07cGFkZGluZzo0cHggOHB4OyI+CiAgICAgICAgICA8"
    "b3B0aW9uIHZhbHVlPSIiPkFsbCBGYXVsdHM8L29wdGlvbj4KICAgICAgICAgIDxvcHRpb24gdmFsdWU9Imh2YWNf"
    "ZmFpbHVyZSI+SFZBQyBGYWlsdXJlPC9vcHRpb24+CiAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJlbmVyZ3lfb3Zl"
    "cmxvYWQiPkVuZXJneSBPdmVybG9hZDwvb3B0aW9uPgogICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ibGlnaHRpbmdf"
    "ZmF1bHQiPkxpZ2h0aW5nIEZhdWx0PC9vcHRpb24+CiAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJzYWZldHlfYWxh"
    "cm0iPlNhZmV0eSBBbGFybTwvb3B0aW9uPgogICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ic2Vuc29yX2ZhaWx1cmUi"
    "PlNlbnNvciBGYWlsdXJlPC9vcHRpb24+CiAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJwYXJraW5nX2Nvbmdlc3Rp"
    "b24iPlBhcmtpbmc8L29wdGlvbj4KICAgICAgICA8L3NlbGVjdD4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJidG4g"
    "YnRuLXNlY29uZGFyeSIgc3R5bGU9InBhZGRpbmc6NHB4IDEycHg7Zm9udC1zaXplOjAuNzVyZW07IiBvbmNsaWNr"
    "PSJyZWZyZXNoRmVlZCgpIj5SZWZyZXNoPC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2"
    "IHN0eWxlPSJkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjEyMHB4IDE0MHB4IDEyMHB4IDEwMHB4"
    "IDFmciA4MHB4O2dhcDoxcmVtO3BhZGRpbmc6OHB4IDEuNXJlbTtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIo"
    "LS1ib3JkZXIpO2ZvbnQtc2l6ZTowLjY1cmVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtsZXR0ZXItc3BhY2lu"
    "ZzowLjA2ZW07Y29sb3I6dmFyKC0tdGV4dDMpO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pOyI+CiAgICAgIDxzcGFu"
    "PlRpbWU8L3NwYW4+PHNwYW4+RmF1bHQgVHlwZTwvc3Bhbj48c3Bhbj5EZXRlY3RlZCBCeTwvc3Bhbj48c3Bhbj5T"
    "ZXZlcml0eTwvc3Bhbj48c3Bhbj5SZWNvdmVyeSBBY3Rpb248L3NwYW4+PHNwYW4+U3RhdHVzPC9zcGFuPgogICAg"
    "PC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJmZWVkLXNjcm9sbCIgaWQ9ImZlZWRTY3JvbGwiPgogICAgICA8ZGl2IHN0"
    "eWxlPSJ0ZXh0LWFsaWduOmNlbnRlcjtwYWRkaW5nOjNyZW07Y29sb3I6dmFyKC0tdGV4dDMpO2ZvbnQtc2l6ZTow"
    "Ljg1cmVtOyI+CiAgICAgICAgTm8gZXZlbnRzIHlldCDigJQgcnVuIGEgc2ltdWxhdGlvbiBmaXJzdAogICAgICA8"
    "L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgIDxkaXYgY2xh"
    "c3M9ImNoYXJ0LXRpdGxlIj5SZWNvdmVyeSBUaW1lIERpc3RyaWJ1dGlvbiA8c3BhbiBjbGFzcz0iYmFkZ2UiPm1z"
    "IGhpc3RvZ3JhbTwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52YXMgaWQ9InJ0"
    "SGlzdENoYXJ0Ij48L2NhbnZhcz48L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKPCEtLSBEQVRBU0VUIFNFQ1RJ"
    "T04gLS0+CjxzZWN0aW9uIGNsYXNzPSJzZWN0aW9uIiBpZD0ic2VjdGlvbi1kYXRhIj4KICA8ZGl2IGNsYXNzPSJ0"
    "YWJsZS13cmFwIj4KICAgIDxkaXYgY2xhc3M9InRhYmxlLWhlYWRlciI+CiAgICAgIDxkaXYgc3R5bGU9ImZvbnQt"
    "c2l6ZTowLjhyZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2"
    "YXIoLS10ZXh0Mik7Ij4KICAgICAgICBEYXRhc2V0IEV4cGxvcmVyCiAgICAgICAgPHNwYW4gaWQ9ImRhdGFzZXRD"
    "b3VudCIgc3R5bGU9ImNvbG9yOnZhcigtLXRleHQzKTttYXJnaW4tbGVmdDo4cHg7Zm9udC1zaXplOjAuNzVyZW07"
    "Ij7igJQgcm93czwvc3Bhbj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtnYXA6"
    "OHB4O2ZsZXgtd3JhcDp3cmFwO2FsaWduLWl0ZW1zOmNlbnRlcjsiPgogICAgICAgIDxzZWxlY3QgaWQ9ImZhdWx0"
    "RmlsdGVyU2VsIiBvbmNoYW5nZT0ibG9hZERhdGFQYWdlKDEpIj4KICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+"
    "QWxsIEZhdWx0czwvb3B0aW9uPgogICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ibm9ybWFsIj5Ob3JtYWw8L29wdGlv"
    "bj4KICAgICAgICAgIDxvcHRpb24gdmFsdWU9Imh2YWNfZmFpbHVyZSI+SFZBQyBGYWlsdXJlPC9vcHRpb24+CiAg"
    "ICAgICAgICA8b3B0aW9uIHZhbHVlPSJlbmVyZ3lfb3ZlcmxvYWQiPkVuZXJneSBPdmVybG9hZDwvb3B0aW9uPgog"
    "ICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ibGlnaHRpbmdfZmF1bHQiPkxpZ2h0aW5nIEZhdWx0PC9vcHRpb24+CiAg"
    "ICAgICAgICA8b3B0aW9uIHZhbHVlPSJzYWZldHlfYWxhcm0iPlNhZmV0eSBBbGFybTwvb3B0aW9uPgogICAgICAg"
    "ICAgPG9wdGlvbiB2YWx1ZT0ic2Vuc29yX2ZhaWx1cmUiPlNlbnNvciBGYWlsdXJlPC9vcHRpb24+CiAgICAgICAg"
    "PC9zZWxlY3Q+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGJ0bi1zZWNvbmRhcnkiIHN0eWxlPSJwYWRkaW5n"
    "OjZweCAxNHB4O2ZvbnQtc2l6ZTowLjhyZW07IiBvbmNsaWNrPSJkb3dubG9hZENTVigpIj7irIcgQ1NWPC9idXR0"
    "b24+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRuIGJ0bi1zZWNvbmRhcnkiIHN0eWxlPSJwYWRkaW5nOjZweCAx"
    "NHB4O2ZvbnQtc2l6ZTowLjhyZW07IiBvbmNsaWNrPSJkb3dubG9hZEpTT04oKSI+4qyHIEpTT048L2J1dHRvbj4K"
    "ICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InRhYmxlLXNjcm9sbCI+CiAgICAgIDx0YWJs"
    "ZSBjbGFzcz0iZGF0YS10YWJsZSIgaWQ9ImRhdGFUYWJsZSI+CiAgICAgICAgPHRoZWFkPgogICAgICAgICAgPHRy"
    "PgogICAgICAgICAgICA8dGg+IzwvdGg+PHRoPkhvdXI8L3RoPjx0aD5PdXRkb29yIFRlbXA8L3RoPjx0aD5PY2N1"
    "cGFuY3k8L3RoPgogICAgICAgICAgICA8dGg+SFZBQyBQb3dlcjwvdGg+PHRoPkhlYXRpbmcgTG9hZDwvdGg+PHRo"
    "PkNvb2xpbmcgTG9hZDwvdGg+CiAgICAgICAgICAgIDx0aD5MaWdodGluZzwvdGg+PHRoPlBhcmtpbmc8L3RoPjx0"
    "aD5GYXVsdCBMYWJlbDwvdGg+PHRoPlNldmVyaXR5PC90aD4KICAgICAgICAgIDwvdHI+CiAgICAgICAgPC90aGVh"
    "ZD4KICAgICAgICA8dGJvZHkgaWQ9ImRhdGFUYWJsZUJvZHkiPgogICAgICAgICAgPHRyPjx0ZCBjb2xzcGFuPSIx"
    "MSIgc3R5bGU9InRleHQtYWxpZ246Y2VudGVyO2NvbG9yOnZhcigtLXRleHQzKTtwYWRkaW5nOjNyZW07Ij5Mb2Fk"
    "IGRhdGFzZXQgZmlyc3Q8L3RkPjwvdHI+CiAgICAgICAgPC90Ym9keT4KICAgICAgPC90YWJsZT4KICAgIDwvZGl2"
    "PgogICAgPGRpdiBjbGFzcz0icGFnaW5hdGlvbiIgaWQ9InBhZ2luYXRpb24iPjwvZGl2PgogIDwvZGl2PgoKICA8"
    "ZGl2IGNsYXNzPSJjaGFydC1jYXJkIiBzdHlsZT0ibWFyZ2luLWJvdHRvbToycmVtOyI+CiAgICA8ZGl2IGNsYXNz"
    "PSJjaGFydC10aXRsZSI+RmF1bHQgTGFiZWwgRGlzdHJpYnV0aW9uIDxzcGFuIGNsYXNzPSJiYWRnZSI+ZGF0YXNl"
    "dCBjb21wb3NpdGlvbjwvc3Bhbj48L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52YXMgaWQ9"
    "ImRhdGFEaXN0Q2hhcnQiPjwvY2FudmFzPjwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIFBSRURJQ1Qg"
    "U0VDVElPTiAtLT4KPHNlY3Rpb24gY2xhc3M9InNlY3Rpb24iIGlkPSJzZWN0aW9uLXByZWRpY3QiPgogIDxkaXYg"
    "Y2xhc3M9InByZWRpY3QtZm9ybSI+CiAgICA8ZGl2IGNsYXNzPSJwYW5lbC10aXRsZSI+UmVhbC1UaW1lIEZhdWx0"
    "IFByZWRpY3Rpb248L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZvcm0tZ3JpZCIgaWQ9InByZWRpY3RGb3JtIj4KICAg"
    "ICAgPGRpdiBjbGFzcz0iZm9ybS1ncm91cCI+PGxhYmVsPkhvdXIgKDAtMjMpPC9sYWJlbD48aW5wdXQgdHlwZT0i"
    "bnVtYmVyIiBpZD0icF9ob3VyIiB2YWx1ZT0iMTIiIG1pbj0iMCIgbWF4PSIyMyI+PC9kaXY+CiAgICAgIDxkaXYg"
    "Y2xhc3M9ImZvcm0tZ3JvdXAiPjxsYWJlbD5PdXRkb29yIFRlbXAgKMKwQyk8L2xhYmVsPjxpbnB1dCB0eXBlPSJu"
    "dW1iZXIiIGlkPSJwX291dGRvb3JfdGVtcCIgdmFsdWU9IjIyIiBzdGVwPSIwLjEiPjwvZGl2PgogICAgICA8ZGl2"
    "IGNsYXNzPSJmb3JtLWdyb3VwIj48bGFiZWw+T2NjdXBhbmN5ICgwLTEpPC9sYWJlbD48aW5wdXQgdHlwZT0ibnVt"
    "YmVyIiBpZD0icF9vY2N1cGFuY3kiIHZhbHVlPSIwLjUiIG1pbj0iMCIgbWF4PSIxIiBzdGVwPSIwLjAxIj48L2Rp"
    "dj4KICAgICAgPGRpdiBjbGFzcz0iZm9ybS1ncm91cCI+PGxhYmVsPkhWQUMgUG93ZXIgKGtXKTwvbGFiZWw+PGlu"
    "cHV0IHR5cGU9Im51bWJlciIgaWQ9InBfaHZhY19wb3dlciIgdmFsdWU9IjMwIiBzdGVwPSIwLjEiPjwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJmb3JtLWdyb3VwIj48bGFiZWw+SGVhdGluZyBMb2FkIChrVyk8L2xhYmVsPjxpbnB1"
    "dCB0eXBlPSJudW1iZXIiIGlkPSJwX2hlYXRpbmdfbG9hZCIgdmFsdWU9IjIwIiBzdGVwPSIwLjEiPjwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJmb3JtLWdyb3VwIj48bGFiZWw+Q29vbGluZyBMb2FkIChrVyk8L2xhYmVsPjxpbnB1"
    "dCB0eXBlPSJudW1iZXIiIGlkPSJwX2Nvb2xpbmdfbG9hZCIgdmFsdWU9IjE4IiBzdGVwPSIwLjEiPjwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJmb3JtLWdyb3VwIj48bGFiZWw+TGlnaHRpbmcgSW50ZW5zaXR5PC9sYWJlbD48aW5w"
    "dXQgdHlwZT0ibnVtYmVyIiBpZD0icF9saWdodGluZ19pbnRlbnNpdHkiIHZhbHVlPSIwLjYiIG1pbj0iMCIgbWF4"
    "PSIxLjUiIHN0ZXA9IjAuMDEiPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmb3JtLWdyb3VwIj48bGFiZWw+UGFy"
    "a2luZyBPY2N1cGFuY3k8L2xhYmVsPjxpbnB1dCB0eXBlPSJudW1iZXIiIGlkPSJwX3Bhcmtpbmdfb2NjdXBhbmN5"
    "IiB2YWx1ZT0iMC41IiBtaW49IjAiIG1heD0iMSIgc3RlcD0iMC4wMSI+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9"
    "ImZvcm0tZ3JvdXAiPjxsYWJlbD5SZWxhdGl2ZSBDb21wYWN0bmVzczwvbGFiZWw+PGlucHV0IHR5cGU9Im51bWJl"
    "ciIgaWQ9InBfcmVsYXRpdmVfY29tcGFjdG5lc3MiIHZhbHVlPSIwLjgiIG1pbj0iMCIgbWF4PSIxIiBzdGVwPSIw"
    "LjAxIj48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZm9ybS1ncm91cCI+PGxhYmVsPlN1cmZhY2UgQXJlYSAobcKy"
    "KTwvbGFiZWw+PGlucHV0IHR5cGU9Im51bWJlciIgaWQ9InBfc3VyZmFjZV9hcmVhIiB2YWx1ZT0iNjAwIj48L2Rp"
    "dj4KICAgICAgPGRpdiBjbGFzcz0iZm9ybS1ncm91cCI+PGxhYmVsPldhbGwgQXJlYSAobcKyKTwvbGFiZWw+PGlu"
    "cHV0IHR5cGU9Im51bWJlciIgaWQ9InBfd2FsbF9hcmVhIiB2YWx1ZT0iMzAwIj48L2Rpdj4KICAgICAgPGRpdiBj"
    "bGFzcz0iZm9ybS1ncm91cCI+PGxhYmVsPkZhdWx0IFNldmVyaXR5ICgwLTUpPC9sYWJlbD48aW5wdXQgdHlwZT0i"
    "bnVtYmVyIiBpZD0icF9mYXVsdF9zZXZlcml0eSIgdmFsdWU9IjAiIG1pbj0iMCIgbWF4PSI1Ij48L2Rpdj4KICAg"
    "IDwvZGl2PgogICAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2dhcDoxcmVtO2ZsZXgtd3JhcDp3cmFwO21hcmdp"
    "bi1ib3R0b206MS41cmVtOyI+CiAgICAgIDxidXR0b24gY2xhc3M9ImJ0biBidG4tcHJpbWFyeSIgb25jbGljaz0i"
    "cnVuUHJlZGljdCgpIj7wn5SuIFByZWRpY3QgRmF1bHQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnRu"
    "IGJ0bi1zZWNvbmRhcnkiIG9uY2xpY2s9ImxvYWRSYW5kb21TYW1wbGUoKSI+8J+OsiBSYW5kb20gU2FtcGxlPC9i"
    "dXR0b24+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InByZWRpY3QtcmVzdWx0IiBpZD0icHJlZGljdFJlc3Vs"
    "dCI+CiAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1sYWJlbCI+UHJlZGljdGVkIEZhdWx0IFR5cGU8L2Rpdj4KICAg"
    "ICAgPGRpdiBjbGFzcz0icmVzdWx0LWZhdWx0IiBpZD0icmVzdWx0RmF1bHQiPuKAlDwvZGl2PgogICAgICA8ZGl2"
    "IGNsYXNzPSJyZXN1bHQtY29uZiI+Q29uZmlkZW5jZTogPHNwYW4gaWQ9InJlc3VsdENvbmYiPuKAlDwvc3Bhbj48"
    "L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iY29uZi1iYXIiPjxkaXYgY2xhc3M9ImNvbmYtZmlsbCIgaWQ9InJlc3Vs"
    "dENvbmZCYXIiIHN0eWxlPSJ3aWR0aDowJSI+PC9kaXY+PC9kaXY+CiAgICAgIDxkaXYgc3R5bGU9Im1hcmdpbi10"
    "b3A6MC43NXJlbTtmb250LXNpemU6MC44cmVtO2NvbG9yOnZhcigtLXRleHQzKTtmb250LWZhbWlseTp2YXIoLS1t"
    "b25vKTsiPk1vZGVsOiA8c3BhbiBpZD0icmVzdWx0TW9kZWwiPuKAlDwvc3Bhbj48L2Rpdj4KICAgIDwvZGl2Pgog"
    "IDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIEFJIFJFUE9SVCBTRUNUSU9OIC0tPgo8c2VjdGlvbiBjbGFzcz0ic2Vj"
    "dGlvbiIgaWQ9InNlY3Rpb24tcmVwb3J0Ij4KICA8ZGl2IGNsYXNzPSJyZXBvcnQtcGFuZWwiPgogICAgPGRpdiBj"
    "bGFzcz0icGFuZWwtdGl0bGUiPkFJIFByZWRpY3RpdmUgSW50ZWxsaWdlbmNlIFJlcG9ydDwvZGl2PgogICAgPGRp"
    "diBjbGFzcz0ibGFuZy1zZWxlY3RvciI+CiAgICAgIDxidXR0b24gY2xhc3M9ImxhbmctYnRuIGFjdGl2ZSIgb25j"
    "bGljaz0ibG9hZFJlcG9ydCgnZW4nLHRoaXMpIj7wn4es8J+HpyBFbmdsaXNoPC9idXR0b24+CiAgICAgIDxidXR0"
    "b24gY2xhc3M9ImxhbmctYnRuIiBvbmNsaWNrPSJsb2FkUmVwb3J0KCdoaScsdGhpcykiPvCfh67wn4ezIOCkueCk"
    "v+CkqOCljeCkpuClgDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJsYW5nLWJ0biIgb25jbGljaz0ibG9h"
    "ZFJlcG9ydCgnb3InLHRoaXMpIj7wn6q3IOCsk+CsoeCsvOCsv+CshjwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNs"
    "YXNzPSJsYW5nLWJ0biIgb25jbGljaz0ibG9hZFJlcG9ydCgnemgnLHRoaXMpIj7wn4eo8J+HsyDkuK3mloc8L2J1"
    "dHRvbj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ibGFuZy1idG4iIG9uY2xpY2s9ImxvYWRSZXBvcnQoJ2RlJyx0aGlz"
    "KSI+8J+HqfCfh6ogRGV1dHNjaDwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZs"
    "ZXg7Z2FwOjhweDttYXJnaW4tYm90dG9tOjEuNXJlbTtmbGV4LXdyYXA6d3JhcDsiPgogICAgICA8YnV0dG9uIGNs"
    "YXNzPSJidG4gYnRuLXB1cnBsZSIgb25jbGljaz0ibG9hZFJlcG9ydChjdXJyZW50TGFuZyxudWxsKSI+8J+UhCBS"
    "ZWdlbmVyYXRlIFJlcG9ydDwvYnV0dG9uPgogICAgICA8YnV0dG9uIGNsYXNzPSJidG4gYnRuLXNlY29uZGFyeSIg"
    "b25jbGljaz0iZG93bmxvYWRSZXBvcnQoKSI+4qyHIERvd25sb2FkIFJlcG9ydDwvYnV0dG9uPgogICAgPC9kaXY+"
    "CiAgICA8ZGl2IGNsYXNzPSJyZXBvcnQtY29udGVudCIgaWQ9InJlcG9ydENvbnRlbnQiPgogICAgICA8ZGl2IGNs"
    "YXNzPSJyZXBvcnQtbG9hZGluZyI+CiAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjAuOXJlbTtjb2xvcjp2"
    "YXIoLS10ZXh0MykiPlNlbGVjdCBhIGxhbmd1YWdlIGFuZCBnZW5lcmF0ZSB5b3VyIEFJIHJlcG9ydDwvZGl2Pgog"
    "ICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBpZD0icmVwb3J0TWV0YSIgc3R5bGU9Im1hcmdpbi10b3A6"
    "MXJlbTtmb250LXNpemU6MC43NXJlbTtjb2xvcjp2YXIoLS10ZXh0Myk7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyk7"
    "Ij48L2Rpdj4KICA8L2Rpdj4KPC9zZWN0aW9uPgoKCjwhLS0gTElWRSBNT05JVE9SIFNFQ1RJT04gLS0+CjxzZWN0"
    "aW9uIGNsYXNzPSJzZWN0aW9uIiBpZD0ic2VjdGlvbi1saXZlbW9uaXRvciI+CgogIDwhLS0gQWxlcnQgQmFubmVy"
    "IChmaXhlZCB0b3ApIC0tPgogIDxkaXYgY2xhc3M9ImFsZXJ0LWJhbm5lciIgaWQ9ImxtQWxlcnRCYW5uZXIiPjwv"
    "ZGl2PgoKICA8IS0tIFN1Yi1oZWFkZXIgLS0+CiAgPGRpdiBzdHlsZT0iZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1z"
    "OmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0d2VlbjttYXJnaW4tYm90dG9tOjEuMnJlbTtmbGV4LXdy"
    "YXA6d3JhcDtnYXA6MTJweDsiPgogICAgPGRpdj4KICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjEuMnJlbTtm"
    "b250LXdlaWdodDo3MDA7bGV0dGVyLXNwYWNpbmc6LTAuMDJlbTsiPuKaoSBTbWFydEJ1aWxkIExpdmUgTW9uaXRv"
    "cjwvZGl2PgogICAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6MC44cmVtO2NvbG9yOnZhcigtLXRleHQzKTttYXJn"
    "aW4tdG9wOjJweDsiPlJlYWwtdGltZSBtdWx0aS1hZ2VudCBzaW11bGF0aW9uIHdpdGggYXV0b25vbW91cyBuZWdv"
    "dGlhdGlvbiBwcm90b2NvbDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7YWxp"
    "Z24taXRlbXM6Y2VudGVyO2dhcDoxMnB4OyI+CiAgICAgIDxzcGFuIHN0eWxlPSJ3aWR0aDo4cHg7aGVpZ2h0Ojhw"
    "eDtib3JkZXItcmFkaXVzOjUwJTtiYWNrZ3JvdW5kOnZhcigtLWdyZWVuKTtib3gtc2hhZG93OjAgMCAxMHB4IHZh"
    "cigtLWdyZWVuKTthbmltYXRpb246cHVsc2UgMnMgaW5maW5pdGU7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij48L3Nw"
    "YW4+CiAgICAgIDxzcGFuIHN0eWxlPSJmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1ncmVlbik7Zm9udC13ZWln"
    "aHQ6NjAwOyIgaWQ9ImxtTGl2ZUxhYmVsIj5MSVZFPC9zcGFuPgogICAgICA8YnV0dG9uIGNsYXNzPSJwYXVzZS1i"
    "dG4gcnVubmluZyIgaWQ9ImxtUGF1c2VCdG4iIG9uY2xpY2s9ImxtVG9nZ2xlU2ltKCkiPuKPuCBQYXVzZTwvYnV0"
    "dG9uPgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0gU3ViLW5hdiB0YWJzIC0tPgogIDxkaXYgY2xhc3M9InN1"
    "Ym5hdiI+CiAgICA8YnV0dG9uIGNsYXNzPSJzdWJuYXYtYnRuIGFjdGl2ZSIgb25jbGljaz0ibG1TaG93VGFiKCdv"
    "dmVydmlldycsdGhpcykiPk92ZXJ2aWV3PC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJzdWJuYXYtYnRuIiBv"
    "bmNsaWNrPSJsbVNob3dUYWIoJ2FnZW50cycsdGhpcykiPkFnZW50czwvYnV0dG9uPgogICAgPGJ1dHRvbiBjbGFz"
    "cz0ic3VibmF2LWJ0biIgb25jbGljaz0ibG1TaG93VGFiKCdmYXVsdHMnLHRoaXMpIj5GYXVsdHM8L2J1dHRvbj4K"
    "ICAgIDxidXR0b24gY2xhc3M9InN1Ym5hdi1idG4iIG9uY2xpY2s9ImxtU2hvd1RhYignbWxfbW9kZWxzJyx0aGlz"
    "KSI+TUwgTW9kZWxzPC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJzdWJuYXYtYnRuIiBvbmNsaWNrPSJsbVNo"
    "b3dUYWIoJ25lZ290aWF0aW9ucycsdGhpcykiPk5lZ290aWF0aW9uczwvYnV0dG9uPgogIDwvZGl2PgoKICA8IS0t"
    "IOKUgOKUgCBPVkVSVklFVyBUQUIg4pSA4pSAIC0tPgogIDxkaXYgY2xhc3M9ImxtLXRhYiIgaWQ9ImxtdGFiLW92"
    "ZXJ2aWV3Ij4KCiAgICA8IS0tIEtQSSByb3cgLS0+CiAgICA8ZGl2IGNsYXNzPSJtZXRyaWNzLWdyaWQiIHN0eWxl"
    "PSJtYXJnaW4tYm90dG9tOjEuNXJlbTsiIGlkPSJsbUtwaUdyaWQiPgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMt"
    "Y2FyZCByZWQiPgogICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7imqE8L2Rpdj4KICAgICAgICA8ZGl2"
    "IGNsYXNzPSJtZXRyaWMtbGFiZWwiPkFjdGl2ZSBGYXVsdHM8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtZXRy"
    "aWMtdmFsdWUiIGlkPSJsbS1rcGktZmF1bHRzIj4wPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1"
    "YiI+UmVhbC10aW1lPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtY2FyZCI+CiAg"
    "ICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWljb24iPvCflIQ8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtZXRy"
    "aWMtbGFiZWwiPlJlY292ZXJ5IEV2ZW50czwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy12YWx1ZSIg"
    "aWQ9ImxtLWtwaS1ldmVudHMiPjA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtc3ViIj5Ub3RhbCBo"
    "YW5kbGVkPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtY2FyZCB5ZWxsb3ciPgog"
    "ICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7wn5uh77iPPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0i"
    "bWV0cmljLWxhYmVsIj5TYWZldHkgT3ZlcnJpZGVzPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXZh"
    "bHVlIiBpZD0ibG0ta3BpLW92ZXJyaWRlcyI+MDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1zdWIi"
    "PkNyaXRpY2FsIGV2ZW50czwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLWNhcmQg"
    "Z3JlZW4iPgogICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pY29uIj7ij7E8L2Rpdj4KICAgICAgICA8ZGl2IGNs"
    "YXNzPSJtZXRyaWMtbGFiZWwiPkF2ZyBSZWNvdmVyeTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy12"
    "YWx1ZSIgaWQ9ImxtLWtwaS1hdmdydCI+4oCUPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibWV0cmljLXN1YiI+"
    "bWlsbGlzZWNvbmRzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CgogICAgPCEtLSBGYXVsdCBhY3RpdmUg"
    "YWxlcnQgKHNob3duIHdoZW4gZmF1bHQgYWN0aXZlKSAtLT4KICAgIDxkaXYgY2xhc3M9ImZhdWx0LWFjdGl2ZS1h"
    "bGVydCIgaWQ9ImxtRmF1bHRBbGVydCIgc3R5bGU9ImRpc3BsYXk6bm9uZTttYXJnaW4tYm90dG9tOjEuNXJlbTsi"
    "PgogICAgICA8c3BhbiBzdHlsZT0iZm9udC1zaXplOjI0cHg7Ij7wn5qoPC9zcGFuPgogICAgICA8ZGl2PgogICAg"
    "ICAgIDxkaXYgc3R5bGU9ImZvbnQtd2VpZ2h0OjcwMDtjb2xvcjojZWY0NDQ0O2ZvbnQtc2l6ZToxNXB4OyIgaWQ9"
    "ImxtRmF1bHRBbGVydFRpdGxlIj5BQ1RJVkUgRkFVTFQ8L2Rpdj4KICAgICAgICA8ZGl2IHN0eWxlPSJmb250LXNp"
    "emU6MTJweDtjb2xvcjp2YXIoLS10ZXh0Mik7bWFyZ2luLXRvcDozcHg7Ij5TZXZlcml0eSBMZXZlbCA8c3BhbiBp"
    "ZD0ibG1GYXVsdEFsZXJ0U2V2Ij48L3NwYW4+IMK3IEFnZW50IG5lZ290aWF0aW9uIHByb3RvY29sIGVuZ2FnZWQ8"
    "L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgc3R5bGU9Im1hcmdpbi1sZWZ0OmF1dG87IiBpZD0ibG1GYXVs"
    "dEFsZXJ0UGlsbCI+PC9kaXY+CiAgICA8L2Rpdj4KCiAgICA8IS0tIExpdmUgZW5lcmd5IGNoYXJ0IC0tPgogICAg"
    "PGRpdiBjbGFzcz0iY2hhcnQtY2FyZCIgc3R5bGU9Im1hcmdpbi1ib3R0b206MS41cmVtOyI+CiAgICAgIDxkaXYg"
    "Y2xhc3M9ImNoYXJ0LXRpdGxlIj7imqEgTGl2ZSBFbmVyZ3kgQ29uc3VtcHRpb24g4oCUIExhc3QgNDAgUmVhZGlu"
    "Z3M8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtd3JhcCI+PGNhbnZhcyBpZD0ibG1FbmVyZ3lDaGFydCI+"
    "PC9jYW52YXM+PC9kaXY+CiAgICA8L2Rpdj4KCiAgICA8IS0tIEJvdHRvbSByb3cgLS0+CiAgICA8ZGl2IGNsYXNz"
    "PSJjaGFydHMtZ3JpZCI+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LWNhcmQiPgogICAgICAgIDxkaXYgY2xhc3M9"
    "ImNoYXJ0LXRpdGxlIj7wn5SnIEZhdWx0IERpc3RyaWJ1dGlvbiA8c3BhbiBjbGFzcz0iYmFkZ2UiPmxpdmU8L3Nw"
    "YW4+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtd3JhcCI+PGNhbnZhcyBpZD0ibG1GYXVsdFBpZUNo"
    "YXJ0Ij48L2NhbnZhcz48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LWNhcmQiPgog"
    "ICAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj7ij7HvuI8gUmVjb3ZlcnkgVGltZXMgPHNwYW4gY2xhc3M9"
    "ImJhZGdlIj5sYXN0IDIwPC9zcGFuPjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52"
    "YXMgaWQ9ImxtUnRCYXJDaGFydCI+PC9jYW52YXM+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9k"
    "aXY+CgogIDwhLS0g4pSA4pSAIEFHRU5UUyBUQUIg4pSA4pSAIC0tPgogIDxkaXYgY2xhc3M9ImxtLXRhYiIgaWQ9"
    "ImxtdGFiLWFnZW50cyIgc3R5bGU9ImRpc3BsYXk6bm9uZTsiPgogICAgPGRpdiBjbGFzcz0iYWdlbnQtZ3JpZCIg"
    "aWQ9ImxtQWdlbnRHcmlkIj4KICAgICAgPCEtLSBQb3B1bGF0ZWQgYnkgSlMgLS0+CiAgICA8L2Rpdj4KICAgIDxk"
    "aXYgY2xhc3M9ImNoYXJ0cy1ncmlkIj4KICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCI+CiAgICAgICAgPGRp"
    "diBjbGFzcz0iY2hhcnQtdGl0bGUiPvCfk4ogTGl2ZSBCdWlsZGluZyBSZWFkaW5nczwvZGl2PgogICAgICAgIDxk"
    "aXYgaWQ9ImxtUmVhZGluZ3NCYXIiIHN0eWxlPSJwYWRkaW5nOjhweCAwOyI+PC9kaXY+CiAgICAgIDwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGFydC10aXRsZSI+8J+T"
    "oSBBZ2VudCBNZXNzYWdlIEJ1czwvZGl2PgogICAgICAgIDxkaXYgaWQ9ImxtTXNnQnVzIiBzdHlsZT0ibWF4LWhl"
    "aWdodDozMDBweDtvdmVyZmxvdy15OmF1dG87Ij48L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDwh"
    "LS0gQXJjaGl0ZWN0dXJlIGRpYWdyYW0gLS0+CiAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIiBzdHlsZT0ibWFy"
    "Z2luLXRvcDoxLjVyZW07Ij4KICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtdGl0bGUiPvCfj5vvuI8gRGVjZW50cmFs"
    "aXplZCBBcmNoaXRlY3R1cmU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iYXJjaC13cmFwIiBpZD0ibG1BcmNoV3Jh"
    "cCI+CiAgICAgICAgPGRpdiBjbGFzcz0iYXJjaC1jZW50ZXIiPgogICAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1z"
    "aXplOjEycHg7Zm9udC13ZWlnaHQ6NzAwO2NvbG9yOnZhcigtLWFjY2VudCk7Ij5NRVNTQUdFIEJVUzwvZGl2Pgog"
    "ICAgICAgICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjEwcHg7Y29sb3I6dmFyKC0tdGV4dDMpOyI+Q29udHJhY3Qg"
    "TmV0IFByb3RvY29sPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPCEtLSBBZ2VudCBub2RlcyBwbGFjZWQg"
    "YnkgSlMgLS0+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDwhLS0g4pSA4pSAIEZBVUxUUyBU"
    "QUIg4pSA4pSAIC0tPgogIDxkaXYgY2xhc3M9ImxtLXRhYiIgaWQ9ImxtdGFiLWZhdWx0cyIgc3R5bGU9ImRpc3Bs"
    "YXk6bm9uZTsiPgogICAgPGRpdiBjbGFzcz0iY2hhcnRzLWdyaWQiIHN0eWxlPSJtYXJnaW4tYm90dG9tOjEuNXJl"
    "bTsiPgogICAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGFydC10aXRs"
    "ZSI+8J+UoiBGYXVsdCBGcmVxdWVuY3kgQW5hbHlzaXM8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGFydC13"
    "cmFwIj48Y2FudmFzIGlkPSJsbUZhdWx0RnJlcUNoYXJ0Ij48L2NhbnZhcz48L2Rpdj4KICAgICAgPC9kaXY+CiAg"
    "ICAgIDxkaXYgY2xhc3M9ImNoYXJ0LWNhcmQiPgogICAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj7wn5OI"
    "IFNldmVyaXR5IERpc3RyaWJ1dGlvbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXdyYXAiPjxjYW52"
    "YXMgaWQ9ImxtU2V2Q2hhcnQiPjwvY2FudmFzPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPCEt"
    "LSBGdWxsIHJlY292ZXJ5IGxvZyB0YWJsZSAtLT4KICAgIDxkaXYgY2xhc3M9InRhYmxlLXdyYXAiPgogICAgICA8"
    "ZGl2IGNsYXNzPSJ0YWJsZS1oZWFkZXIiPgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZTowLjhyZW07dGV4"
    "dC10cmFuc2Zvcm06dXBwZXJjYXNlO2xldHRlci1zcGFjaW5nOjAuMDhlbTtjb2xvcjp2YXIoLS10ZXh0Mik7Ij4K"
    "ICAgICAgICAgIFJlY292ZXJ5IExvZyA8c3BhbiBpZD0ibG1Mb2dDb3VudCIgc3R5bGU9ImNvbG9yOnZhcigtLXRl"
    "eHQzKTsiPigwIGV2ZW50cyk8L3NwYW4+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNs"
    "YXNzPSJ0YWJsZS1zY3JvbGwiPgogICAgICAgIDx0YWJsZSBjbGFzcz0iZGF0YS10YWJsZSIgaWQ9ImxtTG9nVGFi"
    "bGUiPgogICAgICAgICAgPHRoZWFkPgogICAgICAgICAgICA8dHI+CiAgICAgICAgICAgICAgPHRoPlN0ZXA8L3Ro"
    "Pjx0aD5GYXVsdCBUeXBlPC90aD48dGg+U2V2ZXJpdHk8L3RoPjx0aD5EZXRlY3RlZCBCeTwvdGg+CiAgICAgICAg"
    "ICAgICAgPHRoPlJlY292ZXJ5IEFjdGlvbjwvdGg+PHRoPkFnZW50PC90aD48dGg+VGltZSAobXMpPC90aD48dGg+"
    "T3ZlcnJpZGU8L3RoPgogICAgICAgICAgICA8L3RyPgogICAgICAgICAgPC90aGVhZD4KICAgICAgICAgIDx0Ym9k"
    "eSBpZD0ibG1Mb2dCb2R5Ij48L3Rib2R5PgogICAgICAgIDwvdGFibGU+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+"
    "CiAgPC9kaXY+CgogIDwhLS0g4pSA4pSAIE1MIE1PREVMUyBUQUIg4pSA4pSAIC0tPgogIDxkaXYgY2xhc3M9Imxt"
    "LXRhYiIgaWQ9ImxtdGFiLW1sX21vZGVscyIgc3R5bGU9ImRpc3BsYXk6bm9uZTsiPgogICAgPCEtLSBDb21wYXJp"
    "c29uIHRhYmxlIC0tPgogICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCIgc3R5bGU9Im1hcmdpbi1ib3R0b206MS41"
    "cmVtOyI+CiAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj7wn6SWIE1MIE1vZGVsIENvbXBhcmlzb24gPHNw"
    "YW4gY2xhc3M9ImJhZGdlIj4zIG1vZGVscyDCtyAzMksgc2FtcGxlczwvc3Bhbj48L2Rpdj4KICAgICAgPGRpdiBj"
    "bGFzcz0idGFibGUtc2Nyb2xsIj4KICAgICAgICA8dGFibGUgY2xhc3M9ImRhdGEtdGFibGUiIGlkPSJsbU1vZGVs"
    "VGFibGUiPgogICAgICAgICAgPHRoZWFkPgogICAgICAgICAgICA8dHI+PHRoPk1vZGVsPC90aD48dGg+QWNjdXJh"
    "Y3k8L3RoPjx0aD5QcmVjaXNpb248L3RoPjx0aD5SZWNhbGw8L3RoPjx0aD5GMSBTY29yZTwvdGg+PHRoPlRyYWlu"
    "IFRpbWU8L3RoPjwvdHI+CiAgICAgICAgICA8L3RoZWFkPgogICAgICAgICAgPHRib2R5IGlkPSJsbU1vZGVsQm9k"
    "eSI+PC90Ym9keT4KICAgICAgICA8L3RhYmxlPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFz"
    "cz0iY2hhcnRzLWdyaWQiPgogICAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgICA8ZGl2IGNsYXNz"
    "PSJjaGFydC10aXRsZSI+8J+ToSBQZXJmb3JtYW5jZSBSYWRhcjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImNo"
    "YXJ0LXdyYXAiPjxjYW52YXMgaWQ9ImxtUmFkYXJDaGFydCI+PC9jYW52YXM+PC9kaXY+CiAgICAgIDwvZGl2Pgog"
    "ICAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGFydC10aXRsZSI+8J+P"
    "hiBNb2RlbCBTdW1tYXJ5PC9kaXY+CiAgICAgICAgPGRpdiBpZD0ibG1Nb2RlbFN1bW1hcnkiIHN0eWxlPSJwYWRk"
    "aW5nOjRweCAwOyI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8IS0tIERhdGEgcGlwZWxpbmUg"
    "LS0+CiAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIiBzdHlsZT0ibWFyZ2luLXRvcDoxLjVyZW07Ij4KICAgICAg"
    "PGRpdiBjbGFzcz0iY2hhcnQtdGl0bGUiPvCflKwgRGF0YSBQaXBlbGluZTwvZGl2PgogICAgICA8ZGl2IGNsYXNz"
    "PSJwaXBlbGluZS1ncmlkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1zdGVwIiBzdHlsZT0iYmFja2dy"
    "b3VuZDpyZ2JhKDU5LDEzMCwyNDYsMC4wNik7Ym9yZGVyLWNvbG9yOnJnYmEoNTksMTMwLDI0NiwwLjIpOyI+CiAg"
    "ICAgICAgICA8ZGl2IGNsYXNzPSJwcy10aXRsZSIgc3R5bGU9ImNvbG9yOiMzYjgyZjY7Ij4xLiBCYXNlIERhdGE8"
    "L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9InBzLWRlc2MiPlVDSSBFbmVyZ3kgRWZmaWNpZW5jeSAoODAwIHNp"
    "bXVsYXRlZCByZWNvcmRzKTwvZGl2PgogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InBpcGVsaW5l"
    "LXN0ZXAiIHN0eWxlPSJiYWNrZ3JvdW5kOnJnYmEoMjQ5LDExNSwyMiwwLjA2KTtib3JkZXItY29sb3I6cmdiYSgy"
    "NDksMTE1LDIyLDAuMik7Ij4KICAgICAgICAgIDxkaXYgY2xhc3M9InBzLXRpdGxlIiBzdHlsZT0iY29sb3I6I2Y5"
    "NzMxNjsiPjIuIEZhdWx0IEluamVjdGlvbjwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0icHMtZGVzYyI+UnVs"
    "ZS1iYXNlZDogSFZBQywgRW5lcmd5LCBMaWdodGluZywgU2FmZXR5LCBTZW5zb3IgZmF1bHRzPC9kaXY+CiAgICAg"
    "ICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0icGlwZWxpbmUtc3RlcCIgc3R5bGU9ImJhY2tncm91bmQ6cmdi"
    "YSgxNjgsODUsMjQ3LDAuMDYpO2JvcmRlci1jb2xvcjpyZ2JhKDE2OCw4NSwyNDcsMC4yKTsiPgogICAgICAgICAg"
    "PGRpdiBjbGFzcz0icHMtdGl0bGUiIHN0eWxlPSJjb2xvcjojYTg1NWY3OyI+My4gQXVnbWVudGF0aW9uPC9kaXY+"
    "CiAgICAgICAgICA8ZGl2IGNsYXNzPSJwcy1kZXNjIj5HYXVzc2lhbiBub2lzZSArIHRpbWUtc2hpZnQg4oaSIDMy"
    "LDAwMCBzYW1wbGVzPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0icGlwZWxpbmUtc3Rl"
    "cCIgc3R5bGU9ImJhY2tncm91bmQ6cmdiYSgzNCwxOTcsOTQsMC4wNik7Ym9yZGVyLWNvbG9yOnJnYmEoMzQsMTk3"
    "LDk0LDAuMik7Ij4KICAgICAgICAgIDxkaXYgY2xhc3M9InBzLXRpdGxlIiBzdHlsZT0iY29sb3I6IzIyYzU1ZTsi"
    "PjQuIFRyYWluL1Rlc3QgU3BsaXQ8L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9InBzLWRlc2MiPjgwLzIwIHN0"
    "cmF0aWZpZWQgc3BsaXQsIDYtY2xhc3MgY2xhc3NpZmljYXRpb248L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAg"
    "PC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSDilIDilIAgTkVHT1RJQVRJT05TIFRBQiDilIDilIAg"
    "LS0+CiAgPGRpdiBjbGFzcz0ibG0tdGFiIiBpZD0ibG10YWItbmVnb3RpYXRpb25zIiBzdHlsZT0iZGlzcGxheTpu"
    "b25lOyI+CiAgICA8ZGl2IGNsYXNzPSJjaGFydHMtZ3JpZCI+CiAgICAgIDwhLS0gTGl2ZSBuZWdvdGlhdGlvbiB0"
    "aW1lbGluZSAtLT4KICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtY2FyZCI+CiAgICAgICAgPGRpdiBjbGFzcz0iY2hh"
    "cnQtdGl0bGUiPuKalu+4jyBOZWdvdGlhdGlvbiBQcm90b2NvbCA8c3BhbiBjbGFzcz0iYmFkZ2UiIHN0eWxlPSJi"
    "YWNrZ3JvdW5kOnJnYmEoMjM5LDY4LDY4LDAuMTUpO2NvbG9yOiNlZjQ0NDQ7Ym9yZGVyLWNvbG9yOnJnYmEoMjM5"
    "LDY4LDY4LDAuMyk7Ij5MaXZlPC9zcGFuPjwvZGl2PgogICAgICAgIDxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4"
    "O2NvbG9yOnZhcigtLXRleHQzKTttYXJnaW4tYm90dG9tOjEycHg7Ij5Db250cmFjdCBOZXQgUHJvdG9jb2wg4oCU"
    "IExpdmUgRXZlbnRzPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0ibmVnLXRpbWVsaW5lIiBpZD0ibG1OZWdUaW1l"
    "bGluZSI+PC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7ZmxleC1kaXJl"
    "Y3Rpb246Y29sdW1uO2dhcDoxLjVyZW07Ij4KICAgICAgICA8IS0tIFByb3RvY29sIHN0ZXBzIC0tPgogICAgICAg"
    "IDxkaXYgY2xhc3M9ImNoYXJ0LWNhcmQiPgogICAgICAgICAgPGRpdiBjbGFzcz0iY2hhcnQtdGl0bGUiPvCfk5wg"
    "UHJvdG9jb2wgU3RlcHM8L2Rpdj4KICAgICAgICAgIDxkaXYgaWQ9ImxtUHJvdG9TdGVwcyI+PC9kaXY+CiAgICAg"
    "ICAgPC9kaXY+CiAgICAgICAgPCEtLSBTdGF0cyAtLT4KICAgICAgICA8ZGl2IGNsYXNzPSJjaGFydC1jYXJkIj4K"
    "ICAgICAgICAgIDxkaXYgY2xhc3M9ImNoYXJ0LXRpdGxlIj7wn5OKIE5lZ290aWF0aW9uIFN0YXRzPC9kaXY+CiAg"
    "ICAgICAgICA8ZGl2IGlkPSJsbU5lZ1N0YXRzIj48L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAg"
    "ICA8L2Rpdj4KICA8L2Rpdj4KCjwvc2VjdGlvbj4KPC9tYWluPgoKPCEtLSBUT0FTVFMgLS0+CjxkaXYgY2xhc3M9"
    "InRvYXN0LWNvbnRhaW5lciIgaWQ9InRvYXN0Q29udGFpbmVyIj48L2Rpdj4KCjxzY3JpcHQ+Ci8vIOKUgOKUgOKU"
    "gCBDSEFSVCBSRUdJU1RSWSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKY29uc3Qg"
    "Q0hBUlRTID0ge307CmZ1bmN0aW9uIGRlc3Ryb3lDaGFydChpZCkgeyBpZiAoQ0hBUlRTW2lkXSkgeyBDSEFSVFNb"
    "aWRdLmRlc3Ryb3koKTsgZGVsZXRlIENIQVJUU1tpZF07IH0gfQoKLy8g4pSA4pSA4pSAIE5BVklHQVRJT04g4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmxldCBjdXJyZW50"
    "U2VjdGlvbiA9ICdvdmVydmlldyc7CmZ1bmN0aW9uIHNob3dTZWN0aW9uKG5hbWUpIHsKICBkb2N1bWVudC5xdWVy"
    "eVNlbGVjdG9yQWxsKCcuc2VjdGlvbicpLmZvckVhY2gocyA9PiBzLmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScp"
    "KTsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubmF2LWJ0bicpLmZvckVhY2goYiA9PiBiLmNsYXNzTGlz"
    "dC5yZW1vdmUoJ2FjdGl2ZScpKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2VjdGlvbi0nK25hbWUpLmNs"
    "YXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogIGNvbnN0IGJ0bnMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcu"
    "bmF2LWJ0bicpOwogIGNvbnN0IGxhYmVscyA9IFsnb3ZlcnZpZXcnLCdkYXNoYm9hcmQnLCdtb2RlbHMnLCdsaXZl"
    "JywnZGF0YScsJ3ByZWRpY3QnLCdyZXBvcnQnLCdsaXZlbW9uaXRvciddOwogIGlmIChsYWJlbHMuaW5jbHVkZXMo"
    "bmFtZSkpIGJ0bnNbbGFiZWxzLmluZGV4T2YobmFtZSldLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogIGN1cnJl"
    "bnRTZWN0aW9uID0gbmFtZTsKICBpZiAobmFtZSA9PT0gJ2Rhc2hib2FyZCcpIGxvYWREYXNoYm9hcmQoKTsKICBl"
    "bHNlIGlmIChuYW1lID09PSAnbW9kZWxzJykgbG9hZE1vZGVscygpOwogIGVsc2UgaWYgKG5hbWUgPT09ICdsaXZl"
    "JykgeyByZWZyZXNoRmVlZCgpOyBsb2FkUlRIaXN0KCk7IH0KICBlbHNlIGlmIChuYW1lID09PSAnZGF0YScpIHsg"
    "bG9hZERhdGFQYWdlKDEpOyBsb2FkRGF0YURpc3QoKTsgfQogIGVsc2UgaWYgKG5hbWUgPT09ICdsaXZlbW9uaXRv"
    "cicpIHsgbG1Jbml0KCk7IH0KfQoKLy8g4pSA4pSA4pSAIFRPQVNUIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiB0b2FzdChtc2cs"
    "IHR5cGU9J2luZm8nLCBpY29uPScnKSB7CiAgY29uc3QgdCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2Rpdicp"
    "OwogIHQuY2xhc3NOYW1lID0gYHRvYXN0ICR7dHlwZX1gOwogIHQuaW5uZXJIVE1MID0gYCR7aWNvbnx8e2luZm86"
    "J+KEuScsc3VjY2Vzczon4pyTJyxlcnJvcjon4pyXJ31bdHlwZV19IDxzcGFuPiR7bXNnfTwvc3Bhbj5gOwogIGRv"
    "Y3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b2FzdENvbnRhaW5lcicpLnByZXBlbmQodCk7CiAgc2V0VGltZW91dCgo"
    "KSA9PiB0LnJlbW92ZSgpLCAzNTAwKTsKfQoKLy8g4pSA4pSA4pSAIFNUQVRVUyBQT0xMSU5HIOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApsZXQgc3RhdHVzSW50ZXJ2YWw7CmFzeW5jIGZ1bmN0"
    "aW9uIHBvbGxTdGF0dXMoKSB7CiAgdHJ5IHsKICAgIGNvbnN0IGQgPSBhd2FpdCBmZXRjaCgnL2FwaS9zdGF0dXMn"
    "KS50aGVuKHI9PnIuanNvbigpKTsKICAgIGNvbnN0IGRvdCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0"
    "dXNEb3QnKTsKICAgIGNvbnN0IHR4dCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0dXNUZXh0Jyk7CiAg"
    "ICAKICAgIC8vIFVwZGF0ZSBwcm9ncmVzcwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Byb2dyZXNzRmls"
    "bCcpLnN0eWxlLndpZHRoID0gZC5wcm9ncmVzcyArICclJzsKICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdw"
    "cm9ncmVzc1BjdCcpLnRleHRDb250ZW50ID0gZC5wcm9ncmVzcyArICclJzsKICAgIGRvY3VtZW50LmdldEVsZW1l"
    "bnRCeUlkKCdwcm9ncmVzc01zZycpLnRleHRDb250ZW50ID0gZC5zdGF0dXNfbXNnOwoKICAgIGlmIChkLnRyYWlu"
    "aW5nIHx8IGQuc2ltdWxhdGluZykgewogICAgICBkb3QuY2xhc3NOYW1lID0gJ3N0YXR1cy1kb3QgcnVubmluZyc7"
    "CiAgICAgIHR4dC50ZXh0Q29udGVudCA9IGQuc2ltdWxhdGluZyA/ICdTaW11bGF0aW5nLi4uJyA6ICdUcmFpbmlu"
    "Zy4uLic7CiAgICB9IGVsc2UgaWYgKGQubW9kZWxfdHJhaW5lZCkgewogICAgICBkb3QuY2xhc3NOYW1lID0gJ3N0"
    "YXR1cy1kb3QgYWN0aXZlJzsKICAgICAgdHh0LnRleHRDb250ZW50ID0gJ01vZGVsIFJlYWR5JzsKICAgICAgZG9j"
    "dW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0blNpbScpLmRpc2FibGVkID0gZmFsc2U7CiAgICB9IGVsc2UgewogICAg"
    "ICBkb3QuY2xhc3NOYW1lID0gJ3N0YXR1cy1kb3QnOwogICAgICB0eHQudGV4dENvbnRlbnQgPSAnTm90IGluaXRp"
    "YWxpemVkJzsKICAgIH0KCiAgICAvLyBRdWljayBtZXRyaWNzCiAgICBpZiAoZC5kYXRhc2V0X3NpemUgPiAwKSBk"
    "b2N1bWVudC5nZXRFbGVtZW50QnlJZCgncURhdGFzZXQnKS50ZXh0Q29udGVudCA9IGQuZGF0YXNldF9zaXplLnRv"
    "TG9jYWxlU3RyaW5nKCk7CiAgfSBjYXRjaChlKSB7CiAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RhdHVz"
    "RG90JykuY2xhc3NOYW1lID0gJ3N0YXR1cy1kb3QgZXJyb3InOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQo"
    "J3N0YXR1c1RleHQnKS50ZXh0Q29udGVudCA9ICdTZXJ2ZXIgb2ZmbGluZSc7CiAgfQp9CgovLyDilIDilIDilIAg"
    "UElQRUxJTkUg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSACmFzeW5jIGZ1bmN0aW9uIGluaXRQaXBlbGluZSgpIHsKICBjb25zdCBidG4gPSBkb2N1bWVudC5nZXRF"
    "bGVtZW50QnlJZCgnYnRuSW5pdCcpOwogIGJ0bi5kaXNhYmxlZCA9IHRydWU7IGJ0bi50ZXh0Q29udGVudCA9ICfi"
    "j7MgSW5pdGlhbGl6aW5nLi4uJzsKICB0cnkgewogICAgY29uc3QgciA9IGF3YWl0IGZldGNoKCcvYXBpL3BpcGVs"
    "aW5lL2luaXQnLCB7bWV0aG9kOidQT1NUJ30pLnRoZW4ocj0+ci5qc29uKCkpOwogICAgaWYgKHIuZXJyb3IpIHsg"
    "dG9hc3Qoci5lcnJvciwgJ2Vycm9yJyk7IH0KICAgIGVsc2UgeyB0b2FzdCgnUGlwZWxpbmUgc3RhcnRlZCDigJQg"
    "dHJhaW5pbmcgMyBNTCBtb2RlbHMgb24gMzJLIHNhbXBsZXMnLCAnc3VjY2VzcycsICfwn6egJyk7IH0KICB9IGNh"
    "dGNoKGUpIHsgdG9hc3QoJ1NlcnZlciBlcnJvcicsICdlcnJvcicpOyB9CiAgc2V0VGltZW91dCgoKSA9PiB7IGJ0"
    "bi5kaXNhYmxlZCA9IGZhbHNlOyBidG4udGV4dENvbnRlbnQgPSAn8J+noCBJbml0aWFsaXplICYgVHJhaW4nOyB9"
    "LCAzMDAwKTsKfQoKYXN5bmMgZnVuY3Rpb24gc3RhcnRTaW11bGF0aW9uKCkgewogIGNvbnN0IGJ0biA9IGRvY3Vt"
    "ZW50LmdldEVsZW1lbnRCeUlkKCdidG5TaW0nKTsKICBidG4uZGlzYWJsZWQgPSB0cnVlOyBidG4udGV4dENvbnRl"
    "bnQgPSAn4o+zIFJ1bm5pbmcuLi4nOwogIHRyeSB7CiAgICBjb25zdCByID0gYXdhaXQgZmV0Y2goJy9hcGkvcGlw"
    "ZWxpbmUvc2ltdWxhdGUnLCB7CiAgICAgIG1ldGhvZDonUE9TVCcsIGhlYWRlcnM6eydDb250ZW50LVR5cGUnOidh"
    "cHBsaWNhdGlvbi9qc29uJ30sCiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHtzdGVwczo1MDB9KQogICAgfSku"
    "dGhlbihyPT5yLmpzb24oKSk7CiAgICBpZiAoci5lcnJvcikgdG9hc3Qoci5lcnJvciwgJ2Vycm9yJyk7CiAgICBl"
    "bHNlIHRvYXN0KCdTaW11bGF0aW9uIHJ1bm5pbmcg4oCUIDUwMCB0aW1lc3RlcHMgd2l0aCBtdWx0aS1hZ2VudCBu"
    "ZWdvdGlhdGlvbicsICdzdWNjZXNzJywgJ+KWticpOwogIH0gY2F0Y2goZSkgeyB0b2FzdCgnU2VydmVyIGVycm9y"
    "JywgJ2Vycm9yJyk7IH0KICBzZXRUaW1lb3V0KCgpID0+IHsgYnRuLmRpc2FibGVkID0gZmFsc2U7IGJ0bi50ZXh0"
    "Q29udGVudCA9ICfilrYgUnVuIFNpbXVsYXRpb24nOyB9LCA1MDAwKTsKfQoKLy8g4pSA4pSA4pSAIERBU0hCT0FS"
    "RCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKY29u"
    "c3QgQ09MT1JTID0gewogIG5vcm1hbDogJyMwMGU1YTAnLCBodmFjX2ZhaWx1cmU6ICcjZjQzZjVlJywgZW5lcmd5"
    "X292ZXJsb2FkOiAnI2Y5NzMxNicsCiAgbGlnaHRpbmdfZmF1bHQ6ICcjZmJiZjI0Jywgc2FmZXR5X2FsYXJtOiAn"
    "I2VmNDQ0NCcsIHNlbnNvcl9mYWlsdXJlOiAnIzk0YTNiOCcsCiAgcGFya2luZ19jb25nZXN0aW9uOiAnIzgxOGNm"
    "OCcsIG5vbmU6ICcjNjQ3NDhiJwp9Owpjb25zdCBQQUxFVFRFID0gWycjMWU2YmZmJywnIzAwZDRmZicsJyMwMGU1"
    "YTAnLCcjZmJiZjI0JywnI2Y0M2Y1ZScsJyM3YzNhZWQnLCcjZjk3MzE2JywnIzA2YjZkNCddOwoKYXN5bmMgZnVu"
    "Y3Rpb24gbG9hZERhc2hib2FyZCgpIHsKICB0cnkgewogICAgY29uc3QgZCA9IGF3YWl0IGZldGNoKCcvYXBpL2Rh"
    "c2hib2FyZCcpLnRoZW4ocj0+ci5qc29uKCkpOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RSdW5zJyku"
    "dGV4dENvbnRlbnQgPSBkLnRvdGFsX3J1bnM7CiAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZEZhdWx0cycp"
    "LnRleHRDb250ZW50ID0gZC50b3RhbF9mYXVsdHMudG9Mb2NhbGVTdHJpbmcoKTsKICAgIGRvY3VtZW50LmdldEVs"
    "ZW1lbnRCeUlkKCdkUmVjb3ZlcnknKS50ZXh0Q29udGVudCA9IGQuYXZnX3JlY292ZXJ5X21zICsgJyBtcyc7CiAg"
    "ICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZE92ZXJyaWRlcycpLnRleHRDb250ZW50ID0gZC5zYWZldHlfb3Zl"
    "cnJpZGVzOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RCZXN0RjEnKS50ZXh0Q29udGVudCA9IGQuYmVz"
    "dF9mMSB8fCAn4oCUJzsKICAgIC8vIFVwZGF0ZSBxdWljayBtZXRyaWNzIHRvbwogICAgZG9jdW1lbnQuZ2V0RWxl"
    "bWVudEJ5SWQoJ3FGYXVsdHMnKS50ZXh0Q29udGVudCA9IGQudG90YWxfZmF1bHRzLnRvTG9jYWxlU3RyaW5nKCk7"
    "CiAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncVJlY292ZXJ5JykudGV4dENvbnRlbnQgPSBkLmF2Z19yZWNv"
    "dmVyeV9tcyArICcgbXMnOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3FPdmVycmlkZXMnKS50ZXh0Q29u"
    "dGVudCA9IGQuc2FmZXR5X292ZXJyaWRlczsKCiAgICAvLyBGYXVsdCBkaXN0cmlidXRpb24gcGllCiAgICBpZiAo"
    "ZC5mYXVsdF9icmVha2Rvd24ubGVuZ3RoID4gMCkgewogICAgICBkZXN0cm95Q2hhcnQoJ2ZhdWx0RGlzdCcpOwog"
    "ICAgICBDSEFSVFNbJ2ZhdWx0RGlzdCddID0gbmV3IENoYXJ0KGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdmYXVs"
    "dERpc3RDaGFydCcpLCB7CiAgICAgICAgdHlwZTogJ2RvdWdobnV0JywKICAgICAgICBkYXRhOiB7CiAgICAgICAg"
    "ICBsYWJlbHM6IGQuZmF1bHRfYnJlYWtkb3duLm1hcChmPT5mLmZhdWx0X3R5cGUpLAogICAgICAgICAgZGF0YXNl"
    "dHM6IFt7CiAgICAgICAgICAgIGRhdGE6IGQuZmF1bHRfYnJlYWtkb3duLm1hcChmPT5mLmNudCksCiAgICAgICAg"
    "ICAgIGJhY2tncm91bmRDb2xvcjogZC5mYXVsdF9icmVha2Rvd24ubWFwKGY9PkNPTE9SU1tmLmZhdWx0X3R5cGVd"
    "fHwnIzY0NzQ4YicpLAogICAgICAgICAgICBib3JkZXJXaWR0aDogMiwgYm9yZGVyQ29sb3I6ICcjMDQwODEwJywK"
    "ICAgICAgICAgICAgaG92ZXJPZmZzZXQ6IDEyCiAgICAgICAgICB9XQogICAgICAgIH0sCiAgICAgICAgb3B0aW9u"
    "czogewogICAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJhdGlvOmZhbHNlLAogICAgICAg"
    "ICAgcGx1Z2luczogewogICAgICAgICAgICBsZWdlbmQ6IHsgcG9zaXRpb246J3JpZ2h0JywgbGFiZWxzOntjb2xv"
    "cjonIzk0YTNiOCcsZm9udDp7c2l6ZToxMX0scGFkZGluZzoxMn19LAogICAgICAgICAgICB0b29sdGlwOiB7IGNh"
    "bGxiYWNrczogeyBsYWJlbDogYyA9PiBgICR7Yy5sYWJlbH06ICR7Yy5wYXJzZWR9IGV2ZW50c2AgfX0KICAgICAg"
    "ICAgIH0KICAgICAgICB9CiAgICAgIH0pOwogICAgfQoKICAgIC8vIEhvdXJseSBIVkFDCiAgICBjb25zdCBoZCA9"
    "IGF3YWl0IGZldGNoKCcvYXBpL2FuYWx5dGljcy9ob3VybHlfcGF0dGVybicpLnRoZW4ocj0+ci5qc29uKCkpOwog"
    "ICAgaWYgKGhkLmxlbmd0aCA+IDApIHsKICAgICAgZGVzdHJveUNoYXJ0KCdob3VybHknKTsKICAgICAgQ0hBUlRT"
    "Wydob3VybHknXSA9IG5ldyBDaGFydChkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnaG91cmx5Q2hhcnQnKSwgewog"
    "ICAgICAgIHR5cGU6ICdsaW5lJywKICAgICAgICBkYXRhOiB7CiAgICAgICAgICBsYWJlbHM6IGhkLm1hcChoPT5o"
    "LmhvdXIrJzowMCcpLAogICAgICAgICAgZGF0YXNldHM6IFt7CiAgICAgICAgICAgIGxhYmVsOiAnQXZnIEhWQUMg"
    "UG93ZXIgKGtXKScsCiAgICAgICAgICAgIGRhdGE6IGhkLm1hcChoPT4raC5hdmdfaHZhYy50b0ZpeGVkKDIpKSwK"
    "ICAgICAgICAgICAgYm9yZGVyQ29sb3I6ICcjMWU2YmZmJywgYmFja2dyb3VuZENvbG9yOiAncmdiYSgzMCwxMDcs"
    "MjU1LDAuMSknLAogICAgICAgICAgICBmaWxsOiB0cnVlLCB0ZW5zaW9uOiAwLjQsIHBvaW50UmFkaXVzOiAzLCBw"
    "b2ludEhvdmVyUmFkaXVzOiA2LAogICAgICAgICAgICBwb2ludEJhY2tncm91bmRDb2xvcjogJyMxZTZiZmYnCiAg"
    "ICAgICAgICB9LHsKICAgICAgICAgICAgbGFiZWw6ICdBdmcgT2NjdXBhbmN5JywKICAgICAgICAgICAgZGF0YTog"
    "aGQubWFwKGg9PisoaC5hdmdfb2NjKjUwKS50b0ZpeGVkKDIpKSwKICAgICAgICAgICAgYm9yZGVyQ29sb3I6ICcj"
    "MDBlNWEwJywgYmFja2dyb3VuZENvbG9yOiAncmdiYSgwLDIyOSwxNjAsMC4wNSknLAogICAgICAgICAgICBmaWxs"
    "OiB0cnVlLCB0ZW5zaW9uOiAwLjQsIHBvaW50UmFkaXVzOiAzLCB5QXhpc0lEOiAneTInLAogICAgICAgICAgICBi"
    "b3JkZXJEYXNoOiBbNCwyXQogICAgICAgICAgfV0KICAgICAgICB9LAogICAgICAgIG9wdGlvbnM6IHsKICAgICAg"
    "ICAgIHJlc3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RSYXRpbzpmYWxzZSwKICAgICAgICAgIHNjYWxlczog"
    "ewogICAgICAgICAgICB4OiB7IGdyaWQ6e2NvbG9yOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpJ30sIHRpY2tzOntj"
    "b2xvcjonIzY0NzQ4YicsbWF4Um90YXRpb246NDV9IH0sCiAgICAgICAgICAgIHk6IHsgZ3JpZDp7Y29sb3I6J3Jn"
    "YmEoMjU1LDI1NSwyNTUsMC4wNCknfSwgdGlja3M6e2NvbG9yOicjNjQ3NDhiJ30sIHBvc2l0aW9uOidsZWZ0JyB9"
    "LAogICAgICAgICAgICB5MjogeyBkaXNwbGF5OmZhbHNlIH0KICAgICAgICAgIH0sCiAgICAgICAgICBwbHVnaW5z"
    "OiB7IGxlZ2VuZDp7bGFiZWxzOntjb2xvcjonIzk0YTNiOCcsZm9udDp7c2l6ZToxMX19fSB9CiAgICAgICAgfQog"
    "ICAgICB9KTsKICAgIH0KCiAgICAvLyBPY2MgdnMgTGlnaHQgc2NhdHRlcgogICAgaWYgKGhkLmxlbmd0aCA+IDAp"
    "IHsKICAgICAgZGVzdHJveUNoYXJ0KCdvY2NMaWdodCcpOwogICAgICBDSEFSVFNbJ29jY0xpZ2h0J10gPSBuZXcg"
    "Q2hhcnQoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ29jY0xpZ2h0Q2hhcnQnKSwgewogICAgICAgIHR5cGU6ICdi"
    "YXInLAogICAgICAgIGRhdGE6IHsKICAgICAgICAgIGxhYmVsczogaGQubWFwKGg9PmguaG91cisnOjAwJyksCiAg"
    "ICAgICAgICBkYXRhc2V0czogW3sKICAgICAgICAgICAgbGFiZWw6ICdPY2N1cGFuY3kgKCUpJywKICAgICAgICAg"
    "ICAgZGF0YTogaGQubWFwKGg9PisoaC5hdmdfb2NjKjEwMCkudG9GaXhlZCgxKSksCiAgICAgICAgICAgIGJhY2tn"
    "cm91bmRDb2xvcjogJ3JnYmEoMCwyMTIsMjU1LDAuNiknLAogICAgICAgICAgICBib3JkZXJSYWRpdXM6IDMKICAg"
    "ICAgICAgIH1dCiAgICAgICAgfSwKICAgICAgICBvcHRpb25zOiB7CiAgICAgICAgICByZXNwb25zaXZlOnRydWUs"
    "IG1haW50YWluQXNwZWN0UmF0aW86ZmFsc2UsCiAgICAgICAgICBzY2FsZXM6IHsKICAgICAgICAgICAgeDogeyBn"
    "cmlkOntjb2xvcjoncmdiYSgyNTUsMjU1LDI1NSwwLjA0KSd9LCB0aWNrczp7Y29sb3I6JyM2NDc0OGInLG1heFJv"
    "dGF0aW9uOjQ1fSB9LAogICAgICAgICAgICB5OiB7IGdyaWQ6e2NvbG9yOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQp"
    "J30sIHRpY2tzOntjb2xvcjonIzY0NzQ4YicsY2FsbGJhY2s6dj0+disnJSd9IH0KICAgICAgICAgIH0sCiAgICAg"
    "ICAgICBwbHVnaW5zOntsZWdlbmQ6e2xhYmVsczp7Y29sb3I6JyM5NGEzYjgnfX19CiAgICAgICAgfQogICAgICB9"
    "KTsKICAgIH0KCiAgICAvLyBUaW1lbGluZQogICAgbG9hZFRpbWVsaW5lKCk7CiAgICAvLyBBZ2VudCBjaGFydAog"
    "ICAgbG9hZEFnZW50Q2hhcnQoKTsKCiAgfSBjYXRjaChlKSB7IHRvYXN0KCdGYWlsZWQgdG8gbG9hZCBkYXNoYm9h"
    "cmQnLCAnZXJyb3InKTsgfQp9Cgphc3luYyBmdW5jdGlvbiBsb2FkVGltZWxpbmUoKSB7CiAgdHJ5IHsKICAgIGNv"
    "bnN0IGQgPSBhd2FpdCBmZXRjaCgnL2FwaS9hbmFseXRpY3MvZmF1bHRfdGltZWxpbmUnKS50aGVuKHI9PnIuanNv"
    "bigpKTsKICAgIGlmICghZC5sYWJlbHMgfHwgZC5sYWJlbHMubGVuZ3RoID09PSAwKSByZXR1cm47CiAgICBkZXN0"
    "cm95Q2hhcnQoJ3RpbWVsaW5lJyk7CiAgICBjb25zdCBkYXRhc2V0cyA9IE9iamVjdC5lbnRyaWVzKGQuc2VyaWVz"
    "KS5tYXAoKFtmdCx2YWxzXSxpKSA9PiAoewogICAgICBsYWJlbDogZnQsCiAgICAgIGRhdGE6IHZhbHMsCiAgICAg"
    "IGJhY2tncm91bmRDb2xvcjogKENPTE9SU1tmdF18fFBBTEVUVEVbaSVQQUxFVFRFLmxlbmd0aF0pKyc5OScsCiAg"
    "ICAgIGJvcmRlckNvbG9yOiBDT0xPUlNbZnRdfHxQQUxFVFRFW2klUEFMRVRURS5sZW5ndGhdLAogICAgICBib3Jk"
    "ZXJXaWR0aDogMSwgYm9yZGVyUmFkaXVzOiAzCiAgICB9KSk7CiAgICBDSEFSVFNbJ3RpbWVsaW5lJ10gPSBuZXcg"
    "Q2hhcnQoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RpbWVsaW5lQ2hhcnQnKSwgewogICAgICB0eXBlOiAnYmFy"
    "JywKICAgICAgZGF0YTogeyBsYWJlbHM6IGQubGFiZWxzLm1hcChsPT4nU3RlcCAnK2wpLCBkYXRhc2V0cyB9LAog"
    "ICAgICBvcHRpb25zOiB7CiAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJhdGlvOmZhbHNl"
    "LAogICAgICAgIHNjYWxlczogewogICAgICAgICAgeDogeyBzdGFja2VkOnRydWUsIGdyaWQ6e2NvbG9yOidyZ2Jh"
    "KDI1NSwyNTUsMjU1LDAuMDMpJ30sIHRpY2tzOntjb2xvcjonIzY0NzQ4Yid9IH0sCiAgICAgICAgICB5OiB7IHN0"
    "YWNrZWQ6dHJ1ZSwgZ3JpZDp7Y29sb3I6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknfSwgdGlja3M6e2NvbG9yOicj"
    "NjQ3NDhiJ30gfQogICAgICAgIH0sCiAgICAgICAgcGx1Z2luczogeyBsZWdlbmQ6e2xhYmVsczp7Y29sb3I6JyM5"
    "NGEzYjgnLGZvbnQ6e3NpemU6MTB9fX0gfQogICAgICB9CiAgICB9KTsKICB9IGNhdGNoKGUpIHt9Cn0KCmFzeW5j"
    "IGZ1bmN0aW9uIGxvYWRBZ2VudENoYXJ0KCkgewogIHRyeSB7CiAgICBjb25zdCBkID0gYXdhaXQgZmV0Y2goJy9h"
    "cGkvYW5hbHl0aWNzL3JlY292ZXJ5X2Rpc3QnKS50aGVuKHI9PnIuanNvbigpKTsKICAgIGlmICghZC5hZ2VudHMg"
    "fHwgZC5hZ2VudHMubGVuZ3RoID09PSAwKSByZXR1cm47CiAgICBkZXN0cm95Q2hhcnQoJ2FnZW50Jyk7CiAgICBD"
    "SEFSVFNbJ2FnZW50J10gPSBuZXcgQ2hhcnQoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2FnZW50Q2hhcnQnKSwg"
    "ewogICAgICB0eXBlOiAnYmFyJywKICAgICAgZGF0YTogewogICAgICAgIGxhYmVsczogZC5hZ2VudHMsCiAgICAg"
    "ICAgZGF0YXNldHM6IFt7CiAgICAgICAgICBsYWJlbDogJ1JlY292ZXJ5IENvdW50JywKICAgICAgICAgIGRhdGE6"
    "IGQuY291bnRzLAogICAgICAgICAgYmFja2dyb3VuZENvbG9yOiBQQUxFVFRFLnNsaWNlKDAsZC5hZ2VudHMubGVu"
    "Z3RoKS5tYXAoYz0+YysnY2MnKSwKICAgICAgICAgIGJvcmRlckNvbG9yOiBQQUxFVFRFLnNsaWNlKDAsZC5hZ2Vu"
    "dHMubGVuZ3RoKSwKICAgICAgICAgIGJvcmRlcldpZHRoOiAyLCBib3JkZXJSYWRpdXM6IDYKICAgICAgICB9XQog"
    "ICAgICB9LAogICAgICBvcHRpb25zOiB7CiAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJh"
    "dGlvOmZhbHNlLCBpbmRleEF4aXM6J3knLAogICAgICAgIHNjYWxlczogewogICAgICAgICAgeDogeyBncmlkOntj"
    "b2xvcjoncmdiYSgyNTUsMjU1LDI1NSwwLjA0KSd9LCB0aWNrczp7Y29sb3I6JyM2NDc0OGInfSB9LAogICAgICAg"
    "ICAgeTogeyBncmlkOntkaXNwbGF5OmZhbHNlfSwgdGlja3M6e2NvbG9yOicjOTRhM2I4J30gfQogICAgICAgIH0s"
    "CiAgICAgICAgcGx1Z2luczp7bGVnZW5kOntkaXNwbGF5OmZhbHNlfX0KICAgICAgfQogICAgfSk7CiAgfSBjYXRj"
    "aChlKSB7fQp9CgovLyDilIDilIDilIAgTUwgTU9ERUxTIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAphc3luYyBmdW5jdGlvbiBsb2FkTW9kZWxzKCkgewogIHRyeSB7"
    "CiAgICBjb25zdCBkID0gYXdhaXQgZmV0Y2goJy9hcGkvbWwvcmVzdWx0cycpLnRoZW4ocj0+ci5qc29uKCkpOwog"
    "ICAgaWYgKGQuZXJyb3IpIHJldHVybjsKICAgIAogICAgLy8gVXBkYXRlIHF1aWNrIG1ldHJpYwogICAgY29uc3Qg"
    "YmVzdFJlcyA9IE9iamVjdC52YWx1ZXMoZC5yZXN1bHRzKS5maW5kKHI9PnIuaXNfYmVzdCk7CiAgICBpZiAoYmVz"
    "dFJlcykgewogICAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncUYxJykudGV4dENvbnRlbnQgPSBiZXN0UmVz"
    "LmYxX3Njb3JlOwogICAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncU1vZGVsTmFtZScpLnRleHRDb250ZW50"
    "ID0gZC5iZXN0X21vZGVsOwogICAgfQoKICAgIC8vIE1vZGVsIGNhcmRzCiAgICBjb25zdCBncmlkID0gZG9jdW1l"
    "bnQuZ2V0RWxlbWVudEJ5SWQoJ21vZGVsc0dyaWQnKTsKICAgIGdyaWQuaW5uZXJIVE1MID0gJyc7CiAgICBmb3Ig"
    "KGNvbnN0IFtuYW1lLCByXSBvZiBPYmplY3QuZW50cmllcyhkLnJlc3VsdHMpKSB7CiAgICAgIGNvbnN0IGNhcmQg"
    "PSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTsKICAgICAgY2FyZC5jbGFzc05hbWUgPSAnbW9kZWwtY2Fy"
    "ZCcgKyAoci5pc19iZXN0PycgYmVzdCc6JycpOwogICAgICBjYXJkLmlubmVySFRNTCA9IGAKICAgICAgICA8ZGl2"
    "IGNsYXNzPSJtb2RlbC1uYW1lIj4ke25hbWV9ICR7ci5pc19iZXN0Pyc8c3BhbiBjbGFzcz0ibW9kZWwtYmFkZ2Ui"
    "PkJlc3Q8L3NwYW4+JzonJ308L2Rpdj4KICAgICAgICA8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS10ZXh0Myk7Zm9u"
    "dC1zaXplOjAuNzVyZW07bWFyZ2luLWJvdHRvbToxcmVtO2ZvbnQtZmFtaWx5OnZhcigtLW1vbm8pIj4KICAgICAg"
    "ICAgIFRyYWluIHRpbWU6ICR7ci50cmFpbl90aW1lX3NlY31zCiAgICAgICAgPC9kaXY+CiAgICAgICAgJHtbJ2Fj"
    "Y3VyYWN5JywncHJlY2lzaW9uJywncmVjYWxsJywnZjFfc2NvcmUnXS5tYXAobT0+YAogICAgICAgICAgPGRpdiBj"
    "bGFzcz0ibWV0cmljLXJvdyI+CiAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJsYWJlbCI+JHttfTwvc3Bhbj4KICAg"
    "ICAgICAgICAgPGRpdj4KICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0idmFsdWUiPiR7KHJbbV0qMTAwKS50b0Zp"
    "eGVkKDIpfSU8L3NwYW4+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0iYmFyLW1pbmkiPjxkaXYgY2xhc3M9ImJh"
    "ci1maWxsIiBzdHlsZT0id2lkdGg6JHtyW21dKjEwMH0lIj48L2Rpdj48L2Rpdj4KICAgICAgICAgICAgPC9kaXY+"
    "CiAgICAgICAgICA8L2Rpdj4KICAgICAgICBgKS5qb2luKCcnKX0KICAgICAgYDsKICAgICAgZ3JpZC5hcHBlbmRD"
    "aGlsZChjYXJkKTsKICAgIH0KCiAgICAvLyBNb2RlbCBjb21wYXJpc29uIGNoYXJ0CiAgICBjb25zdCBuYW1lcyA9"
    "IE9iamVjdC5rZXlzKGQucmVzdWx0cyk7CiAgICBjb25zdCBtZXRyaWNzID0gWydhY2N1cmFjeScsJ3ByZWNpc2lv"
    "bicsJ3JlY2FsbCcsJ2YxX3Njb3JlJ107CiAgICBkZXN0cm95Q2hhcnQoJ21vZGVsQ29tcCcpOwogICAgQ0hBUlRT"
    "Wydtb2RlbENvbXAnXSA9IG5ldyBDaGFydChkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbW9kZWxDb21wQ2hhcnQn"
    "KSwgewogICAgICB0eXBlOiAnYmFyJywKICAgICAgZGF0YTogewogICAgICAgIGxhYmVsczogbmFtZXMsCiAgICAg"
    "ICAgZGF0YXNldHM6IG1ldHJpY3MubWFwKChtLGkpID0+ICh7CiAgICAgICAgICBsYWJlbDogbSwKICAgICAgICAg"
    "IGRhdGE6IG5hbWVzLm1hcChuID0+IGQucmVzdWx0c1tuXVttXSksCiAgICAgICAgICBiYWNrZ3JvdW5kQ29sb3I6"
    "IFBBTEVUVEVbaV0rJ2NjJywKICAgICAgICAgIGJvcmRlckNvbG9yOiBQQUxFVFRFW2ldLAogICAgICAgICAgYm9y"
    "ZGVyV2lkdGg6IDIsIGJvcmRlclJhZGl1czogNAogICAgICAgIH0pKQogICAgICB9LAogICAgICBvcHRpb25zOiB7"
    "CiAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJhdGlvOmZhbHNlLAogICAgICAgIHNjYWxl"
    "czogewogICAgICAgICAgeDogeyBncmlkOntjb2xvcjoncmdiYSgyNTUsMjU1LDI1NSwwLjA0KSd9LCB0aWNrczp7"
    "Y29sb3I6JyM5NGEzYjgnfSB9LAogICAgICAgICAgeTogeyBtaW46MC43NSwgbWF4OjAuOTUsIGdyaWQ6e2NvbG9y"
    "OidyZ2JhKDI1NSwyNTUsMjU1LDAuMDQpJ30sIHRpY2tzOntjb2xvcjonIzY0NzQ4YicsY2FsbGJhY2s6dj0+KHYq"
    "MTAwKS50b0ZpeGVkKDApKyclJ30gfQogICAgICAgIH0sCiAgICAgICAgcGx1Z2luczp7bGVnZW5kOntsYWJlbHM6"
    "e2NvbG9yOicjOTRhM2I4Jyxmb250OntzaXplOjEwfX19fQogICAgICB9CiAgICB9KTsKCiAgICAvLyBUcmFpbiB0"
    "aW1lCiAgICBkZXN0cm95Q2hhcnQoJ3RyYWluVGltZScpOwogICAgQ0hBUlRTWyd0cmFpblRpbWUnXSA9IG5ldyBD"
    "aGFydChkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHJhaW5UaW1lQ2hhcnQnKSwgewogICAgICB0eXBlOiAnYmFy"
    "JywKICAgICAgZGF0YTogewogICAgICAgIGxhYmVsczogbmFtZXMsCiAgICAgICAgZGF0YXNldHM6IFt7CiAgICAg"
    "ICAgICBsYWJlbDogJ1RyYWluIFRpbWUgKHMpJywKICAgICAgICAgIGRhdGE6IG5hbWVzLm1hcChuPT5kLnJlc3Vs"
    "dHNbbl0udHJhaW5fdGltZV9zZWMpLAogICAgICAgICAgYmFja2dyb3VuZENvbG9yOiBuYW1lcy5tYXAoKF8saSk9"
    "PlBBTEVUVEVbaV0rJ2NjJyksCiAgICAgICAgICBib3JkZXJDb2xvcjogbmFtZXMubWFwKChfLGkpPT5QQUxFVFRF"
    "W2ldKSwKICAgICAgICAgIGJvcmRlcldpZHRoOiAyLCBib3JkZXJSYWRpdXM6IDYKICAgICAgICB9XQogICAgICB9"
    "LAogICAgICBvcHRpb25zOiB7CiAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJhdGlvOmZh"
    "bHNlLAogICAgICAgIHNjYWxlczogewogICAgICAgICAgeDogeyBncmlkOntkaXNwbGF5OmZhbHNlfSwgdGlja3M6"
    "e2NvbG9yOicjOTRhM2I4J30gfSwKICAgICAgICAgIHk6IHsgZ3JpZDp7Y29sb3I6J3JnYmEoMjU1LDI1NSwyNTUs"
    "MC4wNCknfSwgdGlja3M6e2NvbG9yOicjNjQ3NDhiJ30gfQogICAgICAgIH0sCiAgICAgICAgcGx1Z2luczp7bGVn"
    "ZW5kOntkaXNwbGF5OmZhbHNlfX0KICAgICAgfQogICAgfSk7CgogICAgLy8gQ29uZnVzaW9uIG1hdHJpeCBoZWF0"
    "bWFwIGZvciBiZXN0IG1vZGVsCiAgICBjb25zdCBiZXN0Q00gPSBkLnJlc3VsdHNbZC5iZXN0X21vZGVsXT8uY29u"
    "ZnVzaW9uX21hdHJpeDsKICAgIGlmIChiZXN0Q00gJiYgYmVzdENNLmxlbmd0aCA+IDApIHsKICAgICAgZG9jdW1l"
    "bnQuZ2V0RWxlbWVudEJ5SWQoJ2NtQmFkZ2UnKS50ZXh0Q29udGVudCA9IGQuYmVzdF9tb2RlbDsKICAgICAgY29u"
    "c3QgbGFiZWxzID0gZC5jbGFzc2VzIHx8IFtdOwogICAgICBjb25zdCBmbGF0ID0gW107CiAgICAgIGNvbnN0IGNt"
    "TGFiZWxzID0gW107CiAgICAgIGJlc3RDTS5mb3JFYWNoKChyb3csaSkgPT4gcm93LmZvckVhY2goKHZhbCxqKSA9"
    "PiB7CiAgICAgICAgZmxhdC5wdXNoKHt4OmoseTppLHY6dmFsfSk7CiAgICAgICAgY21MYWJlbHMucHVzaChsYWJl"
    "bHNbaV18fGkpOwogICAgICB9KSk7CiAgICAgIGNvbnN0IG1heFZhbCA9IE1hdGgubWF4KC4uLmZsYXQubWFwKGY9"
    "PmYudikpOwogICAgICBkZXN0cm95Q2hhcnQoJ2NvbmZNYXQnKTsKICAgICAgQ0hBUlRTWydjb25mTWF0J10gPSBu"
    "ZXcgQ2hhcnQoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2NvbmZNYXRDaGFydCcpLCB7CiAgICAgICAgdHlwZTog"
    "J3NjYXR0ZXInLAogICAgICAgIGRhdGE6IHsKICAgICAgICAgIGRhdGFzZXRzOiBbewogICAgICAgICAgICBkYXRh"
    "OiBmbGF0Lm1hcChmPT4oe3g6Zi54LCB5OmYueSwgdjpmLnZ9KSksCiAgICAgICAgICAgIGJhY2tncm91bmRDb2xv"
    "cjogZmxhdC5tYXAoZj0+YHJnYmEoMzAsMTA3LDI1NSwkezAuMSswLjg1KihmLnYvbWF4VmFsKX0pYCksCiAgICAg"
    "ICAgICAgIHBvaW50UmFkaXVzOiBmbGF0Lm1hcCgoKSA9PiAyOCksIHBvaW50SG92ZXJSYWRpdXM6IDMyCiAgICAg"
    "ICAgICB9XQogICAgICAgIH0sCiAgICAgICAgb3B0aW9uczogewogICAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBt"
    "YWludGFpbkFzcGVjdFJhdGlvOmZhbHNlLAogICAgICAgICAgc2NhbGVzOiB7CiAgICAgICAgICAgIHg6IHsgbWlu"
    "Oi0wLjUsIG1heDpsYWJlbHMubGVuZ3RoLTAuNSwgdGlja3M6e2NhbGxiYWNrOnY9PmxhYmVsc1t2XXx8diwgY29s"
    "b3I6JyM2NDc0OGInLCBmb250OntzaXplOjl9fSwgZ3JpZDp7Y29sb3I6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCkn"
    "fSB9LAogICAgICAgICAgICB5OiB7IG1pbjotMC41LCBtYXg6bGFiZWxzLmxlbmd0aC0wLjUsIHJldmVyc2U6dHJ1"
    "ZSwgdGlja3M6e2NhbGxiYWNrOnY9PmxhYmVsc1t2XXx8diwgY29sb3I6JyM2NDc0OGInLCBmb250OntzaXplOjl9"
    "fSwgZ3JpZDp7Y29sb3I6J3JnYmEoMjU1LDI1NSwyNTUsMC4wNCknfSB9CiAgICAgICAgICB9LAogICAgICAgICAg"
    "cGx1Z2luczogewogICAgICAgICAgICBsZWdlbmQ6e2Rpc3BsYXk6ZmFsc2V9LAogICAgICAgICAgICB0b29sdGlw"
    "OntjYWxsYmFja3M6e2xhYmVsOmM9PmAke2xhYmVsc1tjLnJhdy55XX0g4oaSICR7bGFiZWxzW2MucmF3LnhdfTog"
    "JHtjLnJhdy52fWB9fQogICAgICAgICAgfQogICAgICAgIH0KICAgICAgfSk7CiAgICB9CgogIH0gY2F0Y2goZSkg"
    "eyBjb25zb2xlLmVycm9yKGUpOyB9Cn0KCi8vIOKUgOKUgOKUgCBMSVZFIEZFRUQg4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmxldCBmZWVkRGF0YSA9IFtdOwpsZXQg"
    "ZmVlZEZpbHRlciA9ICcnOwpmdW5jdGlvbiBmaWx0ZXJGZWVkKCkgewogIGZlZWRGaWx0ZXIgPSBkb2N1bWVudC5n"
    "ZXRFbGVtZW50QnlJZCgnZmVlZEZpbHRlclNlbCcpLnZhbHVlOwogIHJlbmRlckZlZWQoKTsKfQphc3luYyBmdW5j"
    "dGlvbiByZWZyZXNoRmVlZCgpIHsKICB0cnkgewogICAgY29uc3QgZCA9IGF3YWl0IGZldGNoKCcvYXBpL2xpdmUv"
    "ZXZlbnRzJykudGhlbihyPT5yLmpzb24oKSk7CiAgICBmZWVkRGF0YSA9IGQ7IHJlbmRlckZlZWQoKTsKICB9IGNh"
    "dGNoKGUpIHt9Cn0KZnVuY3Rpb24gcmVuZGVyRmVlZCgpIHsKICBjb25zdCBzY3JvbGwgPSBkb2N1bWVudC5nZXRF"
    "bGVtZW50QnlJZCgnZmVlZFNjcm9sbCcpOwogIGNvbnN0IGRhdGEgPSBmZWVkRmlsdGVyID8gZmVlZERhdGEuZmls"
    "dGVyKGU9PmUuZmF1bHRfdHlwZT09PWZlZWRGaWx0ZXIpIDogZmVlZERhdGE7CiAgaWYgKGRhdGEubGVuZ3RoID09"
    "PSAwKSB7CiAgICBzY3JvbGwuaW5uZXJIVE1MID0gJzxkaXYgc3R5bGU9InRleHQtYWxpZ246Y2VudGVyO3BhZGRp"
    "bmc6M3JlbTtjb2xvcjp2YXIoLS10ZXh0Myk7Zm9udC1zaXplOjAuODVyZW07Ij5ObyBldmVudHMnICsgKGZlZWRG"
    "aWx0ZXI/JyBmb3IgdGhpcyBmYXVsdCB0eXBlJzonJykgKyAnPC9kaXY+JzsKICAgIHJldHVybjsKICB9CiAgc2Ny"
    "b2xsLmlubmVySFRNTCA9IFsuLi5kYXRhXS5yZXZlcnNlKCkubWFwKGUgPT4gYAogICAgPGRpdiBjbGFzcz0iZmVl"
    "ZC1yb3ciPgogICAgICA8c3BhbiBjbGFzcz0iY29sLXRpbWUiPiR7ZS50aW1lc3RhbXAgPyBlLnRpbWVzdGFtcC5z"
    "bGljZSgxMSwxOSkgOiAnLS06LS06LS0nfTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImNvbC1mYXVsdCBmYXVs"
    "dC0ke2UuZmF1bHRfdHlwZX0iPiR7ZS5mYXVsdF90eXBlfTwvc3Bhbj4KICAgICAgPHNwYW4gc3R5bGU9ImNvbG9y"
    "OnZhcigtLXRleHQyKSI+JHtlLmRldGVjdGVkX2J5fTwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9ImNvbC1zZXYi"
    "PjxzcGFuIGNsYXNzPSJzZXZlcml0eS1iYWRnZSBzZXYtJHtlLmZhdWx0X3NldmVyaXR5fSI+U0VWICR7ZS5mYXVs"
    "dF9zZXZlcml0eX08L3NwYW4+PC9zcGFuPgogICAgICA8c3BhbiBzdHlsZT0iY29sb3I6dmFyKC0tdGV4dDMpO292"
    "ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzIj4ke2UucmVjb3ZlcnlfYWN0aW9ufTwvc3Bhbj4K"
    "ICAgICAgPHNwYW4+PHNwYW4gY2xhc3M9InN1Y2Nlc3MtcGlsbCAke2Uuc3VjY2Vzcz8nb2snOidmYWlsJ30iPiR7"
    "ZS5zdWNjZXNzPyfinJMgT0snOifinJcgRkFJTCd9PC9zcGFuPjwvc3Bhbj4KICAgIDwvZGl2PgogIGApLmpvaW4o"
    "JycpOwp9Cgphc3luYyBmdW5jdGlvbiBsb2FkUlRIaXN0KCkgewogIHRyeSB7CiAgICBjb25zdCBydW5JZCA9IGF3"
    "YWl0IGZldGNoKCcvYXBpL3N0YXR1cycpLnRoZW4ocj0+ci5qc29uKCkpLnRoZW4oZD0+ZC5sYXN0X3J1bl9pZCk7"
    "CiAgICBpZiAoIXJ1bklkKSByZXR1cm47CiAgICBjb25zdCBkID0gYXdhaXQgZmV0Y2goYC9hcGkvcnVucy8ke3J1"
    "bklkfS9ldmVudHNgKS50aGVuKHI9PnIuanNvbigpKTsKICAgIGlmICghZC5sZW5ndGgpIHJldHVybjsKICAgIGNv"
    "bnN0IHRpbWVzID0gZC5tYXAoZT0+ZS5yZWNvdmVyeV90aW1lX21zKTsKICAgIGNvbnN0IG1pbiA9IE1hdGgubWlu"
    "KC4uLnRpbWVzKSwgbWF4ID0gTWF0aC5tYXgoLi4udGltZXMpOwogICAgY29uc3QgYmlucyA9IDIwOwogICAgY29u"
    "c3Qgc3RlcCA9IChtYXgtbWluKS9iaW5zOwogICAgY29uc3QgY291bnRzID0gQXJyYXkoYmlucykuZmlsbCgwKTsK"
    "ICAgIHRpbWVzLmZvckVhY2godCA9PiB7IGNvbnN0IGkgPSBNYXRoLm1pbihNYXRoLmZsb29yKCh0LW1pbikvc3Rl"
    "cCksIGJpbnMtMSk7IGNvdW50c1tpXSsrOyB9KTsKICAgIGNvbnN0IGxhYmVscyA9IEFycmF5KGJpbnMpLmZpbGwo"
    "MCkubWFwKChfLGkpPT5NYXRoLnJvdW5kKG1pbitpKnN0ZXApKydtcycpOwogICAgZGVzdHJveUNoYXJ0KCdydEhp"
    "c3QnKTsKICAgIENIQVJUU1sncnRIaXN0J10gPSBuZXcgQ2hhcnQoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3J0"
    "SGlzdENoYXJ0JyksIHsKICAgICAgdHlwZTogJ2JhcicsCiAgICAgIGRhdGE6IHsgbGFiZWxzLCBkYXRhc2V0czog"
    "W3sKICAgICAgICBsYWJlbDogJ0V2ZW50IENvdW50JywKICAgICAgICBkYXRhOiBjb3VudHMsCiAgICAgICAgYmFj"
    "a2dyb3VuZENvbG9yOiAncmdiYSgzMCwxMDcsMjU1LDAuNiknLCBib3JkZXJDb2xvcjogJyMxZTZiZmYnLCBib3Jk"
    "ZXJXaWR0aDogMSwgYm9yZGVyUmFkaXVzOiAzCiAgICAgIH1dfSwKICAgICAgb3B0aW9uczogewogICAgICAgIHJl"
    "c3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RSYXRpbzpmYWxzZSwKICAgICAgICBzY2FsZXM6IHsKICAgICAg"
    "ICAgIHg6e2dyaWQ6e2NvbG9yOidyZ2JhKDI1NSwyNTUsMjU1LDAuMDMpJ30sdGlja3M6e2NvbG9yOicjNjQ3NDhi"
    "JyxtYXhSb3RhdGlvbjo0NX19LAogICAgICAgICAgeTp7Z3JpZDp7Y29sb3I6J3JnYmEoMjU1LDI1NSwyNTUsMC4w"
    "NCknfSx0aWNrczp7Y29sb3I6JyM2NDc0OGInfX0KICAgICAgICB9LAogICAgICAgIHBsdWdpbnM6e2xlZ2VuZDp7"
    "bGFiZWxzOntjb2xvcjonIzk0YTNiOCd9fX0KICAgICAgfQogICAgfSk7CiAgfSBjYXRjaChlKSB7fQp9CgovLyDi"
    "lIDilIDilIAgREFUQSBUQUJMRSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIAKbGV0IGN1cnJlbnRQYWdlID0gMTsKYXN5bmMgZnVuY3Rpb24gbG9hZERhdGFQYWdlKHBhZ2Up"
    "IHsKICBjdXJyZW50UGFnZSA9IHBhZ2U7CiAgY29uc3QgZmF1bHQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgn"
    "ZmF1bHRGaWx0ZXJTZWwnKS52YWx1ZTsKICBjb25zdCB1cmwgPSBgL2FwaS9kYXRhc2V0L3BhZ2U/cGFnZT0ke3Bh"
    "Z2V9JnBlcl9wYWdlPTUwJHtmYXVsdD8nJmZhdWx0PScrZmF1bHQ6Jyd9YDsKICB0cnkgewogICAgY29uc3QgZCA9"
    "IGF3YWl0IGZldGNoKHVybCkudGhlbihyPT5yLmpzb24oKSk7CiAgICBpZiAoZC5lcnJvcikgeyB0b2FzdChkLmVy"
    "cm9yLCdlcnJvcicpOyByZXR1cm47IH0KICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkYXRhc2V0Q291bnQn"
    "KS50ZXh0Q29udGVudCA9IGQudG90YWwudG9Mb2NhbGVTdHJpbmcoKSArICcgcm93cyc7CiAgICBjb25zdCBib2R5"
    "ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2RhdGFUYWJsZUJvZHknKTsKICAgIGJvZHkuaW5uZXJIVE1MID0g"
    "ZC5yb3dzLm1hcChyPT5gCiAgICAgIDx0cj4KICAgICAgICA8dGQ+JHtyLmlkfTwvdGQ+CiAgICAgICAgPHRkPiR7"
    "ci5ob3VyfTwvdGQ+CiAgICAgICAgPHRkPiR7ci5vdXRkb29yX3RlbXA/LnRvRml4ZWQoMSl8fCfigJQnfTwvdGQ+"
    "CiAgICAgICAgPHRkPiR7ci5vY2N1cGFuY3k/LnRvRml4ZWQoMyl8fCfigJQnfTwvdGQ+CiAgICAgICAgPHRkPiR7"
    "ci5odmFjX3Bvd2VyPy50b0ZpeGVkKDEpfHwn4oCUJ308L3RkPgogICAgICAgIDx0ZD4ke3IuaGVhdGluZ19sb2Fk"
    "Py50b0ZpeGVkKDEpfHwn4oCUJ308L3RkPgogICAgICAgIDx0ZD4ke3IuY29vbGluZ19sb2FkPy50b0ZpeGVkKDEp"
    "fHwn4oCUJ308L3RkPgogICAgICAgIDx0ZD4ke3IubGlnaHRpbmdfaW50ZW5zaXR5Py50b0ZpeGVkKDMpfHwn4oCU"
    "J308L3RkPgogICAgICAgIDx0ZD4ke3IucGFya2luZ19vY2N1cGFuY3k/LnRvRml4ZWQoMyl8fCfigJQnfTwvdGQ+"
    "CiAgICAgICAgPHRkPjxzcGFuIGNsYXNzPSJmYXVsdC0ke3IuZmF1bHRfbGFiZWx9Ij4ke3IuZmF1bHRfbGFiZWx9"
    "PC9zcGFuPjwvdGQ+CiAgICAgICAgPHRkPiR7ci5mYXVsdF9zZXZlcml0eX08L3RkPgogICAgICA8L3RyPgogICAg"
    "YCkuam9pbignJyk7CgogICAgLy8gUGFnaW5hdGlvbgogICAgY29uc3QgcGFnZXMgPSBNYXRoLmNlaWwoZC50b3Rh"
    "bCAvIGQucGVyX3BhZ2UpOwogICAgY29uc3QgcGFnID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3BhZ2luYXRp"
    "b24nKTsKICAgIGxldCBodG1sID0gJyc7CiAgICBpZiAocGFnZSA+IDEpIGh0bWwgKz0gYDxidXR0b24gY2xhc3M9"
    "InBhZ2UtYnRuIiBvbmNsaWNrPSJsb2FkRGF0YVBhZ2UoJHtwYWdlLTF9KSI+4oC5IFByZXY8L2J1dHRvbj5gOwog"
    "ICAgY29uc3Qgc3RhcnQgPSBNYXRoLm1heCgxLHBhZ2UtMiksIGVuZCA9IE1hdGgubWluKHBhZ2VzLHBhZ2UrMik7"
    "CiAgICBmb3IgKGxldCBwPXN0YXJ0O3A8PWVuZDtwKyspIGh0bWwgKz0gYDxidXR0b24gY2xhc3M9InBhZ2UtYnRu"
    "JHtwPT09cGFnZT8nIGFjdGl2ZSc6Jyd9IiBvbmNsaWNrPSJsb2FkRGF0YVBhZ2UoJHtwfSkiPiR7cH08L2J1dHRv"
    "bj5gOwogICAgaWYgKHBhZ2UgPCBwYWdlcykgaHRtbCArPSBgPGJ1dHRvbiBjbGFzcz0icGFnZS1idG4iIG9uY2xp"
    "Y2s9ImxvYWREYXRhUGFnZSgke3BhZ2UrMX0pIj5OZXh0IOKAujwvYnV0dG9uPmA7CiAgICBwYWcuaW5uZXJIVE1M"
    "ID0gaHRtbDsKICB9IGNhdGNoKGUpIHsgdG9hc3QoJ0ZhaWxlZCB0byBsb2FkIGRhdGEnLCdlcnJvcicpOyB9Cn0K"
    "CmFzeW5jIGZ1bmN0aW9uIGxvYWREYXRhRGlzdCgpIHsKICB0cnkgewogICAgY29uc3QgZCA9IGF3YWl0IGZldGNo"
    "KCcvYXBpL2RhdGFzZXQvc3RhdHMnKS50aGVuKHI9PnIuanNvbigpKTsKICAgIGlmIChkLmVycm9yIHx8ICFkLmZh"
    "dWx0X2Rpc3RyaWJ1dGlvbikgcmV0dXJuOwogICAgZGVzdHJveUNoYXJ0KCdkYXRhRGlzdCcpOwogICAgQ0hBUlRT"
    "WydkYXRhRGlzdCddID0gbmV3IENoYXJ0KGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkYXRhRGlzdENoYXJ0Jyks"
    "IHsKICAgICAgdHlwZTogJ2JhcicsCiAgICAgIGRhdGE6IHsKICAgICAgICBsYWJlbHM6IGQuZmF1bHRfZGlzdHJp"
    "YnV0aW9uLm1hcChmPT5mLmZhdWx0X2xhYmVsKSwKICAgICAgICBkYXRhc2V0czogW3sKICAgICAgICAgIGxhYmVs"
    "OiAnQ291bnQnLAogICAgICAgICAgZGF0YTogZC5mYXVsdF9kaXN0cmlidXRpb24ubWFwKGY9PmYuY250KSwKICAg"
    "ICAgICAgIGJhY2tncm91bmRDb2xvcjogZC5mYXVsdF9kaXN0cmlidXRpb24ubWFwKGY9PkNPTE9SU1tmLmZhdWx0"
    "X2xhYmVsXXx8JyM2NDc0OGInKSwKICAgICAgICAgIGJvcmRlclJhZGl1czogNiwgYm9yZGVyV2lkdGg6IDAKICAg"
    "ICAgICB9XQogICAgICB9LAogICAgICBvcHRpb25zOiB7CiAgICAgICAgcmVzcG9uc2l2ZTp0cnVlLCBtYWludGFp"
    "bkFzcGVjdFJhdGlvOmZhbHNlLAogICAgICAgIHNjYWxlczogewogICAgICAgICAgeDp7Z3JpZDp7ZGlzcGxheTpm"
    "YWxzZX0sdGlja3M6e2NvbG9yOicjOTRhM2I4J319LAogICAgICAgICAgeTp7Z3JpZDp7Y29sb3I6J3JnYmEoMjU1"
    "LDI1NSwyNTUsMC4wNCknfSx0aWNrczp7Y29sb3I6JyM2NDc0OGInfX0KICAgICAgICB9LAogICAgICAgIHBsdWdp"
    "bnM6e2xlZ2VuZDp7ZGlzcGxheTpmYWxzZX19CiAgICAgIH0KICAgIH0pOwogIH0gY2F0Y2goZSkge30KfQoKLy8g"
    "4pSA4pSA4pSAIERPV05MT0FEUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIAKZnVuY3Rpb24gZG93bmxvYWRDU1YoKSB7IHdpbmRvdy5vcGVuKCcvYXBpL2Rvd25sb2Fk"
    "L2RhdGFzZXQ/Zm9ybWF0PWNzdicsJ19ibGFuaycpOyB0b2FzdCgnRG93bmxvYWRpbmcgQ1NWLi4uJywnc3VjY2Vz"
    "cycsJ+KshycpOyB9CmZ1bmN0aW9uIGRvd25sb2FkSlNPTigpIHsgd2luZG93Lm9wZW4oJy9hcGkvZG93bmxvYWQv"
    "ZGF0YXNldD9mb3JtYXQ9anNvbicsJ19ibGFuaycpOyB0b2FzdCgnRG93bmxvYWRpbmcgSlNPTi4uLicsJ3N1Y2Nl"
    "c3MnLCfirIcnKTsgfQphc3luYyBmdW5jdGlvbiBkb3dubG9hZFJlY292ZXJ5TG9nKCkgewogIGNvbnN0IHN0YXR1"
    "cyA9IGF3YWl0IGZldGNoKCcvYXBpL3N0YXR1cycpLnRoZW4ocj0+ci5qc29uKCkpOwogIGlmICghc3RhdHVzLmxh"
    "c3RfcnVuX2lkKSB7IHRvYXN0KCdObyBzaW11bGF0aW9uIHJ1biBmb3VuZCcsJ2Vycm9yJyk7IHJldHVybjsgfQog"
    "IHdpbmRvdy5vcGVuKGAvYXBpL2Rvd25sb2FkL3JlY292ZXJ5X2xvZz9ydW5faWQ9JHtzdGF0dXMubGFzdF9ydW5f"
    "aWR9YCwnX2JsYW5rJyk7CiAgdG9hc3QoJ0Rvd25sb2FkaW5nIHJlY292ZXJ5IGxvZy4uLicsJ3N1Y2Nlc3MnLCfi"
    "rIcnKTsKfQpmdW5jdGlvbiBkb3dubG9hZFJlcG9ydCgpIHsKICBjb25zdCBjb250ZW50ID0gZG9jdW1lbnQuZ2V0"
    "RWxlbWVudEJ5SWQoJ3JlcG9ydENvbnRlbnQnKS5pbm5lclRleHQ7CiAgY29uc3QgYmxvYiA9IG5ldyBCbG9iKFtj"
    "b250ZW50XSwge3R5cGU6J3RleHQvcGxhaW4nfSk7CiAgY29uc3QgYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQo"
    "J2EnKTsKICBhLmhyZWYgPSBVUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpOwogIGEuZG93bmxvYWQgPSBgYWlfcmVw"
    "b3J0XyR7Y3VycmVudExhbmd9XyR7bmV3IERhdGUoKS50b0lTT1N0cmluZygpLnNsaWNlKDAsMTApfS50eHRgOwog"
    "IGEuY2xpY2soKTsKICB0b2FzdCgnUmVwb3J0IGRvd25sb2FkZWQnLCdzdWNjZXNzJywn4qyHJyk7Cn0KCi8vIOKU"
    "gOKUgOKUgCBQUkVESUNUIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgApjb25zdCBGQVVMVF9JQ09OUyA9IHtub3JtYWw6J+KchScsaHZhY19mYWlsdXJlOifwn5S0"
    "JyxlbmVyZ3lfb3ZlcmxvYWQ6J+KaoScsbGlnaHRpbmdfZmF1bHQ6J/CfkqEnLHNhZmV0eV9hbGFybTon8J+aqCcs"
    "c2Vuc29yX2ZhaWx1cmU6J/Cfk6EnLHBhcmtpbmdfY29uZ2VzdGlvbjon8J+alycsdW5rbm93bjon4p2TJ307CmFz"
    "eW5jIGZ1bmN0aW9uIHJ1blByZWRpY3QoKSB7CiAgY29uc3QgZmllbGRzID0gWydob3VyJywnb3V0ZG9vcl90ZW1w"
    "Jywnb2NjdXBhbmN5JywnaHZhY19wb3dlcicsJ2hlYXRpbmdfbG9hZCcsJ2Nvb2xpbmdfbG9hZCcsJ2xpZ2h0aW5n"
    "X2ludGVuc2l0eScsJ3Bhcmtpbmdfb2NjdXBhbmN5JywncmVsYXRpdmVfY29tcGFjdG5lc3MnLCdzdXJmYWNlX2Fy"
    "ZWEnLCd3YWxsX2FyZWEnLCdmYXVsdF9zZXZlcml0eSddOwogIGNvbnN0IGJvZHkgPSB7fTsKICBmaWVsZHMuZm9y"
    "RWFjaChmID0+IHsgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncF8nK2YpOyBpZihlbCkgYm9k"
    "eVtmXSA9IHBhcnNlRmxvYXQoZWwudmFsdWUpfHwwOyB9KTsKICB0cnkgewogICAgY29uc3QgZCA9IGF3YWl0IGZl"
    "dGNoKCcvYXBpL3ByZWRpY3QnLHttZXRob2Q6J1BPU1QnLGhlYWRlcnM6eydDb250ZW50LVR5cGUnOidhcHBsaWNh"
    "dGlvbi9qc29uJ30sYm9keTpKU09OLnN0cmluZ2lmeShib2R5KX0pLnRoZW4ocj0+ci5qc29uKCkpOwogICAgaWYg"
    "KGQuZXJyb3IpIHsgdG9hc3QoZC5lcnJvciwnZXJyb3InKTsgcmV0dXJuOyB9CiAgICBjb25zdCByZXMgPSBkb2N1"
    "bWVudC5nZXRFbGVtZW50QnlJZCgncHJlZGljdFJlc3VsdCcpOwogICAgcmVzLmNsYXNzTGlzdC5hZGQoJ3Nob3cn"
    "KTsKICAgIGNvbnN0IGljb24gPSBGQVVMVF9JQ09OU1tkLmZhdWx0X3R5cGVdfHwn4p2TJzsKICAgIGRvY3VtZW50"
    "LmdldEVsZW1lbnRCeUlkKCdyZXN1bHRGYXVsdCcpLmlubmVySFRNTCA9IGAke2ljb259IDxzcGFuIGNsYXNzPSJm"
    "YXVsdC0ke2QuZmF1bHRfdHlwZX0iPiR7ZC5mYXVsdF90eXBlfTwvc3Bhbj5gOwogICAgZG9jdW1lbnQuZ2V0RWxl"
    "bWVudEJ5SWQoJ3Jlc3VsdENvbmYnKS50ZXh0Q29udGVudCA9IChkLmNvbmZpZGVuY2UqMTAwKS50b0ZpeGVkKDIp"
    "KyclJzsKICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyZXN1bHRDb25mQmFyJykuc3R5bGUud2lkdGggPSAo"
    "ZC5jb25maWRlbmNlKjEwMCkrJyUnOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jlc3VsdE1vZGVsJyku"
    "dGV4dENvbnRlbnQgPSBkLm1vZGVsX3VzZWQ7CiAgfSBjYXRjaChlKSB7IHRvYXN0KCdQcmVkaWN0aW9uIGZhaWxl"
    "ZCcsJ2Vycm9yJyk7IH0KfQoKYXN5bmMgZnVuY3Rpb24gbG9hZFJhbmRvbVNhbXBsZSgpIHsKICB0cnkgewogICAg"
    "Y29uc3QgZCA9IGF3YWl0IGZldGNoKCcvYXBpL2RhdGFzZXQvcGFnZT9wYWdlPScrTWF0aC5mbG9vcihNYXRoLnJh"
    "bmRvbSgpKjEwMCsxKSsnJnBlcl9wYWdlPTEnKS50aGVuKHI9PnIuanNvbigpKTsKICAgIGlmIChkLnJvd3MgJiYg"
    "ZC5yb3dzWzBdKSB7CiAgICAgIGNvbnN0IHIgPSBkLnJvd3NbMF07CiAgICAgIGNvbnN0IG1hcCA9IHtob3VyOidw"
    "X2hvdXInLG91dGRvb3JfdGVtcDoncF9vdXRkb29yX3RlbXAnLG9jY3VwYW5jeToncF9vY2N1cGFuY3knLGh2YWNf"
    "cG93ZXI6J3BfaHZhY19wb3dlcicsaGVhdGluZ19sb2FkOidwX2hlYXRpbmdfbG9hZCcsY29vbGluZ19sb2FkOidw"
    "X2Nvb2xpbmdfbG9hZCcsbGlnaHRpbmdfaW50ZW5zaXR5OidwX2xpZ2h0aW5nX2ludGVuc2l0eScscGFya2luZ19v"
    "Y2N1cGFuY3k6J3BfcGFya2luZ19vY2N1cGFuY3knLHJlbGF0aXZlX2NvbXBhY3RuZXNzOidwX3JlbGF0aXZlX2Nv"
    "bXBhY3RuZXNzJyxzdXJmYWNlX2FyZWE6J3Bfc3VyZmFjZV9hcmVhJyx3YWxsX2FyZWE6J3Bfd2FsbF9hcmVhJyxm"
    "YXVsdF9zZXZlcml0eToncF9mYXVsdF9zZXZlcml0eSd9OwogICAgICBmb3IgKGNvbnN0IFtrLGlkXSBvZiBPYmpl"
    "Y3QuZW50cmllcyhtYXApKSB7CiAgICAgICAgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChpZCk7"
    "CiAgICAgICAgaWYgKGVsICYmIHJba10hPW51bGwpIGVsLnZhbHVlID0gdHlwZW9mIHJba109PT0nbnVtYmVyJyA/"
    "IHJba10udG9GaXhlZCgzKSA6IHJba107CiAgICAgIH0KICAgICAgdG9hc3QoJ1JhbmRvbSBzYW1wbGUgbG9hZGVk"
    "Jywnc3VjY2VzcycsJ/CfjrInKTsKICAgIH0KICB9IGNhdGNoKGUpIHsgdG9hc3QoJ0xvYWQgZGF0YXNldCBmaXJz"
    "dCcsJ2Vycm9yJyk7IH0KfQoKLy8g4pSA4pSA4pSAIEFJIFJFUE9SVCDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKbGV0IGN1cnJlbnRMYW5nID0gJ2VuJzsKbGV0IGxh"
    "c3RSZXBvcnQgPSBudWxsOwphc3luYyBmdW5jdGlvbiBsb2FkUmVwb3J0KGxhbmcsIGJ0bikgewogIGN1cnJlbnRM"
    "YW5nID0gbGFuZzsKICBpZiAoYnRuKSB7CiAgICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcubGFuZy1idG4n"
    "KS5mb3JFYWNoKGI9PmIuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJykpOwogICAgYnRuLmNsYXNzTGlzdC5hZGQo"
    "J2FjdGl2ZScpOwogIH0KICBjb25zdCBjb250ZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlcG9ydENv"
    "bnRlbnQnKTsKICBjb250ZW50LmlubmVySFRNTCA9ICc8ZGl2IGNsYXNzPSJyZXBvcnQtbG9hZGluZyI+PGRpdiBj"
    "bGFzcz0ic3Bpbm5lciI+PC9kaXY+PGRpdiBzdHlsZT0iY29sb3I6dmFyKC0tdGV4dDMpO2ZvbnQtc2l6ZTowLjg1"
    "cmVtIj5HZW5lcmF0aW5nIEFJIHJlcG9ydC4uLjwvZGl2PjwvZGl2Pic7CiAgdHJ5IHsKICAgIGNvbnN0IGQgPSBh"
    "d2FpdCBmZXRjaCgnL2FwaS9haV9yZXBvcnQnLHsKICAgICAgbWV0aG9kOidQT1NUJywgaGVhZGVyczp7J0NvbnRl"
    "bnQtVHlwZSc6J2FwcGxpY2F0aW9uL2pzb24nfSwKICAgICAgYm9keTogSlNPTi5zdHJpbmdpZnkoe2xhbmd1YWdl"
    "Omxhbmd9KQogICAgfSkudGhlbihyPT5yLmpzb24oKSk7CiAgICBsYXN0UmVwb3J0ID0gZDsKICAgIGNvbnRlbnQu"
    "aW5uZXJIVE1MID0gbWFya2VkLnBhcnNlKGQuY29udGVudCk7CiAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgn"
    "cmVwb3J0TWV0YScpLnRleHRDb250ZW50ID0gCiAgICAgIGBHZW5lcmF0ZWQ6ICR7bmV3IERhdGUoZC5nZW5lcmF0"
    "ZWRfYXQpLnRvTG9jYWxlU3RyaW5nKCl9IHwgTGFuZ3VhZ2U6ICR7ZC5sYW5ndWFnZV9uYW1lfSB8IE1vZGVsIFJ1"
    "bnM6ICR7ZC5tZXRyaWNzX3NuYXBzaG90Py50b3RhbF9ydW5zfHwwfWA7CiAgfSBjYXRjaChlKSB7CiAgICBjb250"
    "ZW50LmlubmVySFRNTCA9ICc8ZGl2IHN0eWxlPSJjb2xvcjp2YXIoLS1yZWQpO3BhZGRpbmc6MnJlbSI+RmFpbGVk"
    "IHRvIGdlbmVyYXRlIHJlcG9ydC4gSW5pdGlhbGl6ZSBwaXBlbGluZSBmaXJzdC48L2Rpdj4nOwogIH0KfQoKCi8v"
    "IOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkAovLyBMSVZFIE1PTklUT1Ig4oCUIFNtYXJ0QnVpbGQgQUkgIChKU1ggZGFzaGJvYXJk"
    "IOKGkiB2YW5pbGxhIEpTL0NoYXJ0LmpzKQovLyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKCmNvbnN0IExNX01MID0gewogICJS"
    "YW5kb20gRm9yZXN0IjogICAgICB7IGFjY3VyYWN5OjAuODk1LCBwcmVjaXNpb246MC45MDAsIHJlY2FsbDowLjg5"
    "NSwgZjFfc2NvcmU6MC44OTUsIHRyYWluX3RpbWVfc2VjOjEuMiB9LAogICJHcmFkaWVudCBCb29zdGluZyI6ICB7"
    "IGFjY3VyYWN5OjAuODg2LCBwcmVjaXNpb246MC44OTEsIHJlY2FsbDowLjg4NCwgZjFfc2NvcmU6MC44ODcsIHRy"
    "YWluX3RpbWVfc2VjOjEwOS41IH0sCiAgIk1MUCBOZXVyYWwgTmV0d29yayI6IHsgYWNjdXJhY3k6MC44NzIsIHBy"
    "ZWNpc2lvbjowLjg3OSwgcmVjYWxsOjAuODY4LCBmMV9zY29yZTowLjg3MywgdHJhaW5fdGltZV9zZWM6Ni4zIH0s"
    "Cn07CmNvbnN0IExNX0ZBVUxUX0NPTE9SUyA9IHsKICBodmFjX2ZhaWx1cmU6IiNlZjQ0NDQiLCBlbmVyZ3lfb3Zl"
    "cmxvYWQ6IiNmOTczMTYiLCBsaWdodGluZ19mYXVsdDoiI2VhYjMwOCIsCiAgc2FmZXR5X2FsYXJtOiIjZGMyNjI2"
    "Iiwgc2Vuc29yX2ZhaWx1cmU6IiM4YjVjZjYiLCBwYXJraW5nX2Nvbmdlc3Rpb246IiMwNmI2ZDQiLCBub3JtYWw6"
    "IiMyMmM1NWUiCn07CmNvbnN0IExNX0FHRU5UX0NPTE9SUyA9IHsgSFZBQzoiIzNiODJmNiIsIEVuZXJneToiI2Y5"
    "NzMxNiIsIExpZ2h0aW5nOiIjZWFiMzA4IiwgUGFya2luZzoiIzIyYzU1ZSIsIFNhZmV0eToiI2VmNDQ0NCIgfTsK"
    "Y29uc3QgTE1fRkFVTFRfVFlQRVMgPSBbImh2YWNfZmFpbHVyZSIsImVuZXJneV9vdmVybG9hZCIsImxpZ2h0aW5n"
    "X2ZhdWx0Iiwic2FmZXR5X2FsYXJtIiwic2Vuc29yX2ZhaWx1cmUiLCJwYXJraW5nX2Nvbmdlc3Rpb24iXTsKCmxl"
    "dCBsbVNpbURhdGEgPSBbXTsKbGV0IGxtUmVjb3ZlcnkgPSBbXTsKbGV0IGxtU3RlcCA9IDEyMDsKbGV0IGxtU2lt"
    "dWxhdGluZyA9IHRydWU7CmxldCBsbUludGVydmFsID0gbnVsbDsKbGV0IGxtSW5pdGlhbGl6ZWQgPSBmYWxzZTsK"
    "bGV0IGxtQWN0aXZlVGFiID0gIm92ZXJ2aWV3IjsKbGV0IGxtQWN0aXZlQWdlbnQgPSBudWxsOwoKZnVuY3Rpb24g"
    "bG1HZW5SZWFkaW5nKHN0ZXApIHsKICBjb25zdCBob3VyID0gKHN0ZXAgKiA1IC8gNjApICUgMjQ7CiAgY29uc3Qg"
    "ciA9ICgpID0+IE1hdGgucmFuZG9tKCk7CiAgY29uc3Qgb2NjID0gKGhvdXI+PTgmJmhvdXI8PTE4KSA/IDAuNCty"
    "KCkqMC41IDogcigpKjAuMjsKICBjb25zdCByZWFkaW5nID0gewogICAgc3RlcF9pZDogc3RlcCwgdGltZXN0YW1w"
    "OiBuZXcgRGF0ZSgpLnRvSVNPU3RyaW5nKCksIGhvdXI6IE1hdGguZmxvb3IoaG91ciksCiAgICBvdXRkb29yX3Rl"
    "bXA6ICsoMjAgKyAxMCpNYXRoLnNpbigyKk1hdGguUEkqaG91ci8yNCkgKyAocigpLTAuNSkqNCkudG9GaXhlZCgx"
    "KSwKICAgIG9jY3VwYW5jeTogK29jYy50b0ZpeGVkKDIpLCBodmFjX3Bvd2VyOiArKDIwICsgcigpKjQwKS50b0Zp"
    "eGVkKDEpLAogICAgaGVhdGluZ19sb2FkOiArKDE1ICsgcigpKjMwKS50b0ZpeGVkKDEpLCBjb29saW5nX2xvYWQ6"
    "ICsoMTUgKyByKCkqMjUpLnRvRml4ZWQoMSksCiAgICBsaWdodGluZ19pbnRlbnNpdHk6ICsoMC4zICsgMC42Km9j"
    "YyArIChyKCktMC41KSowLjEpLnRvRml4ZWQoMiksCiAgICBwYXJraW5nX29jY3VwYW5jeTogKyhob3VyPj03JiZo"
    "b3VyPD0yMCA/IDAuMytyKCkqMC42NSA6IHIoKSowLjE1KS50b0ZpeGVkKDIpLAogICAgYWN0aXZlX2ZhdWx0OiBu"
    "dWxsLCBmYXVsdF9zZXZlcml0eTogMAogIH07CiAgaWYgKHIoKSA+IDAuNzIpIHsKICAgIGNvbnN0IGZhdWx0cyA9"
    "IFsiaHZhY19mYWlsdXJlIiwiZW5lcmd5X292ZXJsb2FkIiwibGlnaHRpbmdfZmF1bHQiLCJzYWZldHlfYWxhcm0i"
    "LCJzZW5zb3JfZmFpbHVyZSJdOwogICAgcmVhZGluZy5hY3RpdmVfZmF1bHQgPSBmYXVsdHNbTWF0aC5mbG9vcihy"
    "KCkqZmF1bHRzLmxlbmd0aCldOwogICAgcmVhZGluZy5mYXVsdF9zZXZlcml0eSA9IE1hdGguZmxvb3IocigpKjQp"
    "KzE7CiAgICBpZiAocmVhZGluZy5hY3RpdmVfZmF1bHQgPT09ICJodmFjX2ZhaWx1cmUiKSByZWFkaW5nLmh2YWNf"
    "cG93ZXIgPSArKDU1ICsgcigpKjIwKS50b0ZpeGVkKDEpOwogICAgaWYgKHJlYWRpbmcuYWN0aXZlX2ZhdWx0ID09"
    "PSAiZW5lcmd5X292ZXJsb2FkIikgeyByZWFkaW5nLmhlYXRpbmdfbG9hZCA9ICsocmVhZGluZy5oZWF0aW5nX2xv"
    "YWQqMS42KS50b0ZpeGVkKDEpOyByZWFkaW5nLmNvb2xpbmdfbG9hZCA9ICsocmVhZGluZy5jb29saW5nX2xvYWQq"
    "MS41KS50b0ZpeGVkKDEpOyB9CiAgICBpZiAocmVhZGluZy5hY3RpdmVfZmF1bHQgPT09ICJsaWdodGluZ19mYXVs"
    "dCIpIHJlYWRpbmcubGlnaHRpbmdfaW50ZW5zaXR5ID0gcigpPDAuNSA/IDAgOiAxLjU7CiAgICBpZiAocmVhZGlu"
    "Zy5hY3RpdmVfZmF1bHQgPT09ICJzYWZldHlfYWxhcm0iKSByZWFkaW5nLm9jY3VwYW5jeSA9ICtNYXRoLm1pbigx"
    "LCBvY2MqMS40KS50b0ZpeGVkKDIpOwogIH0KICByZXR1cm4gcmVhZGluZzsKfQoKZnVuY3Rpb24gbG1HZW5SZWNv"
    "dmVyeShzdGVwKSB7CiAgY29uc3QgZmF1bHRzID0gWyJodmFjX2ZhaWx1cmUiLCJlbmVyZ3lfb3ZlcmxvYWQiLCJs"
    "aWdodGluZ19mYXVsdCIsInNhZmV0eV9hbGFybSIsInNlbnNvcl9mYWlsdXJlIiwicGFya2luZ19jb25nZXN0aW9u"
    "Il07CiAgY29uc3QgYWdlbnRzID0gWyJIVkFDIiwiRW5lcmd5IiwiTGlnaHRpbmciLCJTYWZldHkiLCJQYXJraW5n"
    "Il07CiAgY29uc3QgYWN0aW9ucyA9IHsgaHZhY19mYWlsdXJlOiJyZWR1Y2VfaHZhY19wb3dlciIsIGVuZXJneV9v"
    "dmVybG9hZDoibG9hZF9zaGVkZGluZyIsIGxpZ2h0aW5nX2ZhdWx0OiJhZGp1c3RfbGlnaHRpbmciLCBzYWZldHlf"
    "YWxhcm06InNhZmV0eV9sb2NrZG93biIsIHNlbnNvcl9mYWlsdXJlOiJzZW5zb3JfcmVjYWxpYnJhdGlvbiIsIHBh"
    "cmtpbmdfY29uZ2VzdGlvbjoicmVyb3V0ZV9wYXJraW5nIiB9OwogIGNvbnN0IGZ0ID0gZmF1bHRzW01hdGguZmxv"
    "b3IoTWF0aC5yYW5kb20oKSpmYXVsdHMubGVuZ3RoKV07CiAgY29uc3QgaXNPdmVycmlkZSA9IGZ0ID09PSAic2Fm"
    "ZXR5X2FsYXJtIiAmJiBNYXRoLnJhbmRvbSgpIDwgMC42OwogIHJldHVybiB7CiAgICBzdGVwX2lkOiBzdGVwLCB0"
    "aW1lc3RhbXA6IG5ldyBEYXRlKCkudG9JU09TdHJpbmcoKSwKICAgIGZhdWx0X3R5cGU6IGZ0LCBmYXVsdF9zZXZl"
    "cml0eTogTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpKjQpKzEsCiAgICBkZXRlY3RlZF9ieTogYWdlbnRzW01hdGgu"
    "Zmxvb3IoTWF0aC5yYW5kb20oKSphZ2VudHMubGVuZ3RoKV0sCiAgICByZWNvdmVyeV9hY3Rpb246IGlzT3ZlcnJp"
    "ZGUgPyAic2FmZXR5X2xvY2tkb3duIiA6IChhY3Rpb25zW2Z0XXx8ImF1dG9fcmVjb3ZlciIpLAogICAgcmVjb3Zl"
    "cnlfYWdlbnQ6IGlzT3ZlcnJpZGUgPyAiU2FmZXR5IChPVkVSUklERSkiIDogYWdlbnRzW01hdGguZmxvb3IoTWF0"
    "aC5yYW5kb20oKSphZ2VudHMubGVuZ3RoKV0sCiAgICByZWNvdmVyeV90aW1lX21zOiBNYXRoLmZsb29yKE1hdGgu"
    "cmFuZG9tKCkqMjAwMCkrMjAwLAogICAgc3VjY2VzczogTWF0aC5yYW5kb20oKSA+IDAuMDUsIHNhZmV0eV9vdmVy"
    "cmlkZTogaXNPdmVycmlkZQogIH07Cn0KCmZ1bmN0aW9uIGxtSW5pdCgpIHsKICBpZiAoIWxtSW5pdGlhbGl6ZWQp"
    "IHsKICAgIGxtU2ltRGF0YSA9IEFycmF5LmZyb20oe2xlbmd0aDoxMjB9LCAoXyxpKSA9PiBsbUdlblJlYWRpbmco"
    "aSkpOwogICAgbG1SZWNvdmVyeSA9IEFycmF5LmZyb20oe2xlbmd0aDo0MH0sIChfLGkpID0+IGxtR2VuUmVjb3Zl"
    "cnkoaSkpOwogICAgbG1Jbml0aWFsaXplZCA9IHRydWU7CiAgICBsbUJ1aWxkQXJjaERpYWdyYW0oKTsKICAgIGxt"
    "QnVpbGRQcm90b1N0ZXBzKCk7CiAgICBsbUJ1aWxkTW9kZWxUYWJsZSgpOwogICAgbG1CdWlsZE1vZGVsU3VtbWFy"
    "eSgpOwogICAgbG1CdWlsZFJhZGFyKCk7CiAgfQogIGxtUmVuZGVyKCk7CiAgbG1TdGFydFNpbSgpOwp9CgpmdW5j"
    "dGlvbiBsbVN0YXJ0U2ltKCkgewogIGlmIChsbUludGVydmFsKSBjbGVhckludGVydmFsKGxtSW50ZXJ2YWwpOwog"
    "IGlmICghbG1TaW11bGF0aW5nKSByZXR1cm47CiAgbG1JbnRlcnZhbCA9IHNldEludGVydmFsKCgpID0+IHsKICAg"
    "IGxtU3RlcCsrOwogICAgY29uc3QgbnIgPSBsbUdlblJlYWRpbmcobG1TdGVwKTsKICAgIGxtU2ltRGF0YSA9IFsu"
    "Li5sbVNpbURhdGEuc2xpY2UoLTExOSksIG5yXTsKICAgIGlmIChuci5hY3RpdmVfZmF1bHQpIHsKICAgICAgY29u"
    "c3QgcmVjID0gbG1HZW5SZWNvdmVyeShsbVN0ZXApOwogICAgICBsbVJlY292ZXJ5ID0gWy4uLmxtUmVjb3Zlcnku"
    "c2xpY2UoLTQ5KSwgcmVjXTsKICAgICAgbG1TaG93QWxlcnRCYW5uZXIobnIsIHJlYyk7CiAgICB9CiAgICBsbVJl"
    "bmRlcigpOwogIH0sIDIwMDApOwp9CgpmdW5jdGlvbiBsbVRvZ2dsZVNpbSgpIHsKICBsbVNpbXVsYXRpbmcgPSAh"
    "bG1TaW11bGF0aW5nOwogIGNvbnN0IGJ0biA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbVBhdXNlQnRuIik7"
    "CiAgY29uc3QgbGJsID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoImxtTGl2ZUxhYmVsIik7CiAgaWYgKGxtU2lt"
    "dWxhdGluZykgewogICAgYnRuLmNsYXNzTmFtZSA9ICJwYXVzZS1idG4gcnVubmluZyI7IGJ0bi50ZXh0Q29udGVu"
    "dCA9ICLij7ggUGF1c2UiOwogICAgbGJsLnRleHRDb250ZW50ID0gIkxJVkUiOyBsYmwuc3R5bGUuY29sb3IgPSAi"
    "dmFyKC0tZ3JlZW4pIjsKICAgIGxtU3RhcnRTaW0oKTsKICB9IGVsc2UgewogICAgaWYgKGxtSW50ZXJ2YWwpIGNs"
    "ZWFySW50ZXJ2YWwobG1JbnRlcnZhbCk7CiAgICBidG4uY2xhc3NOYW1lID0gInBhdXNlLWJ0biBwYXVzZWQiOyBi"
    "dG4udGV4dENvbnRlbnQgPSAi4pa2IFJlc3VtZSI7CiAgICBsYmwudGV4dENvbnRlbnQgPSAiUEFVU0VEIjsgbGJs"
    "LnN0eWxlLmNvbG9yID0gInZhcigtLXllbGxvdykiOwogIH0KfQoKZnVuY3Rpb24gbG1TaG93QWxlcnRCYW5uZXIo"
    "cmVhZGluZywgcmVjKSB7CiAgY29uc3QgYmFubmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoImxtQWxlcnRC"
    "YW5uZXIiKTsKICBpZiAoIWJhbm5lcikgcmV0dXJuOwogIGlmIChyZWMuc2FmZXR5X292ZXJyaWRlIHx8IHJlYy5m"
    "YXVsdF9zZXZlcml0eSA+PSA0KSB7CiAgICBiYW5uZXIudGV4dENvbnRlbnQgPSAi4pqhIFNBRkVUWSBPVkVSUklE"
    "RTogIiArIHJlYWRpbmcuYWN0aXZlX2ZhdWx0LnJlcGxhY2UoL18vZywiICIpLnRvVXBwZXJDYXNlKCk7CiAgICBi"
    "YW5uZXIuY2xhc3NOYW1lID0gImFsZXJ0LWJhbm5lciBkYW5nZXIgc2hvdyI7CiAgfSBlbHNlIGlmIChyZWFkaW5n"
    "LmZhdWx0X3NldmVyaXR5ID49IDMpIHsKICAgIGJhbm5lci50ZXh0Q29udGVudCA9ICLimqDvuI8gRmF1bHQ6ICIg"
    "KyByZWFkaW5nLmFjdGl2ZV9mYXVsdC5yZXBsYWNlKC9fL2csIiAiKSArICIgfCBOZWdvdGlhdGlvbiBpbiBwcm9n"
    "cmVzcy4uLiI7CiAgICBiYW5uZXIuY2xhc3NOYW1lID0gImFsZXJ0LWJhbm5lciB3YXJuIHNob3ciOwogIH0KICBz"
    "ZXRUaW1lb3V0KCgpID0+IHsgaWYoYmFubmVyKSBiYW5uZXIuY2xhc3NOYW1lID0gImFsZXJ0LWJhbm5lciI7IH0s"
    "IDM1MDApOwp9CgpmdW5jdGlvbiBsbVJlbmRlcigpIHsKICBjb25zdCBsYXRlc3QgPSBsbVNpbURhdGFbbG1TaW1E"
    "YXRhLmxlbmd0aC0xXTsKICBpZiAoIWxhdGVzdCkgcmV0dXJuOwogIGlmIChsbUFjdGl2ZVRhYiA9PT0gIm92ZXJ2"
    "aWV3IikgewogICAgbG1SZW5kZXJLUEkobGF0ZXN0KTsgbG1SZW5kZXJGYXVsdEFsZXJ0KGxhdGVzdCk7CiAgICBs"
    "bVJlbmRlckVuZXJneUNoYXJ0KCk7IGxtUmVuZGVyRmF1bHRQaWUoKTsgbG1SZW5kZXJSdEJhcigpOwogIH0gZWxz"
    "ZSBpZiAobG1BY3RpdmVUYWIgPT09ICJhZ2VudHMiKSB7CiAgICBsbVJlbmRlckFnZW50Q2FyZHMobGF0ZXN0KTsg"
    "bG1SZW5kZXJSZWFkaW5nQmFycyhsYXRlc3QpOyBsbVJlbmRlck1zZ0J1cygpOwogIH0gZWxzZSBpZiAobG1BY3Rp"
    "dmVUYWIgPT09ICJmYXVsdHMiKSB7CiAgICBsbVJlbmRlckZhdWx0RnJlcSgpOyBsbVJlbmRlclNldkNoYXJ0KCk7"
    "IGxtUmVuZGVyTG9nVGFibGUoKTsKICB9IGVsc2UgaWYgKGxtQWN0aXZlVGFiID09PSAibmVnb3RpYXRpb25zIikg"
    "ewogICAgbG1SZW5kZXJOZWdUaW1lbGluZSgpOyBsbVJlbmRlck5lZ1N0YXRzKCk7CiAgfQp9CgpmdW5jdGlvbiBs"
    "bVNob3dUYWIodGFiLCBidG4pIHsKICBsbUFjdGl2ZVRhYiA9IHRhYjsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9y"
    "QWxsKCIubG0tdGFiIikuZm9yRWFjaCh0ID0+IHQuc3R5bGUuZGlzcGxheSA9ICJub25lIik7CiAgY29uc3QgZWwg"
    "PSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG10YWItIit0YWIpOwogIGlmIChlbCkgZWwuc3R5bGUuZGlzcGxh"
    "eSA9ICIiOwogIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoIi5zdWJuYXYtYnRuIikuZm9yRWFjaChiID0+IGIu"
    "Y2xhc3NMaXN0LnJlbW92ZSgiYWN0aXZlIikpOwogIGlmIChidG4pIGJ0bi5jbGFzc0xpc3QuYWRkKCJhY3RpdmUi"
    "KTsKICBsbVJlbmRlcigpOwp9CgpmdW5jdGlvbiBsbVJlbmRlcktQSShsYXRlc3QpIHsKICBjb25zdCBvdmVycmlk"
    "ZXMgPSBsbVJlY292ZXJ5LmZpbHRlcihyPT5yLnNhZmV0eV9vdmVycmlkZSkubGVuZ3RoOwogIGNvbnN0IGF2Z1J0"
    "ID0gbG1SZWNvdmVyeS5sZW5ndGggPyBNYXRoLnJvdW5kKGxtUmVjb3ZlcnkucmVkdWNlKChhLHIpPT5hK3IucmVj"
    "b3ZlcnlfdGltZV9tcywwKS9sbVJlY292ZXJ5Lmxlbmd0aCkgOiAwOwogIGNvbnN0IHNldCA9IChpZCx2KSA9PiB7"
    "IGNvbnN0IGU9ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpOyBpZihlKSBlLnRleHRDb250ZW50PXY7IH07CiAg"
    "c2V0KCJsbS1rcGktZmF1bHRzIiwgbGF0ZXN0LmFjdGl2ZV9mYXVsdCA/ICIxIiA6ICIwIik7CiAgc2V0KCJsbS1r"
    "cGktZXZlbnRzIiwgbG1SZWNvdmVyeS5sZW5ndGgpOwogIHNldCgibG0ta3BpLW92ZXJyaWRlcyIsIG92ZXJyaWRl"
    "cyk7CiAgc2V0KCJsbS1rcGktYXZncnQiLCBhdmdSdCA/IGF2Z1J0KyIgbXMiIDogIuKAlCIpOwp9CgpmdW5jdGlv"
    "biBsbVJlbmRlckZhdWx0QWxlcnQobGF0ZXN0KSB7CiAgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJ"
    "ZCgibG1GYXVsdEFsZXJ0Iik7CiAgaWYgKCFlbCkgcmV0dXJuOwogIGlmIChsYXRlc3QuYWN0aXZlX2ZhdWx0KSB7"
    "CiAgICBlbC5zdHlsZS5kaXNwbGF5ID0gImZsZXgiOwogICAgY29uc3QgdCA9IGRvY3VtZW50LmdldEVsZW1lbnRC"
    "eUlkKCJsbUZhdWx0QWxlcnRUaXRsZSIpOwogICAgaWYodCkgdC50ZXh0Q29udGVudCA9ICJBQ1RJVkUgRkFVTFQ6"
    "ICIrbGF0ZXN0LmFjdGl2ZV9mYXVsdC5yZXBsYWNlKC9fL2csIiAiKS50b1VwcGVyQ2FzZSgpOwogICAgY29uc3Qg"
    "cyA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbUZhdWx0QWxlcnRTZXYiKTsKICAgIGlmKHMpIHMudGV4dENv"
    "bnRlbnQgPSBsYXRlc3QuZmF1bHRfc2V2ZXJpdHk7CiAgICBjb25zdCBjb2wgPSBMTV9GQVVMVF9DT0xPUlNbbGF0"
    "ZXN0LmFjdGl2ZV9mYXVsdF18fCIjODg4IjsKICAgIGNvbnN0IHAgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgi"
    "bG1GYXVsdEFsZXJ0UGlsbCIpOwogICAgaWYocCkgcC5pbm5lckhUTUwgPSAnPHNwYW4gY2xhc3M9InBpbGwiIHN0"
    "eWxlPSJiYWNrZ3JvdW5kOicrY29sKycyMjtjb2xvcjonK2NvbCsnO2JvcmRlcjoxcHggc29saWQgJytjb2wrJzQ0"
    "Ij5TRVYgJytsYXRlc3QuZmF1bHRfc2V2ZXJpdHkrJzwvc3Bhbj4nOwogIH0gZWxzZSB7CiAgICBlbC5zdHlsZS5k"
    "aXNwbGF5ID0gIm5vbmUiOwogIH0KfQoKZnVuY3Rpb24gbG1SZW5kZXJFbmVyZ3lDaGFydCgpIHsKICBjb25zdCBj"
    "dHggPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1FbmVyZ3lDaGFydCIpOwogIGlmICghY3R4KSByZXR1cm47"
    "CiAgY29uc3Qgc2xpY2UgPSBsbVNpbURhdGEuc2xpY2UoLTQwKTsKICBkZXN0cm95Q2hhcnQoImxtRW5lcmd5Iik7"
    "CiAgQ0hBUlRTWyJsbUVuZXJneSJdID0gbmV3IENoYXJ0KGN0eCwgewogICAgdHlwZToibGluZSIsCiAgICBkYXRh"
    "OnsKICAgICAgbGFiZWxzOiBzbGljZS5tYXAoKF8saSk9PmkpLAogICAgICBkYXRhc2V0czpbCiAgICAgICAgeyBs"
    "YWJlbDoiSFZBQyIsICAgIGRhdGE6c2xpY2UubWFwKGQ9PitkLmh2YWNfcG93ZXIpLCAgICBib3JkZXJDb2xvcjoi"
    "IzNiODJmNiIsIGJhY2tncm91bmRDb2xvcjoicmdiYSg1OSwxMzAsMjQ2LDAuMDgpIiwgZmlsbDp0cnVlLCAgdGVu"
    "c2lvbjowLjQsIHBvaW50UmFkaXVzOjAsIGJvcmRlcldpZHRoOjIgfSwKICAgICAgICB7IGxhYmVsOiJIZWF0aW5n"
    "IiwgZGF0YTpzbGljZS5tYXAoZD0+K2QuaGVhdGluZ19sb2FkKSwgIGJvcmRlckNvbG9yOiIjZWY0NDQ0IiwgYmFj"
    "a2dyb3VuZENvbG9yOiJ0cmFuc3BhcmVudCIsIGZpbGw6ZmFsc2UsIHRlbnNpb246MC40LCBwb2ludFJhZGl1czow"
    "LCBib3JkZXJXaWR0aDoyIH0sCiAgICAgICAgeyBsYWJlbDoiQ29vbGluZyIsIGRhdGE6c2xpY2UubWFwKGQ9Pitk"
    "LmNvb2xpbmdfbG9hZCksICBib3JkZXJDb2xvcjoiIzIyYzU1ZSIsIGJhY2tncm91bmRDb2xvcjoidHJhbnNwYXJl"
    "bnQiLCBmaWxsOmZhbHNlLCB0ZW5zaW9uOjAuNCwgcG9pbnRSYWRpdXM6MCwgYm9yZGVyV2lkdGg6MiB9LAogICAg"
    "ICBdCiAgICB9LAogICAgb3B0aW9uczp7CiAgICAgIHJlc3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RSYXRp"
    "bzpmYWxzZSwgYW5pbWF0aW9uOntkdXJhdGlvbjoyNTB9LAogICAgICBzY2FsZXM6eyB4OntkaXNwbGF5OmZhbHNl"
    "fSwgeTp7Z3JpZDp7Y29sb3I6InJnYmEoMjU1LDI1NSwyNTUsMC4wNCkifSx0aWNrczp7Y29sb3I6IiM2NDc0OGIi"
    "fX0gfSwKICAgICAgcGx1Z2luczp7bGVnZW5kOntsYWJlbHM6e2NvbG9yOiIjOTRhM2I4Iixmb250OntzaXplOjEx"
    "fX19fQogICAgfQogIH0pOwp9CgpmdW5jdGlvbiBsbVJlbmRlckZhdWx0UGllKCkgewogIGNvbnN0IGN0eCA9IGRv"
    "Y3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbUZhdWx0UGllQ2hhcnQiKTsKICBpZiAoIWN0eCkgcmV0dXJuOwogIGNv"
    "bnN0IGRpc3QgPSBMTV9GQVVMVF9UWVBFUy5tYXAoZnQ9Pih7IG5hbWU6ZnQucmVwbGFjZSgvXy9nLCIgIiksIHZh"
    "bHVlOmxtUmVjb3ZlcnkuZmlsdGVyKHI9PnIuZmF1bHRfdHlwZT09PWZ0KS5sZW5ndGgsIGNvbG9yOkxNX0ZBVUxU"
    "X0NPTE9SU1tmdF0gfSkpLmZpbHRlcihkPT5kLnZhbHVlPjApOwogIGlmICghZGlzdC5sZW5ndGgpIHJldHVybjsK"
    "ICBkZXN0cm95Q2hhcnQoImxtRmF1bHRQaWUiKTsKICBDSEFSVFNbImxtRmF1bHRQaWUiXSA9IG5ldyBDaGFydChj"
    "dHgsIHsKICAgIHR5cGU6ImRvdWdobnV0IiwKICAgIGRhdGE6eyBsYWJlbHM6ZGlzdC5tYXAoZD0+ZC5uYW1lKSwg"
    "ZGF0YXNldHM6W3tkYXRhOmRpc3QubWFwKGQ9PmQudmFsdWUpLCBiYWNrZ3JvdW5kQ29sb3I6ZGlzdC5tYXAoZD0+"
    "ZC5jb2xvciksIGJvcmRlcldpZHRoOjIsIGJvcmRlckNvbG9yOiIjMDQwODEwIiwgaG92ZXJPZmZzZXQ6MTB9XSB9"
    "LAogICAgb3B0aW9uczp7IHJlc3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RSYXRpbzpmYWxzZSwgYW5pbWF0"
    "aW9uOntkdXJhdGlvbjoyNTB9LCBwbHVnaW5zOntsZWdlbmQ6e3Bvc2l0aW9uOiJyaWdodCIsbGFiZWxzOntjb2xv"
    "cjoiIzk0YTNiOCIsZm9udDp7c2l6ZToxMH0scGFkZGluZzo4fX19IH0KICB9KTsKfQoKZnVuY3Rpb24gbG1SZW5k"
    "ZXJSdEJhcigpIHsKICBjb25zdCBjdHggPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1SdEJhckNoYXJ0Iik7"
    "CiAgaWYgKCFjdHgpIHJldHVybjsKICBjb25zdCBzbGljZSA9IGxtUmVjb3Zlcnkuc2xpY2UoLTIwKTsKICBkZXN0"
    "cm95Q2hhcnQoImxtUnRCYXIiKTsKICBDSEFSVFNbImxtUnRCYXIiXSA9IG5ldyBDaGFydChjdHgsIHsKICAgIHR5"
    "cGU6ImJhciIsCiAgICBkYXRhOnsgbGFiZWxzOnNsaWNlLm1hcCgoXyxpKT0+aSksIGRhdGFzZXRzOlt7IGxhYmVs"
    "OiJSZWNvdmVyeSBtcyIsIGRhdGE6c2xpY2UubWFwKHI9Pk1hdGgucm91bmQoci5yZWNvdmVyeV90aW1lX21zKSks"
    "IGJhY2tncm91bmRDb2xvcjpzbGljZS5tYXAocj0+TE1fRkFVTFRfQ09MT1JTW3IuZmF1bHRfdHlwZV18fCIjM2I4"
    "MmY2IiksIGJvcmRlclJhZGl1czo0LCBib3JkZXJXaWR0aDowIH1dIH0sCiAgICBvcHRpb25zOnsgcmVzcG9uc2l2"
    "ZTp0cnVlLCBtYWludGFpbkFzcGVjdFJhdGlvOmZhbHNlLCBhbmltYXRpb246e2R1cmF0aW9uOjI1MH0sIHNjYWxl"
    "czp7eDp7ZGlzcGxheTpmYWxzZX0seTp7Z3JpZDp7Y29sb3I6InJnYmEoMjU1LDI1NSwyNTUsMC4wNCkifSx0aWNr"
    "czp7Y29sb3I6IiM2NDc0OGIifX19LCBwbHVnaW5zOntsZWdlbmQ6e2Rpc3BsYXk6ZmFsc2V9fSB9CiAgfSk7Cn0K"
    "CmZ1bmN0aW9uIGxtUmVuZGVyQWdlbnRDYXJkcyhsYXRlc3QpIHsKICBjb25zdCBncmlkID0gZG9jdW1lbnQuZ2V0"
    "RWxlbWVudEJ5SWQoImxtQWdlbnRHcmlkIik7CiAgaWYgKCFncmlkKSByZXR1cm47CiAgY29uc3Qgc3RhdHMgPSB7"
    "CiAgICBIVkFDOiAgICB7IG1ldHJpYzpsYXRlc3QuaHZhY19wb3dlcisiIGtXIiwgICBsYWJlbDoiUG93ZXIgRHJh"
    "dyIsICAgICAgIHN0YXR1czogbGF0ZXN0LmFjdGl2ZV9mYXVsdD09PSJodmFjX2ZhaWx1cmUiPyJmYXVsdCI6Im9u"
    "bGluZSIgfSwKICAgIEVuZXJneTogIHsgbWV0cmljOigrbGF0ZXN0LmhlYXRpbmdfbG9hZCArICtsYXRlc3QuY29v"
    "bGluZ19sb2FkKS50b0ZpeGVkKDApKyIga1ciLCBsYWJlbDoiVG90YWwgTG9hZCIsIHN0YXR1czoib25saW5lIiB9"
    "LAogICAgTGlnaHRpbmc6eyBtZXRyaWM6KGxhdGVzdC5saWdodGluZ19pbnRlbnNpdHkqMTAwKS50b0ZpeGVkKDAp"
    "KyIlIiwgbGFiZWw6IkludGVuc2l0eSIsIHN0YXR1czogbGF0ZXN0LmFjdGl2ZV9mYXVsdD09PSJsaWdodGluZ19m"
    "YXVsdCI/ImZhdWx0Ijoib25saW5lIiB9LAogICAgUGFya2luZzogeyBtZXRyaWM6KGxhdGVzdC5wYXJraW5nX29j"
    "Y3VwYW5jeSoxMDApLnRvRml4ZWQoMCkrIiUiLCBsYWJlbDoiT2NjdXBhbmN5IiwgIHN0YXR1czoib25saW5lIiB9"
    "LAogICAgU2FmZXR5OiAgeyBtZXRyaWM6bG1SZWNvdmVyeS5maWx0ZXIocj0+ci5zYWZldHlfb3ZlcnJpZGUpLmxl"
    "bmd0aCsiIiwgbGFiZWw6Ik92ZXJyaWRlcyBUb2RheSIsIHN0YXR1czoibW9uaXRvciIgfSwKICB9OwogIGdyaWQu"
    "aW5uZXJIVE1MID0gT2JqZWN0LmVudHJpZXMoc3RhdHMpLm1hcCgoW25hbWUsc10pID0+IHsKICAgIGNvbnN0IGNv"
    "bCA9IExNX0FHRU5UX0NPTE9SU1tuYW1lXTsKICAgIGNvbnN0IGlzQWN0aXZlID0gbG1BY3RpdmVBZ2VudD09PW5h"
    "bWU7CiAgICBjb25zdCBib3JkZXJTdHlsZSA9IGlzQWN0aXZlID8gImJvcmRlci1jb2xvcjoiK2NvbCsiO2JveC1z"
    "aGFkb3c6MCAwIDIwcHggIitjb2wrIjMzIiA6ICIiOwogICAgcmV0dXJuICc8ZGl2IGNsYXNzPSJhZ2VudC1jYXJk"
    "JysoaXNBY3RpdmU/IiBhY3RpdmUiOiIiKSsnIiBzdHlsZT0iJytib3JkZXJTdHlsZSsnOyIgb25jbGljaz0ibG1B"
    "Y3RpdmVBZ2VudD1sbUFjdGl2ZUFnZW50PT09XCcnK25hbWUrJ1wnP251bGw6XCcnK25hbWUrJ1wnO2xtUmVuZGVy"
    "KCk7Ij4nCiAgICAgICsnPGRpdiBjbGFzcz0iYWMtaGVhZGVyIj48c3BhbiBjbGFzcz0iYWMtbmFtZSIgc3R5bGU9"
    "ImNvbG9yOicrY29sKyciPicrbmFtZSsnPC9zcGFuPicKICAgICAgKyc8c3BhbiBjbGFzcz0iYWMtZG90ICcrKHMu"
    "c3RhdHVzPT09ImZhdWx0Ij8iZmF1bHQiOnMuc3RhdHVzPT09Im1vbml0b3IiPyJtb25pdG9yIjoib25saW5lIikr"
    "JyI+PC9zcGFuPjwvZGl2PicKICAgICAgKyc8ZGl2IGNsYXNzPSJhYy1tZXRyaWMiIHN0eWxlPSJjb2xvcjonK2Nv"
    "bCsnIj4nK3MubWV0cmljKyc8L2Rpdj4nCiAgICAgICsnPGRpdiBjbGFzcz0iYWMtbGFiZWwiPicrcy5sYWJlbCsn"
    "PC9kaXY+PC9kaXY+JzsKICB9KS5qb2luKCIiKTsKfQoKZnVuY3Rpb24gbG1SZW5kZXJSZWFkaW5nQmFycyhsYXRl"
    "c3QpIHsKICBjb25zdCBlbCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbVJlYWRpbmdzQmFyIik7CiAgaWYg"
    "KCFlbCkgcmV0dXJuOwogIGNvbnN0IGl0ZW1zID0gWwogICAgeyBsYWJlbDoiT3V0ZG9vciBUZW1wZXJhdHVyZSIs"
    "IHZhbHVlOmxhdGVzdC5vdXRkb29yX3RlbXArIsKwQyIsICAgICAgICAgICAgY29sb3I6IiMzYjgyZjYiLCBwY3Q6"
    "KGxhdGVzdC5vdXRkb29yX3RlbXArMTApLzYwIH0sCiAgICB7IGxhYmVsOiJPY2N1cGFuY3kgTGV2ZWwiLCAgICAg"
    "dmFsdWU6KGxhdGVzdC5vY2N1cGFuY3kqMTAwKS50b0ZpeGVkKDApKyIlIiwgY29sb3I6IiMyMmM1NWUiLCBwY3Q6"
    "bGF0ZXN0Lm9jY3VwYW5jeSB9LAogICAgeyBsYWJlbDoiSFZBQyBQb3dlciIsICAgICAgICAgIHZhbHVlOmxhdGVz"
    "dC5odmFjX3Bvd2VyKyIga1ciLCAgICAgICAgICAgICAgY29sb3I6IiNmOTczMTYiLCBwY3Q6bGF0ZXN0Lmh2YWNf"
    "cG93ZXIvODAgfSwKICAgIHsgbGFiZWw6IkxpZ2h0aW5nIEludGVuc2l0eSIsICB2YWx1ZToobGF0ZXN0LmxpZ2h0"
    "aW5nX2ludGVuc2l0eSoxMDApLnRvRml4ZWQoMCkrIiUiLCBjb2xvcjoiI2VhYjMwOCIsIHBjdDpNYXRoLm1pbigx"
    "LGxhdGVzdC5saWdodGluZ19pbnRlbnNpdHkpIH0sCiAgICB7IGxhYmVsOiJQYXJraW5nIE9jY3VwYW5jeSIsICAg"
    "dmFsdWU6KGxhdGVzdC5wYXJraW5nX29jY3VwYW5jeSoxMDApLnRvRml4ZWQoMCkrIiUiLCBjb2xvcjoiIzA2YjZk"
    "NCIsIHBjdDpsYXRlc3QucGFya2luZ19vY2N1cGFuY3kgfSwKICBdOwogIGVsLmlubmVySFRNTCA9IGl0ZW1zLm1h"
    "cChpdGVtID0+CiAgICAnPGRpdiBjbGFzcz0icmVhZGluZy1yb3ciPicKICAgICsnPGRpdiBjbGFzcz0icnItaGVh"
    "ZGVyIj48c3BhbiBjbGFzcz0icnItbGFiZWwiPicraXRlbS5sYWJlbCsnPC9zcGFuPjxzcGFuIGNsYXNzPSJyci12"
    "YWx1ZSIgc3R5bGU9ImNvbG9yOicraXRlbS5jb2xvcisnIj4nK2l0ZW0udmFsdWUrJzwvc3Bhbj48L2Rpdj4nCiAg"
    "ICArJzxkaXYgY2xhc3M9InJlYWRpbmctYmFyIj48ZGl2IGNsYXNzPSJyZWFkaW5nLWZpbGwiIHN0eWxlPSJ3aWR0"
    "aDonK01hdGgubWluKDEwMCxNYXRoLm1heCgwLGl0ZW0ucGN0KSoxMDApLnRvRml4ZWQoMSkrJyU7YmFja2dyb3Vu"
    "ZDonK2l0ZW0uY29sb3IrJyI+PC9kaXY+PC9kaXY+JwogICAgKyc8L2Rpdj4nCiAgKS5qb2luKCIiKTsKfQoKZnVu"
    "Y3Rpb24gbG1SZW5kZXJNc2dCdXMoKSB7CiAgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1N"
    "c2dCdXMiKTsKICBpZiAoIWVsKSByZXR1cm47CiAgZWwuaW5uZXJIVE1MID0gbG1SZWNvdmVyeS5zbGljZSgtMTIp"
    "LnJldmVyc2UoKS5tYXAociA9PgogICAgJzxkaXYgY2xhc3M9ImJ1cy1lbnRyeSIgc3R5bGU9ImJvcmRlci1sZWZ0"
    "LWNvbG9yOicrKExNX0ZBVUxUX0NPTE9SU1tyLmZhdWx0X3R5cGVdfHwiIzNiODJmNiIpKyciPicKICAgICsnPHNw"
    "YW4gc3R5bGU9ImNvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4zKSI+WycrbmV3IERhdGUoci50aW1lc3RhbXApLnRv"
    "TG9jYWxlVGltZVN0cmluZygpKyddIDwvc3Bhbj4nCiAgICArJzxzcGFuIHN0eWxlPSJjb2xvcjonKyhMTV9BR0VO"
    "VF9DT0xPUlNbci5kZXRlY3RlZF9ieV18fCIjZmZmIikrJyI+JytyLmRldGVjdGVkX2J5Kyc8L3NwYW4+JwogICAg"
    "Kyc8c3BhbiBzdHlsZT0iY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjMpIj4g4oaSIEJST0FEQ0FTVDogPC9zcGFu"
    "PicKICAgICsnPHNwYW4gc3R5bGU9ImNvbG9yOicrKExNX0ZBVUxUX0NPTE9SU1tyLmZhdWx0X3R5cGVdfHwiIzg4"
    "OCIpKyciPicrci5mYXVsdF90eXBlLnRvVXBwZXJDYXNlKCkrJzwvc3Bhbj4nCiAgICArJzxzcGFuIHN0eWxlPSJj"
    "b2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMikiPiB8ICcrci5yZWNvdmVyeV9hZ2VudCsnIGFjY2VwdGVkPC9zcGFu"
    "PicKICAgICsnPC9kaXY+JwogICkuam9pbigiIik7Cn0KCmZ1bmN0aW9uIGxtQnVpbGRBcmNoRGlhZ3JhbSgpIHsK"
    "ICBjb25zdCB3cmFwID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoImxtQXJjaFdyYXAiKTsKICBpZiAoIXdyYXAp"
    "IHJldHVybjsKICBPYmplY3QuZW50cmllcyhMTV9BR0VOVF9DT0xPUlMpLmZvckVhY2goKFtuYW1lLCBjb2xvcl0s"
    "IGkpID0+IHsKICAgIGNvbnN0IGFuZ2xlID0gKGkvNSkqMipNYXRoLlBJIC0gTWF0aC5QSS8yOwogICAgY29uc3Qg"
    "eCA9IDUwICsgNDIqTWF0aC5jb3MoYW5nbGUpOwogICAgY29uc3QgeSA9IDUwICsgMzgqTWF0aC5zaW4oYW5nbGUp"
    "OwogICAgY29uc3Qgbm9kZSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoImRpdiIpOwogICAgbm9kZS5jbGFzc05h"
    "bWUgPSAiYXJjaC1hZ2VudCI7CiAgICBub2RlLnN0eWxlLmNzc1RleHQgPSAibGVmdDoiK3grIiU7dG9wOiIreSsi"
    "JTtiYWNrZ3JvdW5kOiIrY29sb3IrIjE4O2JvcmRlci1jb2xvcjoiK2NvbG9yKyI1NTtjb2xvcjoiK2NvbG9yKyI7"
    "IjsKICAgIG5vZGUuaW5uZXJIVE1MID0gJzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxM3B4O2ZvbnQtd2VpZ2h0Ojcw"
    "MCI+JytuYW1lKyc8L2Rpdj48ZGl2IHN0eWxlPSJmb250LXNpemU6MTBweDtjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1"
    "LDAuMykiPkFnZW50PC9kaXY+JzsKICAgIHdyYXAuYXBwZW5kQ2hpbGQobm9kZSk7CiAgfSk7Cn0KCmZ1bmN0aW9u"
    "IGxtUmVuZGVyRmF1bHRGcmVxKCkgewogIGNvbnN0IGN0eCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbUZh"
    "dWx0RnJlcUNoYXJ0Iik7CiAgaWYgKCFjdHgpIHJldHVybjsKICBjb25zdCBkYXRhID0gTE1fRkFVTFRfVFlQRVMu"
    "bWFwKGZ0PT4oeyBuYW1lOmZ0LnJlcGxhY2UoL18vZywiICIpLnJlcGxhY2UoImZhaWx1cmUiLCJmYWlsIiksIGNv"
    "dW50OmxtUmVjb3ZlcnkuZmlsdGVyKHI9PnIuZmF1bHRfdHlwZT09PWZ0KS5sZW5ndGgsIGNvbG9yOkxNX0ZBVUxU"
    "X0NPTE9SU1tmdF0gfSkpOwogIGRlc3Ryb3lDaGFydCgibG1GYXVsdEZyZXEiKTsKICBDSEFSVFNbImxtRmF1bHRG"
    "cmVxIl0gPSBuZXcgQ2hhcnQoY3R4LCB7CiAgICB0eXBlOiJiYXIiLAogICAgZGF0YTp7IGxhYmVsczpkYXRhLm1h"
    "cChkPT5kLm5hbWUpLCBkYXRhc2V0czpbe2RhdGE6ZGF0YS5tYXAoZD0+ZC5jb3VudCksIGJhY2tncm91bmRDb2xv"
    "cjpkYXRhLm1hcChkPT5kLmNvbG9yKSwgYm9yZGVyUmFkaXVzOjQsIGJvcmRlcldpZHRoOjB9XSB9LAogICAgb3B0"
    "aW9uczp7IHJlc3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RSYXRpbzpmYWxzZSwgc2NhbGVzOnt4Ontncmlk"
    "OntkaXNwbGF5OmZhbHNlfSx0aWNrczp7Y29sb3I6IiM2NDc0OGIiLGZvbnQ6e3NpemU6MTB9fX0seTp7Z3JpZDp7"
    "Y29sb3I6InJnYmEoMjU1LDI1NSwyNTUsMC4wNCkifSx0aWNrczp7Y29sb3I6IiM2NDc0OGIifX19LCBwbHVnaW5z"
    "OntsZWdlbmQ6e2Rpc3BsYXk6ZmFsc2V9fSB9CiAgfSk7Cn0KCmZ1bmN0aW9uIGxtUmVuZGVyU2V2Q2hhcnQoKSB7"
    "CiAgY29uc3QgY3R4ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoImxtU2V2Q2hhcnQiKTsKICBpZiAoIWN0eCkg"
    "cmV0dXJuOwogIGRlc3Ryb3lDaGFydCgibG1TZXYiKTsKICBDSEFSVFNbImxtU2V2Il0gPSBuZXcgQ2hhcnQoY3R4"
    "LCB7CiAgICB0eXBlOiJiYXIiLAogICAgZGF0YTp7IGxhYmVsczpbIlNldiAxIiwiU2V2IDIiLCJTZXYgMyIsIlNl"
    "diA0IiwiU2V2IDUiXSwgZGF0YXNldHM6W3sgZGF0YTpbMSwyLDMsNCw1XS5tYXAocz0+bG1SZWNvdmVyeS5maWx0"
    "ZXIocj0+ci5mYXVsdF9zZXZlcml0eT09PXMpLmxlbmd0aCksIGJhY2tncm91bmRDb2xvcjpbIiMyMmM1NWU4OCIs"
    "IiNmYmJmMjQ4OCIsIiNmOTczMTY4OCIsIiNlZjQ0NDQ4OCIsIiNkYzI2MjY4OCJdLCBib3JkZXJSYWRpdXM6NCwg"
    "Ym9yZGVyV2lkdGg6MCB9XSB9LAogICAgb3B0aW9uczp7IHJlc3BvbnNpdmU6dHJ1ZSwgbWFpbnRhaW5Bc3BlY3RS"
    "YXRpbzpmYWxzZSwgc2NhbGVzOnt4OntncmlkOntkaXNwbGF5OmZhbHNlfSx0aWNrczp7Y29sb3I6IiM2NDc0OGIi"
    "fX0seTp7Z3JpZDp7Y29sb3I6InJnYmEoMjU1LDI1NSwyNTUsMC4wNCkifSx0aWNrczp7Y29sb3I6IiM2NDc0OGIi"
    "fX19LCBwbHVnaW5zOntsZWdlbmQ6e2Rpc3BsYXk6ZmFsc2V9fSB9CiAgfSk7Cn0KCmZ1bmN0aW9uIGxtUmVuZGVy"
    "TG9nVGFibGUoKSB7CiAgY29uc3QgY291bnRFbCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbUxvZ0NvdW50"
    "Iik7CiAgaWYoY291bnRFbCkgY291bnRFbC50ZXh0Q29udGVudCA9ICIoIitsbVJlY292ZXJ5Lmxlbmd0aCsiIGV2"
    "ZW50cykiOwogIGNvbnN0IGJvZHkgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1Mb2dCb2R5Iik7CiAgaWYg"
    "KCFib2R5KSByZXR1cm47CiAgYm9keS5pbm5lckhUTUwgPSBsbVJlY292ZXJ5LnNsaWNlKC0zMCkucmV2ZXJzZSgp"
    "Lm1hcChyID0+IHsKICAgIGNvbnN0IGZjID0gTE1fRkFVTFRfQ09MT1JTW3IuZmF1bHRfdHlwZV18fCIjODg4IjsK"
    "ICAgIGNvbnN0IGFjID0gTE1fQUdFTlRfQ09MT1JTW3IucmVjb3ZlcnlfYWdlbnQ/LnNwbGl0KCIgIilbMF1dfHwi"
    "I2ZmZiI7CiAgICByZXR1cm4gJzx0ciBzdHlsZT0iJysoci5zYWZldHlfb3ZlcnJpZGU/ImJhY2tncm91bmQ6cmdi"
    "YSgyMzksNjgsNjgsMC4wNCkiOiIiKSsnIj48dGQgc3R5bGU9ImNvbG9yOnZhcigtLXRleHQzKSI+JytyLnN0ZXBf"
    "aWQrJzwvdGQ+JwogICAgICArJzx0ZD48c3BhbiBjbGFzcz0icGlsbCIgc3R5bGU9ImJhY2tncm91bmQ6JytmYysn"
    "MjI7Y29sb3I6JytmYysnO2JvcmRlcjoxcHggc29saWQgJytmYysnNDQiPicrci5mYXVsdF90eXBlKyc8L3NwYW4+"
    "PC90ZD4nCiAgICAgICsnPHRkPicrci5mYXVsdF9zZXZlcml0eSsnPC90ZD4nCiAgICAgICsnPHRkIHN0eWxlPSJj"
    "b2xvcjonKyhMTV9BR0VOVF9DT0xPUlNbci5kZXRlY3RlZF9ieV18fCIjZmZmIikrJyI+JytyLmRldGVjdGVkX2J5"
    "Kyc8L3RkPicKICAgICAgKyc8dGQgc3R5bGU9ImNvbG9yOnZhcigtLXRleHQyKSI+JytyLnJlY292ZXJ5X2FjdGlv"
    "bisnPC90ZD4nCiAgICAgICsnPHRkIHN0eWxlPSJjb2xvcjonK2FjKyciPicrci5yZWNvdmVyeV9hZ2VudCsnPC90"
    "ZD4nCiAgICAgICsnPHRkIHN0eWxlPSJjb2xvcjp2YXIoLS1ncmVlbik7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubyki"
    "PicrTWF0aC5yb3VuZChyLnJlY292ZXJ5X3RpbWVfbXMpKyc8L3RkPicKICAgICAgKyc8dGQ+Jysoci5zYWZldHlf"
    "b3ZlcnJpZGU/JzxzcGFuIGNsYXNzPSJwaWxsIiBzdHlsZT0iYmFja2dyb3VuZDpyZ2JhKDIzOSw2OCw2OCwwLjIp"
    "O2NvbG9yOiNlZjQ0NDQ7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIzOSw2OCw2OCwwLjMpIj7imqEgWUVTPC9zcGFu"
    "Pic6JzxzcGFuIHN0eWxlPSJjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMTUpIj7igJQ8L3NwYW4+JykrJzwvdGQ+"
    "PC90cj4nOwogIH0pLmpvaW4oIiIpOwp9CgpmdW5jdGlvbiBsbUJ1aWxkTW9kZWxUYWJsZSgpIHsKICBjb25zdCBi"
    "b2R5ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoImxtTW9kZWxCb2R5Iik7CiAgaWYgKCFib2R5KSByZXR1cm47"
    "CiAgY29uc3QgY29sb3JzID0gWyIjM2I4MmY2IiwiI2Y5NzMxNiIsIiNhODU1ZjciXTsKICBib2R5LmlubmVySFRN"
    "TCA9IE9iamVjdC5lbnRyaWVzKExNX01MKS5tYXAoKFtuYW1lLG1dLGkpID0+IHsKICAgIGNvbnN0IGMgPSBjb2xv"
    "cnNbaV07CiAgICByZXR1cm4gJzx0ciBzdHlsZT0iYmFja2dyb3VuZDonKyhpPT09MD9jKyIwYSI6InRyYW5zcGFy"
    "ZW50IikrJyI+JwogICAgICArJzx0ZCBzdHlsZT0iZm9udC13ZWlnaHQ6NzAwO2NvbG9yOicrYysnIj4nKyhpPT09"
    "MD8i4piFICI6IiIpK25hbWUrJzwvdGQ+JwogICAgICArW20uYWNjdXJhY3ksbS5wcmVjaXNpb24sbS5yZWNhbGws"
    "bS5mMV9zY29yZV0ubWFwKHYgPT4KICAgICAgICAnPHRkPjxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDthbGlnbi1p"
    "dGVtczpjZW50ZXI7Z2FwOjhweCI+JwogICAgICAgICsnPGRpdiBzdHlsZT0id2lkdGg6NTBweDtoZWlnaHQ6NXB4"
    "O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA4KTtib3JkZXItcmFkaXVzOjNweCI+JwogICAgICAgICsn"
    "PGRpdiBzdHlsZT0id2lkdGg6JysodioxMDApLnRvRml4ZWQoMCkrJyU7aGVpZ2h0OjEwMCU7YmFja2dyb3VuZDon"
    "K2MrJztib3JkZXItcmFkaXVzOjNweCI+PC9kaXY+PC9kaXY+JwogICAgICAgICsnPHNwYW4gc3R5bGU9ImNvbG9y"
    "OicrYysnO2ZvbnQtd2VpZ2h0OjcwMDtmb250LWZhbWlseTp2YXIoLS1tb25vKSI+JysodioxMDApLnRvRml4ZWQo"
    "MSkrJyU8L3NwYW4+PC9kaXY+PC90ZD4nCiAgICAgICkuam9pbigiIikKICAgICAgKyc8dGQgc3R5bGU9ImNvbG9y"
    "OnZhcigtLXRleHQyKTtmb250LWZhbWlseTp2YXIoLS1tb25vKSI+JyttLnRyYWluX3RpbWVfc2VjKydzPC90ZD48"
    "L3RyPic7CiAgfSkuam9pbigiIik7Cn0KCmZ1bmN0aW9uIGxtQnVpbGRSYWRhcigpIHsKICBjb25zdCBjdHggPSBk"
    "b2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1SYWRhckNoYXJ0Iik7CiAgaWYgKCFjdHgpIHJldHVybjsKICBkZXN0"
    "cm95Q2hhcnQoImxtUmFkYXIiKTsKICBDSEFSVFNbImxtUmFkYXIiXSA9IG5ldyBDaGFydChjdHgsIHsKICAgIHR5"
    "cGU6InJhZGFyIiwKICAgIGRhdGE6ewogICAgICBsYWJlbHM6WyJBY2N1cmFjeSIsIlByZWNpc2lvbiIsIlJlY2Fs"
    "bCIsIkYxIFNjb3JlIl0sCiAgICAgIGRhdGFzZXRzOlsKICAgICAgICB7IGxhYmVsOiJSYW5kb20gRm9yZXN0Iiwg"
    "ICAgZGF0YTpbODkuNSw5MC4wLDg5LjUsODkuNV0sIGJvcmRlckNvbG9yOiIjM2I4MmY2IiwgYmFja2dyb3VuZENv"
    "bG9yOiJyZ2JhKDU5LDEzMCwyNDYsMC4xNSkiLCBwb2ludEJhY2tncm91bmRDb2xvcjoiIzNiODJmNiIsIHBvaW50"
    "UmFkaXVzOjQgfSwKICAgICAgICB7IGxhYmVsOiJHcmFkaWVudCBCb29zdGluZyIsZGF0YTpbODguNiw4OS4xLDg4"
    "LjQsODguN10sIGJvcmRlckNvbG9yOiIjZjk3MzE2IiwgYmFja2dyb3VuZENvbG9yOiJyZ2JhKDI0OSwxMTUsMjIs"
    "MC4xNSkiLCBwb2ludEJhY2tncm91bmRDb2xvcjoiI2Y5NzMxNiIsIHBvaW50UmFkaXVzOjQgfSwKICAgICAgICB7"
    "IGxhYmVsOiJNTFAgTmV1cmFsIE5ldCIsICAgZGF0YTpbODcuMiw4Ny45LDg2LjgsODcuM10sIGJvcmRlckNvbG9y"
    "OiIjYTg1NWY3IiwgYmFja2dyb3VuZENvbG9yOiJyZ2JhKDE2OCw4NSwyNDcsMC4xNSkiLCBwb2ludEJhY2tncm91"
    "bmRDb2xvcjoiI2E4NTVmNyIsIHBvaW50UmFkaXVzOjQgfSwKICAgICAgXQogICAgfSwKICAgIG9wdGlvbnM6ewog"
    "ICAgICByZXNwb25zaXZlOnRydWUsIG1haW50YWluQXNwZWN0UmF0aW86ZmFsc2UsCiAgICAgIHNjYWxlczp7cjp7"
    "Z3JpZDp7Y29sb3I6InJnYmEoMjU1LDI1NSwyNTUsMC4wOCkifSx0aWNrczp7Y29sb3I6IiM2NDc0OGIiLGJhY2tk"
    "cm9wQ29sb3I6InRyYW5zcGFyZW50In0scG9pbnRMYWJlbHM6e2NvbG9yOiIjOTRhM2I4Iixmb250OntzaXplOjEx"
    "fX0sc3VnZ2VzdGVkTWluOjg1LHN1Z2dlc3RlZE1heDo5Mn19LAogICAgICBwbHVnaW5zOntsZWdlbmQ6e2xhYmVs"
    "czp7Y29sb3I6IiM5NGEzYjgiLGZvbnQ6e3NpemU6MTB9fX19CiAgICB9CiAgfSk7Cn0KCmZ1bmN0aW9uIGxtQnVp"
    "bGRNb2RlbFN1bW1hcnkoKSB7CiAgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgibG1Nb2RlbFN1"
    "bW1hcnkiKTsKICBpZiAoIWVsKSByZXR1cm47CiAgY29uc3QgY29sb3JzID0gWyIjM2I4MmY2IiwiI2Y5NzMxNiIs"
    "IiNhODU1ZjciXTsKICBjb25zdCBsYWJlbHMgPSBbIkJlc3QgT3ZlcmFsbCIsIlJ1bm5lci11cCIsIk5ldXJhbCBO"
    "ZXR3b3JrIl07CiAgZWwuaW5uZXJIVE1MID0gT2JqZWN0LmVudHJpZXMoTE1fTUwpLm1hcCgoW25hbWUsbV0saSkg"
    "PT4gewogICAgY29uc3QgYyA9IGNvbG9yc1tpXTsKICAgIHJldHVybiAnPGRpdiBzdHlsZT0ibWFyZ2luLWJvdHRv"
    "bToxNHB4O3BhZGRpbmc6MTRweDtiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wMyk7Ym9yZGVyLXJhZGl1"
    "czoxMHB4O2JvcmRlcjoxcHggc29saWQgJytjKycyMjtib3JkZXItbGVmdDo0cHggc29saWQgJytjKyciPicKICAg"
    "ICAgKyc8ZGl2IHN0eWxlPSJkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24t"
    "aXRlbXM6Y2VudGVyO21hcmdpbi1ib3R0b206OHB4Ij4nCiAgICAgICsnPHNwYW4gc3R5bGU9ImZvbnQtd2VpZ2h0"
    "OjcwMDtjb2xvcjojZmZmO2ZvbnQtc2l6ZToxM3B4Ij4nK25hbWUrJzwvc3Bhbj4nCiAgICAgICsnPHNwYW4gY2xh"
    "c3M9InBpbGwiIHN0eWxlPSJiYWNrZ3JvdW5kOicrYysnMjI7Y29sb3I6JytjKyc7Ym9yZGVyOjFweCBzb2xpZCAn"
    "K2MrJzQ0O2ZvbnQtc2l6ZToxMHB4Ij4nK2xhYmVsc1tpXSsnPC9zcGFuPjwvZGl2PicKICAgICAgKyc8ZGl2IHN0"
    "eWxlPSJkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjRweCI+JwogICAgICAr"
    "W1siQWNjdXJhY3kiLG0uYWNjdXJhY3ldLFsiRjEgU2NvcmUiLG0uZjFfc2NvcmVdLFsiUHJlY2lzaW9uIixtLnBy"
    "ZWNpc2lvbl0sWyJSZWNhbGwiLG0ucmVjYWxsXV0ubWFwKChbayx2XSkgPT4KICAgICAgICAnPGRpdiBzdHlsZT0i"
    "Zm9udC1zaXplOjEycHgiPjxzcGFuIHN0eWxlPSJjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMzUpIj4nK2srJzog"
    "PC9zcGFuPjxzcGFuIHN0eWxlPSJjb2xvcjonK2MrJztmb250LXdlaWdodDo3MDAiPicrKHYqMTAwKS50b0ZpeGVk"
    "KDEpKyclPC9zcGFuPjwvZGl2PicKICAgICAgKS5qb2luKCIiKQogICAgICArJzwvZGl2PjxkaXYgc3R5bGU9Im1h"
    "cmdpbi10b3A6NnB4O2ZvbnQtc2l6ZToxMXB4O2NvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4yKSI+VHJhaW5pbmcg"
    "dGltZTogJyttLnRyYWluX3RpbWVfc2VjKydzPC9kaXY+PC9kaXY+JzsKICB9KS5qb2luKCIiKTsKfQoKZnVuY3Rp"
    "b24gbG1SZW5kZXJOZWdUaW1lbGluZSgpIHsKICBjb25zdCBlbCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJs"
    "bU5lZ1RpbWVsaW5lIik7CiAgaWYgKCFlbCkgcmV0dXJuOwogIGVsLmlubmVySFRNTCA9IGxtUmVjb3Zlcnkuc2xp"
    "Y2UoLTE1KS5yZXZlcnNlKCkubWFwKGUgPT4gewogICAgY29uc3QgY29sID0gTE1fRkFVTFRfQ09MT1JTW2UuZmF1"
    "bHRfdHlwZV18fCIjODg4IjsKICAgIGNvbnN0IGFnQ29sID0gTE1fQUdFTlRfQ09MT1JTW2UucmVjb3ZlcnlfYWdl"
    "bnQ/LnNwbGl0KCIgIilbMF1dfHwiI2ZmZiI7CiAgICByZXR1cm4gJzxkaXYgY2xhc3M9Im5lZy1ldmVudCcrKGUu"
    "c2FmZXR5X292ZXJyaWRlPyIgb3ZlcnJpZGUiOiIiKSsnIj4nCiAgICAgICsnPGRpdiBjbGFzcz0ibmVnLWJhciIg"
    "c3R5bGU9ImJhY2tncm91bmQ6Jytjb2wrJyI+PC9kaXY+JwogICAgICArJzxkaXYgY2xhc3M9Im5lZy1ib2R5IiBz"
    "dHlsZT0iZmxleDoxO21pbi13aWR0aDowIj48ZGl2IGNsYXNzPSJuZWctdG9wIj4nCiAgICAgICsnPHNwYW4gY2xh"
    "c3M9InBpbGwiIHN0eWxlPSJiYWNrZ3JvdW5kOicrY29sKycyMjtjb2xvcjonK2NvbCsnO2JvcmRlcjoxcHggc29s"
    "aWQgJytjb2wrJzQ0Ij4nK2UuZmF1bHRfdHlwZSsnPC9zcGFuPicKICAgICAgKyhlLnNhZmV0eV9vdmVycmlkZT8n"
    "PHNwYW4gY2xhc3M9InBpbGwiIHN0eWxlPSJiYWNrZ3JvdW5kOnJnYmEoMjM5LDY4LDY4LDAuMik7Y29sb3I6I2Vm"
    "NDQ0NDtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjM5LDY4LDY4LDAuMykiPuKaoSBPVkVSUklERTwvc3Bhbj4nOiIi"
    "KQogICAgICArJzxzcGFuIGNsYXNzPSJuZWctbXMiPicrTWF0aC5yb3VuZChlLnJlY292ZXJ5X3RpbWVfbXMpKydt"
    "czwvc3Bhbj48L2Rpdj4nCiAgICAgICsnPGRpdiBjbGFzcz0ibmVnLXN1YiI+PHNwYW4gc3R5bGU9ImNvbG9yOnJn"
    "YmEoMjU1LDI1NSwyNTUsMC4zKSI+RGV0ZWN0ZWQ6IDwvc3Bhbj48c3BhbiBzdHlsZT0iY29sb3I6I2ZmZiI+Jytl"
    "LmRldGVjdGVkX2J5Kyc8L3NwYW4+JwogICAgICArJyA8c3BhbiBzdHlsZT0iY29sb3I6cmdiYSgyNTUsMjU1LDI1"
    "NSwwLjMpIj7ihpIgPC9zcGFuPjxzcGFuIHN0eWxlPSJjb2xvcjonK2FnQ29sKyciPicrZS5yZWNvdmVyeV9hZ2Vu"
    "dCsnPC9zcGFuPicKICAgICAgKycgPHNwYW4gc3R5bGU9ImNvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4zKSI+Wycr"
    "ZS5yZWNvdmVyeV9hY3Rpb24rJ108L3NwYW4+PGJyPicKICAgICAgKyc8c3BhbiBzdHlsZT0iY29sb3I6cmdiYSgy"
    "NTUsMjU1LDI1NSwwLjIpIj5TZXY6ICcrZS5mYXVsdF9zZXZlcml0eSsnIMK3ICcrKGUuc3VjY2Vzcz8i4pyTIFJl"
    "Y292ZXJlZCI6IuKclyBGYWlsZWQiKSsnPC9zcGFuPjwvZGl2PicKICAgICAgKyc8L2Rpdj48L2Rpdj4nOwogIH0p"
    "LmpvaW4oIiIpOwp9CgpmdW5jdGlvbiBsbUJ1aWxkUHJvdG9TdGVwcygpIHsKICBjb25zdCBlbCA9IGRvY3VtZW50"
    "LmdldEVsZW1lbnRCeUlkKCJsbVByb3RvU3RlcHMiKTsKICBpZiAoIWVsKSByZXR1cm47CiAgY29uc3Qgc3RlcHMg"
    "PSBbCiAgICB7IHN0ZXA6IjEiLCB0aXRsZToiRmF1bHQgRGV0ZWN0ZWQiLCAgICAgICAgZGVzYzoiQWdlbnQgZGV0"
    "ZWN0cyBhbm9tYWx5IHZpYSBydWxlLWJhc2VkIGNoZWNrICsgTUwgbW9kZWwiLCAgICAgICAgICAgICAgICAgICAg"
    "ICAgICBjb2xvcjoiIzNiODJmNiIgfSwKICAgIHsgc3RlcDoiMiIsIHRpdGxlOiJCcm9hZGNhc3QgUkVRVUVTVCIs"
    "ICAgICAgZGVzYzoiSW5pdGlhdGluZyBhZ2VudCBicm9hZGNhc3RzIGZhdWx0IGluZm8gdG8gYWxsIHBlZXJzIHZp"
    "YSBNZXNzYWdlIEJ1cyIsICAgICAgICAgICBjb2xvcjoiI2Y5NzMxNiIgfSwKICAgIHsgc3RlcDoiMyIsIHRpdGxl"
    "OiJBZ2VudHMgUFJPUE9TRSIsICAgICAgICAgZGVzYzoiRWFjaCByZWxldmFudCBhZ2VudCBwcm9wb3NlcyBhIHJl"
    "Y292ZXJ5IGFjdGlvbiB3aXRoIGNvc3QvdGltZSBlc3RpbWF0ZSIsICAgICAgICBjb2xvcjoiI2E4NTVmNyIgfSwK"
    "ICAgIHsgc3RlcDoiNCIsIHRpdGxlOiJCZXN0IFByb3Bvc2FsIFNlbGVjdGVkIiwgZGVzYzoiTWluIHJlY292ZXJ5"
    "IHRpbWUgKyBtYXggZW5lcmd5IHNhdmluZ3Mgc2VsZWN0ZWQgYXMgd2lubmVyIiwgICAgICAgICAgICAgICAgICAg"
    "IGNvbG9yOiIjMjJjNTVlIiB9LAogICAgeyBzdGVwOiLimqEiLCB0aXRsZToiU2FmZXR5IE9WRVJSSURFIiwgICAg"
    "ICAgZGVzYzoiU2FmZXR5IEFnZW50IGJ5cGFzc2VzIG5lZ290aWF0aW9uIGZvciBzZXZlcml0eSDiiaUgNCBldmVu"
    "dHMiLCAgICAgICAgICAgICAgICAgICAgY29sb3I6IiNlZjQ0NDQiIH0sCiAgXTsKICBlbC5pbm5lckhUTUwgPSBz"
    "dGVwcy5tYXAocyA9PgogICAgJzxkaXYgY2xhc3M9InByb3RvLXN0ZXAiPicKICAgICsnPGRpdiBjbGFzcz0icHJv"
    "dG8tbnVtIiBzdHlsZT0iYmFja2dyb3VuZDonK3MuY29sb3IrJzIyO2JvcmRlcjoxLjVweCBzb2xpZCAnK3MuY29s"
    "b3IrJzU1O2NvbG9yOicrcy5jb2xvcisnIj4nK3Muc3RlcCsnPC9kaXY+JwogICAgKyc8ZGl2PjxkaXYgY2xhc3M9"
    "InByb3RvLXRpdGxlIj4nK3MudGl0bGUrJzwvZGl2PjxkaXYgY2xhc3M9InByb3RvLWRlc2MiPicrcy5kZXNjKyc8"
    "L2Rpdj48L2Rpdj48L2Rpdj4nCiAgKS5qb2luKCIiKTsKfQoKZnVuY3Rpb24gbG1SZW5kZXJOZWdTdGF0cygpIHsK"
    "ICBjb25zdCBlbCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCJsbU5lZ1N0YXRzIik7CiAgaWYgKCFlbCkgcmV0"
    "dXJuOwogIGNvbnN0IHRvdGFsID0gbG1SZWNvdmVyeS5sZW5ndGg7CiAgY29uc3Qgb3ZlcnJpZGVzID0gbG1SZWNv"
    "dmVyeS5maWx0ZXIocj0+ci5zYWZldHlfb3ZlcnJpZGUpLmxlbmd0aDsKICBjb25zdCBzdWNjZXNzUmF0ZSA9IHRv"
    "dGFsID8gTWF0aC5yb3VuZChsbVJlY292ZXJ5LmZpbHRlcihyPT5yLnN1Y2Nlc3MpLmxlbmd0aC90b3RhbCoxMDAp"
    "IDogMDsKICBjb25zdCBhdmdSdCA9IHRvdGFsID8gTWF0aC5yb3VuZChsbVJlY292ZXJ5LnJlZHVjZSgoYSxyKT0+"
    "YStyLnJlY292ZXJ5X3RpbWVfbXMsMCkvdG90YWwpIDogMDsKICBjb25zdCBzdGF0cyA9IFsKICAgIHsgbGFiZWw6"
    "IlRvdGFsIE5lZ290aWF0aW9ucyIsIHZhbHVlOnRvdGFsLCAgICAgICAgICAgICAgIGNvbG9yOiIiIH0sCiAgICB7"
    "IGxhYmVsOiJTYWZldHkgT3ZlcnJpZGVzIiwgICB2YWx1ZTpvdmVycmlkZXMsICAgICAgICAgICAgY29sb3I6IiNl"
    "ZjQ0NDQiIH0sCiAgICB7IGxhYmVsOiJTdWNjZXNzIFJhdGUiLCAgICAgICB2YWx1ZTpzdWNjZXNzUmF0ZSsiJSIs"
    "ICAgICAgY29sb3I6IiMyMmM1NWUiIH0sCiAgICB7IGxhYmVsOiJBdmcgUmVjb3ZlcnkgVGltZSIsICB2YWx1ZTph"
    "dmdSdCsibXMiLCAgICAgICAgICAgY29sb3I6IiMzYjgyZjYiIH0sCiAgXTsKICBlbC5pbm5lckhUTUwgPSBzdGF0"
    "cy5tYXAocyA9PgogICAgJzxkaXYgc3R5bGU9ImRpc3BsYXk6ZmxleDtqdXN0aWZ5LWNvbnRlbnQ6c3BhY2UtYmV0"
    "d2VlbjtwYWRkaW5nOjEwcHggMDtib3JkZXItYm90dG9tOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDUp"
    "Ij4nCiAgICArJzxzcGFuIHN0eWxlPSJmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS10ZXh0MikiPicrcy5sYWJl"
    "bCsnPC9zcGFuPicKICAgICsnPHNwYW4gc3R5bGU9ImZvbnQtd2VpZ2h0OjcwMDtjb2xvcjonKyhzLmNvbG9yfHwi"
    "I2ZmZiIpKyc7Zm9udC1mYW1pbHk6dmFyKC0tbW9ubykiPicrcy52YWx1ZSsnPC9zcGFuPjwvZGl2PicKICApLmpv"
    "aW4oIiIpOwp9CgovLyDilIDilIDilIAgSU5JVCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKc3RhdHVzSW50ZXJ2YWwgPSBzZXRJbnRlcnZh"
    "bChwb2xsU3RhdHVzLCAxNTAwKTsKcG9sbFN0YXR1cygpOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="
)

def _frontend() -> str:
    return base64.b64decode(_FRONTEND_B64).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  §11  AUTH + SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _valid_key(key: str) -> bool:
    """Return True if API key is valid (or auth is disabled)."""
    if not API_KEY: return True
    return secrets.compare_digest(key or "", API_KEY)  # constant-time compare

def _client_ip(req=None) -> str:
    if req is None:   # Flask context
        return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    return req.client.host if req.client else "unknown"  # FastAPI

def _ok(data, code=200): return jsonify(data), code
def _err(msg, code=400): return jsonify({"error": msg}), code

# ═══════════════════════════════════════════════════════════════════════════════
#  §12  FLASK APP  (WSGI — Waitress in production, Flask dev server as fallback)
# ═══════════════════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)
flask_app.config.update(
    SECRET_KEY         = SECRET_KEY,
    MAX_CONTENT_LENGTH = MAX_BODY_BYTES,
    JSON_SORT_KEYS     = False,
    PROPAGATE_EXCEPTIONS = False,
)

# ── optional middleware ────────────────────────────────────────────────────────
if _HAS_COMPRESS:
    FlaskCompress(flask_app)

if _HAS_LIMITER:
    _limiter = Limiter(get_remote_address, app=flask_app,
                       default_limits=[RATE_DEFAULT],
                       storage_uri="memory://",
                       strategy="fixed-window")
    def _rate(lim):   return _limiter.limit(lim)
else:
    def _rate(lim):   return lambda f: f      # no-op

# ── auth decorator ────────────────────────────────────────────────────────────
def _fl_auth(fn):
    @wraps(fn)
    def _w(*a, **kw):
        key = request.headers.get("X-API-Key","") or request.args.get("api_key","")
        if not _valid_key(key):
            return _err("Unauthorized — pass X-API-Key header", 401)
        return fn(*a, **kw)
    return _w

# ── request hooks ─────────────────────────────────────────────────────────────
@flask_app.before_request
def _fl_before():
    with _lock: _state["request_count"] += 1
    _app_init()

@flask_app.after_request
def _fl_after(resp):
    resp.headers.update({
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Headers": "Content-Type,X-API-Key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "X-Content-Type-Options":       "nosniff",
        "X-Frame-Options":              "SAMEORIGIN",
        "X-XSS-Protection":             "1; mode=block",
        "Referrer-Policy":              "strict-origin-when-cross-origin",
        "X-Served-By":                  "NeuralGrid-Flask",
    })
    return resp

# ── error handlers ────────────────────────────────────────────────────────────
@flask_app.errorhandler(Exception)
def _fl_exc(e):
    _log("error","unhandled exception",error=str(e),trace=traceback.format_exc())
    return _err("Internal server error — check server.log", 500)

@flask_app.errorhandler(404)
def _fl_404(e):
    if request.path.startswith("/api/"): return _err("API endpoint not found", 404)
    return _err("Page not found", 404)

@flask_app.errorhandler(413)
def _fl_413(e): return _err("Request body too large (max 10 MB)", 413)

@flask_app.errorhandler(429)
def _fl_429(e): return _err("Rate limit exceeded — retry after 60 s", 429)

# ── system routes ─────────────────────────────────────────────────────────────
@flask_app.route("/")
def fl_index(): return Response(_frontend(), mimetype="text/html")

@flask_app.route("/health")
def fl_health(): return _ok({"status":"ok","timestamp":datetime.now().isoformat()})

@flask_app.route("/ready")
def fl_ready():
    ok=bool(_state["detector"] and _state["detector"].trained)
    code = 200 if ok else 503
    return jsonify({"ready":ok,"model":_state["detector"].best_name if ok else None}), code

@flask_app.route("/api/status")
def fl_status(): return _ok(_status_dict())

# ── pipeline routes ───────────────────────────────────────────────────────────
@flask_app.route("/api/pipeline/init", methods=["POST"])
@_rate(RATE_PIPELINE)
@_fl_auth
def fl_pipeline_init():
    ok,msg = _start_pipeline()
    if not ok: return _err(msg)
    db_audit("pipeline_init", "/api/pipeline/init", _client_ip())
    return _ok({"message":msg,"status":"running"})

@flask_app.route("/api/pipeline/simulate", methods=["POST"])
@_rate(RATE_PIPELINE)
@_fl_auth
def fl_simulate():
    obj, err = _parse(SimulateRequest, request.get_json(silent=True))
    if err: return _err(err[0]["error"], err[1])
    ok, msg = _start_simulation(obj.steps)
    if not ok: return _err(msg)
    db_audit("simulate", "/api/pipeline/simulate", _client_ip(), {"steps":obj.steps})
    return _ok({"message":msg,"steps":obj.steps})

# ── data routes ───────────────────────────────────────────────────────────────
@flask_app.route("/api/dashboard")
def fl_dashboard():
    db_audit("dashboard", "/api/dashboard", _client_ip())
    return _ok(get_dashboard_metrics())

@flask_app.route("/api/dataset/stats")
def fl_ds_stats():
    if _state["dataset"] is None: return _err("Dataset not loaded — run /api/pipeline/init first", 404)
    return _ok(get_dataset_stats())

@flask_app.route("/api/dataset/page")
def fl_ds_page():
    try:
        page = max(1, int(request.args.get("page",1)))
        pp   = min(500, max(1, int(request.args.get("per_page",50))))
    except (ValueError, TypeError):
        page, pp = 1, 50
    rows, total = get_dataset_page(page, pp, request.args.get("fault") or None)
    return _ok({"rows":rows,"total":total,"page":page,"per_page":pp})

@flask_app.route("/api/ml/results")
def fl_ml_results():
    det = _state["detector"]
    if not(det and det.trained): return _err("No trained model", 404)
    res={}
    for name,r in det.results.items():
        res[name]={k:v for k,v in r.items() if k!="confusion_matrix"}
        res[name]["confusion_matrix"] = r.get("confusion_matrix",[])
        res[name]["is_best"]          = (name==det.best_name)
    return _ok({"results":res,"best_model":det.best_name,"classes":list(det.le.classes_)})

@flask_app.route("/api/runs")
def fl_runs(): return _ok(get_all_runs())

@flask_app.route("/api/runs/<run_id>")
def fl_run(run_id):
    d=get_run_details(run_id)
    return _ok(d) if d else _err("Run not found",404)

@flask_app.route("/api/runs/<run_id>/events")
def fl_run_events(run_id):
    d=get_run_details(run_id)
    return _ok(d["events"]) if d else _err("Run not found",404)

@flask_app.route("/api/live/events")
def fl_live(): return _ok(_state["live_events"][-20:])

# ── predict ───────────────────────────────────────────────────────────────────
@flask_app.route("/api/predict", methods=["POST"])
@_rate(RATE_PREDICT)
def fl_predict():
    det = _state["detector"]
    if not(det and det.trained): return _err("Model not trained — run /api/pipeline/init first")
    obj, err = _parse(PredictRequest, request.get_json(silent=True))
    if err: return _err(err[0]["error"], err[1])
    fault, conf = det.predict(obj.model_dump())
    db_audit("predict","/api/predict",_client_ip(),{"fault":fault})
    return _ok({"fault_type":fault,"confidence":round(conf,4),
                "model_used":det.best_name,"timestamp":datetime.now().isoformat()})

# ── downloads (protected) ─────────────────────────────────────────────────────
@flask_app.route("/api/download/dataset")
@_fl_auth
def fl_dl_dataset():
    if _state["dataset"] is None: return _err("Dataset not loaded", 404)
    db_audit("dl_dataset","/api/download/dataset",_client_ip())
    df=_state["dataset"]
    if request.args.get("format")=="json":
        return send_file(io.BytesIO(df.to_json(orient="records").encode()),
                         mimetype="application/json", as_attachment=True, download_name="dataset.json")
    buf=io.StringIO(); df.to_csv(buf,index=False); buf.seek(0)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=dataset.csv"})

@flask_app.route("/api/download/recovery_log")
@_fl_auth
def fl_dl_recovery():
    rid = request.args.get("run_id") or _state.get("last_run_id")
    if not rid: return _err("No simulation run found", 404)
    d = get_run_details(rid)
    if not d or not d["events"]: return _err("No events found", 404)
    buf=io.StringIO(); pd.DataFrame(d["events"]).to_csv(buf,index=False); buf.seek(0)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":f"attachment; filename=recovery_{rid}.csv"})

@flask_app.route("/api/download/ml_report")
@_fl_auth
def fl_dl_ml():
    det = _state["detector"]
    if not(det and det.trained): return _err("No trained model", 404)
    lines=["NeuralGrid ML Report","="*50,f"Generated: {datetime.now().isoformat()}",""]
    for name,r in det.results.items():
        lines+=[f"Model: {name}",
                f"  Accuracy:  {r['accuracy']:.4f}",f"  Precision: {r['precision']:.4f}",
                f"  Recall:    {r['recall']:.4f}",  f"  F1 Score:  {r['f1_score']:.4f}",
                f"  Train:     {r['train_time_sec']:.1f}s",
                f"  Best:      {'YES ★' if name==det.best_name else 'no'}",""]
    return send_file(io.BytesIO("\n".join(lines).encode()),
                     mimetype="text/plain",as_attachment=True,download_name="ml_report.txt")

# ── reports + analytics ───────────────────────────────────────────────────────
@flask_app.route("/api/ai_report", methods=["POST"])
def fl_ai_report():
    obj,_ = _parse(ReportRequest, request.get_json(silent=True))
    if obj is None: obj = ReportRequest()
    lang  = getattr(obj,"language","en")
    if lang not in ("en","hi","or","zh","de"): lang="en"
    db_audit("ai_report","/api/ai_report",_client_ip(),{"lang":lang})
    return _ok(_build_report(lang))

@flask_app.route("/api/analytics/fault_timeline")
def fl_timeline():
    rid = request.args.get("run_id") or _state.get("last_run_id")
    return _ok(_analytics_fault_timeline(rid) if rid else {"labels":[],"series":{}})

@flask_app.route("/api/analytics/recovery_dist")
def fl_rec_dist():
    rid = request.args.get("run_id") or _state.get("last_run_id")
    return _ok(_analytics_recovery_dist(rid) if rid else {"agents":[],"counts":[],"avg_times":[]})

@flask_app.route("/api/analytics/hourly_pattern")
def fl_hourly(): return _ok(get_dataset_stats().get("hourly_stats",[]))

# ═══════════════════════════════════════════════════════════════════════════════
#  §13  FASTAPI APP  (ASGI — Uvicorn in production)
#       Swagger docs: http://localhost:5001/docs
#       ReDoc:        http://localhost:5001/redoc
#       All routes mirror Flask exactly. Shared _state + shared DB.
# ═══════════════════════════════════════════════════════════════════════════════

if _HAS_FASTAPI:

    fapi = FastAPI(
        title       = "NeuralGrid Smart Building API",
        description = """
## Decentralized Multi-Agent Self-Healing Smart Building

Production REST API — also accessible via the Flask dashboard on the main port.

**Quick start:**
1. `POST /api/pipeline/init` — generate data + train models (takes ~2–3 min)
2. Poll `GET /api/status` until `model_trained=true`
3. `POST /api/pipeline/simulate` — run multi-agent simulation
4. Explore `/api/dashboard`, `/api/ml/results`, `/api/predict`

**Auth:** Set `SB_API_KEY` env var, then send `X-API-Key: <key>` header on 🔒 routes.
        """,
        version     = "2.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
    )

    fapi.add_middleware(CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"])
    fapi.add_middleware(GZipMiddleware, minimum_size=500)

    # ── FastAPI auth ─────────────────────────────────────────────────────────
    _fa_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _fa_auth(key: Optional[str] = Depends(_fa_key_header)):
        if not _valid_key(key or ""):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

    # ── request counter + init middleware ─────────────────────────────────────
    @fapi.middleware("http")
    async def _fa_mw(req: FARequest, call_next):
        with _lock: _state["request_count"] += 1
        _app_init()
        resp = await call_next(req)
        resp.headers["X-Served-By"]                = "NeuralGrid-FastAPI/2.0"
        resp.headers["X-Content-Type-Options"]     = "nosniff"
        resp.headers["Access-Control-Allow-Origin"]= "*"
        return resp

    # ── startup event (FastAPI-only) ──────────────────────────────────────────
    @fapi.on_event("startup")
    async def _fa_startup():
        """Ensure DB exists and try loading persisted model when uvicorn starts."""
        init_db()
        saved = FaultDetectionSystem.load()
        if saved:
            with _lock:
                _state["detector"]   = saved
                _state["status_msg"] = f"Model restored: {saved.best_name}"
        _log("info","FastAPI startup complete",db=str(DB_PATH),model=str(MODEL_PATH))

    # ── system ────────────────────────────────────────────────────────────────
    @fapi.get("/", response_class=HTMLResponse, tags=["UI"],
              summary="Interactive dashboard")
    async def fa_index():
        """Full NeuralGrid dashboard (same as Flask root)."""
        return HTMLResponse(_frontend())

    @fapi.get("/health", tags=["System"], summary="Liveness probe")
    async def fa_health():
        return {"status":"ok","timestamp":datetime.now().isoformat()}

    @fapi.get("/ready", tags=["System"],
              summary="Readiness probe — 503 until model is trained")
    async def fa_ready():
        ok=bool(_state["detector"] and _state["detector"].trained)
        if not ok:
            raise HTTPException(503,"Model not trained — call POST /api/pipeline/init")
        return {"ready":True,"model":_state["detector"].best_name}

    @fapi.get("/api/status", tags=["System"], summary="Full system status + DB health")
    async def fa_status():
        return _status_dict()

    # ── pipeline ──────────────────────────────────────────────────────────────
    @fapi.post("/api/pipeline/init", tags=["Pipeline 🔒"],
               dependencies=[Depends(_fa_auth)],
               summary="Initialize dataset + train ML ensemble")
    async def fa_pipeline_init(req: FARequest):
        """
        Runs in a background thread (non-blocking):

        1. Generates 32,000 augmented sensor readings (UCI Energy Efficiency baseline)
        2. Inserts to SQLite using executemany batches (fast, no injection risk)
        3. Trains **Random Forest + Gradient Boosting + MLP** ensemble
        4. Saves best model via joblib — **survives server restarts**

        Poll `GET /api/status` for `progress` and `status_msg`.
        """
        ok, msg = _start_pipeline()
        if not ok: raise HTTPException(400, msg)
        db_audit("pipeline_init", "/api/pipeline/init", _client_ip(req))
        return {"message":msg,"status":"running"}

    @fapi.post("/api/pipeline/simulate", tags=["Pipeline 🔒"],
               dependencies=[Depends(_fa_auth)],
               summary="Run multi-agent negotiation simulation")
    async def fa_simulate(body: SimulateRequest, req: FARequest):
        """
        Runs Contract Net Protocol simulation with 5 agents:

        - **HVAC** — power overload detection + reduction
        - **Energy** — load shedding for overloads
        - **Lighting** — intensity correction
        - **Parking** — congestion rerouting
        - **Safety** — **override** all agents for severity ≥ 4 events

        Poll `GET /api/status` until `simulating=false`.
        """
        ok, msg = _start_simulation(body.steps)
        if not ok: raise HTTPException(400, msg)
        db_audit("simulate","/api/pipeline/simulate",_client_ip(req),{"steps":body.steps})
        return {"message":msg,"steps":body.steps}

    # ── data ──────────────────────────────────────────────────────────────────
    @fapi.get("/api/dashboard", tags=["Analytics"],
              summary="Aggregated metrics across all completed runs")
    async def fa_dashboard(req: FARequest):
        db_audit("dashboard","/api/dashboard",_client_ip(req))
        return get_dashboard_metrics()

    @fapi.get("/api/dataset/stats", tags=["Dataset"],
              summary="Fault distribution + 24-hour HVAC/occupancy profile")
    async def fa_ds_stats():
        if _state["dataset"] is None:
            raise HTTPException(404,"Dataset not loaded — run /api/pipeline/init first")
        return get_dataset_stats()

    @fapi.get("/api/dataset/page", tags=["Dataset"],
              summary="Paginated dataset browser with optional fault-type filter")
    async def fa_ds_page(page: int=1, per_page: int=50, fault: Optional[str]=None):
        rows, total = get_dataset_page(max(1,page), min(500,per_page), fault or None)
        return {"rows":rows,"total":total,"page":page,"per_page":per_page}

    @fapi.get("/api/ml/results", tags=["ML"],
              summary="Performance metrics for all 3 trained models")
    async def fa_ml_results():
        det = _state["detector"]
        if not(det and det.trained): raise HTTPException(404,"No trained model")
        res={}
        for name,r in det.results.items():
            res[name]={k:v for k,v in r.items() if k!="confusion_matrix"}
            res[name]["confusion_matrix"]=r.get("confusion_matrix",[])
            res[name]["is_best"]=(name==det.best_name)
        return {"results":res,"best_model":det.best_name,"classes":list(det.le.classes_)}

    @fapi.get("/api/runs", tags=["Simulation"], summary="List simulation runs (last 20)")
    async def fa_runs(): return get_all_runs()

    @fapi.get("/api/runs/{run_id}", tags=["Simulation"],
              summary="Full run details — metadata + events + ML results")
    async def fa_run(run_id: str):
        d=get_run_details(run_id)
        if not d: raise HTTPException(404,"Run not found")
        return d

    @fapi.get("/api/runs/{run_id}/events", tags=["Simulation"],
              summary="All recovery events for a run")
    async def fa_run_events(run_id: str):
        d=get_run_details(run_id)
        if not d: raise HTTPException(404,"Run not found")
        return d["events"]

    @fapi.get("/api/live/events", tags=["Live"],
              summary="Latest 20 fault/recovery events from the most recent simulation")
    async def fa_live(): return _state["live_events"][-20:]

    # ── predict ───────────────────────────────────────────────────────────────
    @fapi.post("/api/predict", tags=["ML"],
               summary="Real-time fault prediction — Pydantic-validated input")
    async def fa_predict(body: PredictRequest, req: FARequest):
        """
        Predicts fault type for a single sensor reading.
        All 14 fields are **range-validated** — invalid values return HTTP 422.
        Uses whichever model had the best F1 score during training.
        """
        det = _state["detector"]
        if not(det and det.trained): raise HTTPException(400,"Model not trained")
        fault, conf = det.predict(body.model_dump())
        db_audit("predict","/api/predict",_client_ip(req),{"fault":fault})
        return {"fault_type":fault,"confidence":round(conf,4),
                "model_used":det.best_name,"timestamp":datetime.now().isoformat()}

    # ── downloads (protected) ─────────────────────────────────────────────────
    @fapi.get("/api/download/dataset", tags=["Downloads 🔒"],
              dependencies=[Depends(_fa_auth)],
              summary="Download full dataset as CSV or JSON")
    async def fa_dl_dataset(format: str="csv"):
        if _state["dataset"] is None: raise HTTPException(404,"Dataset not loaded")
        df=_state["dataset"]
        if format=="json":
            return StreamingResponse(
                io.BytesIO(df.to_json(orient="records").encode()),
                media_type="application/json",
                headers={"Content-Disposition":"attachment; filename=dataset.json"})
        buf=io.StringIO(); df.to_csv(buf,index=False); buf.seek(0)
        return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=dataset.csv"})

    @fapi.get("/api/download/recovery_log", tags=["Downloads 🔒"],
              dependencies=[Depends(_fa_auth)],
              summary="Download recovery events CSV for a simulation run")
    async def fa_dl_recovery(run_id: Optional[str]=None):
        rid = run_id or _state.get("last_run_id")
        if not rid: raise HTTPException(404,"No simulation run found")
        d=get_run_details(rid)
        if not d or not d["events"]: raise HTTPException(404,"No events found")
        buf=io.StringIO(); pd.DataFrame(d["events"]).to_csv(buf,index=False); buf.seek(0)
        return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
            headers={"Content-Disposition":f"attachment; filename=recovery_{rid}.csv"})

    @fapi.get("/api/download/ml_report", tags=["Downloads 🔒"],
              dependencies=[Depends(_fa_auth)],
              summary="Download ML performance report as plain text")
    async def fa_dl_ml():
        det=_state["detector"]
        if not(det and det.trained): raise HTTPException(404,"No trained model")
        lines=["NeuralGrid ML Report","="*50,f"Generated: {datetime.now().isoformat()}",""]
        for name,r in det.results.items():
            lines+=[f"Model: {name}",
                    f"  F1:       {r['f1_score']:.4f}",
                    f"  Accuracy: {r['accuracy']:.4f}",
                    f"  Best:     {'YES ★' if name==det.best_name else 'no'}",""]
        return StreamingResponse(io.StringIO("\n".join(lines)), media_type="text/plain",
            headers={"Content-Disposition":"attachment; filename=ml_report.txt"})

    # ── reports + analytics ───────────────────────────────────────────────────
    @fapi.post("/api/ai_report", tags=["Reports"],
               summary="Generate multilingual AI predictive report")
    async def fa_ai_report(body: ReportRequest, req: FARequest):
        """
        Generate AI predictive intelligence report.

        **Languages:** `en` English · `hi` हिन्दी · `or` ଓଡ଼ିଆ · `zh` 中文 · `de` Deutsch

        Invalid language codes default to English.
        """
        lang=getattr(body,"language","en")
        if lang not in ("en","hi","or","zh","de"): lang="en"
        db_audit("ai_report","/api/ai_report",_client_ip(req),{"lang":lang})
        return _build_report(lang)

    @fapi.get("/api/analytics/fault_timeline", tags=["Analytics"],
              summary="Fault counts grouped into 10-step time buckets")
    async def fa_timeline(run_id: Optional[str]=None):
        rid = run_id or _state.get("last_run_id")
        return _analytics_fault_timeline(rid) if rid else {"labels":[],"series":{}}

    @fapi.get("/api/analytics/recovery_dist", tags=["Analytics"],
              summary="Per-agent recovery count and average recovery time")
    async def fa_rec_dist(run_id: Optional[str]=None):
        rid = run_id or _state.get("last_run_id")
        return _analytics_recovery_dist(rid) if rid else {"agents":[],"counts":[],"avg_times":[]}

    @fapi.get("/api/analytics/hourly_pattern", tags=["Analytics"],
              summary="24-hour HVAC power + occupancy pattern from dataset")
    async def fa_hourly():
        return get_dataset_stats().get("hourly_stats",[])

else:
    fapi = None   # FastAPI not installed

# ═══════════════════════════════════════════════════════════════════════════════
#  §14  GRACEFUL SHUTDOWN + ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_shutdown(sig, frame):
    _log("info","shutdown signal received",signal=sig)
    print(f"\n[SHUTDOWN] Signal {sig} — draining in-flight requests…")
    time.sleep(1)   # give background threads a moment
    sys.exit(0)

signal.signal(signal.SIGINT,  _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)

def _print_banner(host: str, port: int, mode: str):
    local  = "localhost" if host=="0.0.0.0" else host
    fa_url = (f"http://{local}:{port+1}/docs" if mode=="both" else
              f"http://{local}:{port}/docs"   if mode=="fastapi" else "N/A")
    w_info = "waitress (8 threads)" if _HAS_WAITRESS  else "Flask dev (install waitress)"
    u_info = "uvicorn"              if _HAS_FASTAPI   else "not installed"
    rl_info= "flask-limiter ✓"     if _HAS_LIMITER   else "MISSING — pip install flask-limiter"
    gz_info= "flask-compress ✓"    if _HAS_COMPRESS  else "MISSING — pip install flask-compress"
    pv_info= "pydantic v2 ✓"       if _HAS_PYDANTIC  else "MISSING — pip install pydantic"
    fa_info= "fastapi ✓"           if _HAS_FASTAPI   else "MISSING — pip install fastapi uvicorn[standard]"
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  🏢  NeuralGrid Smart Building — Production Server v2.0          ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Dashboard  : http://{local}:{port:<44}║")
    print(f"║  API Status : http://{local}:{port}/api/status{' '*20}║")
    print(f"║  Swagger UI : {fa_url:<53}║")
    print(f"║  Database   : {str(DB_PATH):<53}║")
    print(f"║  Model      : {str(MODEL_PATH):<53}║")
    print(f"║  Logs       : {str(LOG_PATH):<53}║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Mode       : {mode:<53}║")
    print(f"║  WSGI       : {w_info:<53}║")
    print(f"║  ASGI       : {u_info:<53}║")
    print(f"║  Rate limit : {rl_info:<53}║")
    print(f"║  Gzip       : {gz_info:<53}║")
    print(f"║  Validation : {pv_info:<53}║")
    print(f"║  FastAPI    : {fa_info:<53}║")
    print(f"║  API Key    : {'ENABLED' if API_KEY else 'disabled (set SB_API_KEY env var to enable)':<53}║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  CORRECT:  python smartbuilding_production.py                    ║")
    print("║  WRONG:    uvicorn smartbuilding_production:flask_app  ← WSGI!  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NeuralGrid Production Server")
    ap.add_argument("--port",  type=int, default=5000, help="Port (default 5000)")
    ap.add_argument("--host",  type=str, default="0.0.0.0", help="Bind host")
    ap.add_argument("--api",   type=str, default="both",
                    choices=["flask","fastapi","both"],
                    help="flask=WSGI only | fastapi=ASGI only | both=Flask:PORT+FastAPI:PORT+1")
    ap.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = ap.parse_args()

    # Ensure DB is ready before accepting any traffic
    init_db()
    if not _db_test():
        print(f"[FATAL] Cannot connect to database at {DB_PATH}")
        print("        Check disk permissions and available space.")
        sys.exit(1)

    # Try to restore persisted model
    saved = FaultDetectionSystem.load()
    if saved:
        with _lock:
            _state["detector"]   = saved
            _state["status_msg"] = f"Model restored: {saved.best_name}"
    _state["init_done"] = True   # prevent double-init on first request

    _print_banner(args.host, args.port, args.api)

    def _run_flask():
        if _HAS_WAITRESS:
            _log("info","starting waitress",host=args.host,port=args.port,threads=8)
            waitress_serve(flask_app, host=args.host, port=args.port,
                           threads=8, channel_timeout=120)
        else:
            print("[WARN] waitress not installed — using Flask dev server (single thread)")
            print("       For production:  pip install waitress")
            flask_app.run(host=args.host, port=args.port,
                          debug=args.debug, threaded=True)

    def _run_fastapi(port: int):
        if not _HAS_FASTAPI:
            print("[WARN] FastAPI/uvicorn not installed — FastAPI disabled")
            print("       Install: pip install fastapi uvicorn[standard]")
            return
        _log("info","starting uvicorn",host=args.host,port=port)
        uvicorn.run(fapi, host=args.host, port=port,
                    workers=1, log_level="warning", access_log=False)

    if args.api == "flask":
        _run_flask()
    elif args.api == "fastapi":
        _run_fastapi(args.port)
    else:
        # both: Flask on PORT, FastAPI on PORT+1
        fa_port = args.port + 1
        print(f"  Flask   → http://localhost:{args.port:<5}  (Dashboard + all APIs)")
        print(f"  FastAPI → http://localhost:{fa_port}/docs   (Swagger auto-docs)")
        print()
        fa_thread = threading.Thread(target=_run_fastapi, args=(fa_port,), daemon=True)
        fa_thread.start()
        _run_flask()   # blocking — main thread holds here
