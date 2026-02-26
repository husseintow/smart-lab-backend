import os
import sqlite3
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from whitenoise import WhiteNoise

# =====================================================
# Configuration
# =====================================================
# Absolute path to the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel Read-Only File System Fix
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/security.db"
    # Absolute path to the pre-seeded DB in the repository
    ORIGINAL_DB = os.path.join(BASE_DIR, "security.db")
    if not os.path.exists(DB_PATH) and os.path.exists(ORIGINAL_DB):
        try:
            shutil.copy2(ORIGINAL_DB, DB_PATH)
        except Exception as e:
            print(f"Warn: Could not seed DB: {e}")
else:
    DB_PATH = os.path.join(BASE_DIR, "security.db")

API_KEY = os.getenv("API_KEY", "lab_2026_secure_key")
DEFAULT_DEVICE_ID = os.getenv("DEFAULT_DEVICE_ID", "esp32_lab_1")

# Optional Telegram - Using provided defaults if Env Var is missing
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Security policy window
OFF_HOURS_START = int(os.getenv("OFF_HOURS_START", "18"))
OFF_HOURS_END = int(os.getenv("OFF_HOURS_END", "7"))

# Prevent alarm command spam
ALARM_COOLDOWN_SECONDS = int(os.getenv("ALARM_COOLDOWN_SECONDS", "30"))

import threading
model_lock = threading.Lock()

app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='static', 
            template_folder='templates')

# Wrap the WSGI app with WhiteNoise
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/', prefix='static/')

# Standalone CORS configuration
CORS(app)
model = None
model_trained_at = None


# =====================================================
# Generic Helpers
# =====================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def is_authorized(req) -> bool:
    if not API_KEY:
        return True
    return req.headers.get("X-API-Key", "") == API_KEY


# =====================================================
# Database Init + Migration
# =====================================================
def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                hour INTEGER NOT NULL,
                motion INTEGER NOT NULL,
                door INTEGER NOT NULL,
                sensor_type TEXT NOT NULL DEFAULT 'none',
                sensor_code INTEGER NOT NULL DEFAULT 0,
                system_armed INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                label TEXT,
                source TEXT DEFAULT 'model',
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                command TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consumed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                armed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        row = conn.execute("SELECT id FROM system_state WHERE id=1").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO system_state (id, armed, updated_at) VALUES (1, 0, ?)",
                (now_iso(),),
            )
        conn.commit()


def _table_columns(conn, table_name: str) -> Set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r["name"] for r in rows}


def _ensure_column(conn, table: str, column: str, ddl: str):
    cols = _table_columns(conn, table)
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def migrate_db():
    with get_conn() as conn:
        # events
        _ensure_column(conn, "events", "sensor_type", "TEXT NOT NULL DEFAULT 'none'")
        _ensure_column(conn, "events", "sensor_code", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "events", "label", "TEXT")
        _ensure_column(conn, "events", "source", "TEXT DEFAULT 'model'")
        _ensure_column(conn, "events", "notes", "TEXT")

        # commands
        _ensure_column(conn, "commands", "consumed_at", "TEXT")

        # Backfill sensor_type if old rows
        conn.execute(
            """
            UPDATE events
            SET sensor_type = CASE
                WHEN motion=1 AND door=1 THEN 'both'
                WHEN motion=1 THEN 'motion'
                WHEN door=1 THEN 'door'
                ELSE 'none'
            END
            WHERE sensor_type IS NULL OR TRIM(sensor_type)=''
            """
        )

        # Backfill sensor_code
        conn.execute(
            """
            UPDATE events
            SET sensor_code = CASE
                WHEN sensor_type='motion' THEN 1
                WHEN sensor_type='door' THEN 2
                WHEN sensor_type='both' THEN 3
                ELSE 0
            END
            WHERE sensor_code IS NULL
            """
        )

        # Performance indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_prediction ON events(prediction)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_commands_pending ON commands(device_id, consumed_at, id)")
        conn.commit()


# =====================================================
# State + Command Queue
# =====================================================
@app.teardown_appcontext
def close_connection(exception):
    """Ensure DB connections are closed after every request."""
    pass # Managed by 'with get_conn()' but good for future extensibility

@app.errorhandler(500)
def handle_internal_error(error):
    return jsonify({"ok": False, "error": "Internal Server Error", "details": str(error)}), 500

def get_armed() -> int:
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT armed FROM system_state WHERE id=1").fetchone()
            return int(row["armed"]) if row else 0
    except Exception:
        return 0

def set_armed(armed: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE system_state SET armed=?, updated_at=? WHERE id=1",
            (int(armed), now_iso()),
        )
        conn.commit()


def queue_command(device_id: str, command: str):
    with get_conn() as conn:
        # Avoid duplicate same pending command
        row = conn.execute(
            """
            SELECT id FROM commands
            WHERE device_id=? AND command=? AND consumed_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (device_id, command),
        ).fetchone()

        if row:
            return

        conn.execute(
            "INSERT INTO commands (device_id, command, created_at, consumed_at) VALUES (?, ?, ?, NULL)",
            (device_id, command, now_iso()),
        )
        conn.commit()


def pop_next_command(device_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, command
            FROM commands
            WHERE device_id=? AND consumed_at IS NULL
            ORDER BY id ASC
            LIMIT 1
            """,
            (device_id,),
        ).fetchone()

        if not row:
            return "NONE"

        conn.execute(
            "UPDATE commands SET consumed_at=? WHERE id=?",
            (now_iso(), int(row["id"])),
        )
        conn.commit()
        return str(row["command"])


# =====================================================
# Sensor Mapping + Rules
# =====================================================
def sensor_type_to_code(sensor_type: str, motion: int, door: int) -> int:
    st = (sensor_type or "").strip().lower()
    mapping = {"none": 0, "motion": 1, "door": 2, "both": 3}
    if st in mapping:
        return mapping[st]

    # fallback from bits
    if motion == 1 and door == 1:
        return 3
    if motion == 1:
        return 1
    if door == 1:
        return 2
    return 0


def code_to_sensor_type(code: int) -> str:
    return {0: "none", 1: "motion", 2: "door", 3: "both"}.get(int(code), "none")


def is_off_hours(hour: int) -> bool:
    return hour >= OFF_HOURS_START or hour < OFF_HOURS_END


def should_alarm_by_rule(hour: int, armed: int, motion: int, door: int) -> bool:
    if int(armed) == 0:
        return False
    
    has_activity = (int(motion) == 1 or int(door) == 1)
    if not has_activity:
        return False

    # Off-hours logic: If START > END (e.g., 18 to 7), it spans midnight.
    h = int(hour)
    if OFF_HOURS_START > OFF_HOURS_END:
        is_off = (h >= OFF_HOURS_START or h < OFF_HOURS_END)
    else:
        is_off = (OFF_HOURS_START <= h < OFF_HOURS_END)
        
    return is_off


def can_trigger_alarm(device_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT created_at
            FROM commands
            WHERE device_id=? AND command='ALARM_ON_30'
            ORDER BY id DESC
            LIMIT 1
            """,
            (device_id,),
        ).fetchone()

    if not row:
        return True

    try:
        last_at = datetime.fromisoformat(str(row["created_at"]))
        return (datetime.now() - last_at) >= timedelta(seconds=ALARM_COOLDOWN_SECONDS)
    except Exception:
        return True


# =====================================================
# Notifications
# =====================================================
def log_notification(event_id: int, channel: str, status: str, message: str):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO notifications (event_id, channel, status, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, channel, status, message, now_iso()),
        )
        conn.commit()


def send_telegram(message: str, event_id: int):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_notification(event_id, "telegram", "skipped", "Telegram not configured")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=8)
        if r.ok:
            log_notification(event_id, "telegram", "sent", "OK")
        else:
            log_notification(event_id, "telegram", "failed", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as ex:
        log_notification(event_id, "telegram", "failed", str(ex))


# =====================================================
# Machine Learning (Decision Tree)
# =====================================================
def bootstrap_training_data():
    """
    Features:
      [hour, motion, door, system_armed, sensor_code]
    Label:
      0 = normal
      1 = intrusion
    """
    X, y = [], []
    for hour in range(24):
        for motion in (0, 1):
            for door in (0, 1):
                for armed in (0, 1):
                    sensor_code = sensor_type_to_code("", motion, door)
                    intr = 1 if should_alarm_by_rule(hour, armed, motion, door) else 0
                    X.append([hour, motion, door, armed, sensor_code])
                    y.append(intr)
    return np.array(X, dtype=int), np.array(y, dtype=int)


def train_model():
    """Starts training in a background thread."""
    thread = threading.Thread(target=_train_model_worker, daemon=True)
    thread.start()


def _train_model_worker():
    global model, model_trained_at
    if not model_lock.acquire(blocking=False):
        return  # Training already in progress

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT hour, motion, door, system_armed, sensor_code, label
                FROM events
                WHERE label IN ('normal', 'intrusion')
                """
            ).fetchall()

        X_user: List[List[int]] = []
        y_user: List[int] = []

        for r in rows:
            X_user.append([
                int(r["hour"]),
                int(r["motion"]),
                int(r["door"]),
                int(r["system_armed"]),
                int(r["sensor_code"]),
            ])
            y_user.append(1 if r["label"] == "intrusion" else 0)

        X_boot, y_boot = bootstrap_training_data()

        # Blend with bootstrap if user labels are still small
        if len(X_user) < 30:
            if X_user:
                X = np.vstack([np.array(X_user, dtype=int), X_boot])
                y = np.concatenate([np.array(y_user, dtype=int), y_boot])
            else:
                X, y = X_boot, y_boot
        else:
            X = np.array(X_user, dtype=int)
            y = np.array(y_user, dtype=int)

        clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
        clf.fit(X, y)

        model = clf
        model_trained_at = now_iso()
    finally:
        model_lock.release()


def classify_event(hour: int, motion: int, door: int, armed: int, sensor_code: int) -> str:
    global model
    if model is None:
        train_model()

    pred = int(model.predict(np.array([[hour, motion, door, armed, sensor_code]], dtype=int))[0])
    label = "intrusion" if pred == 1 else "normal"

    # deterministic safety override
    if should_alarm_by_rule(hour, armed, motion, door):
        label = "intrusion"

    return label


# =====================================================
# Page Routes
# =====================================================
@app.route("/")
def page_welcome():
    """Root route renders the welcome interface."""
    return render_template("index.html")


@app.get("/dashboard")
def page_dashboard():
    return render_template("dashboard.html")


@app.get("/events")
def page_events():
    return render_template("events.html")


@app.get("/settings")
def page_settings():
    return render_template("settings.html")


# =====================================================
# Dashboard APIs
# =====================================================
@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "time": now_iso()})


@app.get("/api/state")
def api_state():
    return jsonify({
        "armed": get_armed(),
        "server_time": now_iso(),
        "model_trained_at": model_trained_at,
        "off_hours_start": OFF_HOURS_START,
        "off_hours_end": OFF_HOURS_END,
    })


@app.get("/api/summary")
def api_summary():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        intr = conn.execute("SELECT COUNT(*) AS c FROM events WHERE prediction='intrusion'").fetchone()["c"]
        norm = conn.execute("SELECT COUNT(*) AS c FROM events WHERE prediction='normal'").fetchone()["c"]
        pending = conn.execute("SELECT COUNT(*) AS c FROM commands WHERE consumed_at IS NULL").fetchone()["c"]
        last = conn.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()

    return jsonify({
        "total_events": int(total),
        "intrusions": int(intr),
        "normal_events": int(norm),
        "pending_commands": int(pending),
        "last_event_time": last["ts"] if last else None,
    })


@app.get("/api/events")
def api_events():
    try:
        limit = int(request.args.get("limit", 100))
    except Exception:
        limit = 100

    limit = max(1, min(500, limit))

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, device_id, ts, hour, motion, door, sensor_type, system_armed, prediction, label, source, notes
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return jsonify([row_to_dict(r) for r in rows])


@app.post("/api/arm")
def api_arm():
    data = request.get_json(silent=True) or {}
    device_id = str(data.get("device_id", DEFAULT_DEVICE_ID))

    set_armed(1)
    queue_command(device_id, "ARM")

    return jsonify({"ok": True, "armed": 1, "device_id": device_id})


@app.post("/api/disarm")
def api_disarm():
    data = request.get_json(silent=True) or {}
    device_id = str(data.get("device_id", DEFAULT_DEVICE_ID))

    set_armed(0)
    queue_command(device_id, "DISARM")
    queue_command(device_id, "ALARM_OFF")

    return jsonify({"ok": True, "armed": 0, "device_id": device_id})


@app.post("/api/alarm_test")
def api_alarm_test():
    data = request.get_json(silent=True) or {}
    device_id = str(data.get("device_id", DEFAULT_DEVICE_ID))
    queue_command(device_id, "ALARM_ON_30")
    return jsonify({"ok": True, "queued": "ALARM_ON_30", "device_id": device_id})


@app.post("/api/retrain")
def api_retrain():
    train_model()
    return jsonify({"ok": True, "trained_at": model_trained_at})


@app.post("/api/label/<int:event_id>")
def api_label(event_id: int):
    data = request.get_json(silent=True) or {}
    label = str(data.get("label", "")).strip().lower()
    notes = str(data.get("notes", "")).strip()

    if label not in ("normal", "intrusion"):
        return jsonify({"ok": False, "error": "label must be 'normal' or 'intrusion'"}), 400

    with get_conn() as conn:
        conn.execute(
            "UPDATE events SET label=?, notes=?, source='manual' WHERE id=?",
            (label, notes, event_id),
        )
        conn.commit()

    train_model()
    return jsonify({"ok": True, "event_id": event_id, "label": label, "trained_at": model_trained_at})


@app.post("/api/test_notification")
def api_test_notification():
    send_telegram(f" Test notification from Smart Lab Security at {now_iso()}", event_id=0)
    return jsonify({"ok": True, "message": "Telegram test attempted"})


# =====================================================
# Device APIs (ESP32)
# =====================================================
@app.post("/event")
def receive_event():
    try:
        if not is_authorized(request):
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        device_id = str(data.get("device_id", DEFAULT_DEVICE_ID))
        motion = int(data.get("motion", 0))
        door = int(data.get("door", 0))
        sensor_type_raw = str(data.get("sensor_type", "")).strip().lower()

        # Server is source of truth for arm state
        armed = int(get_armed())
        sensor_code = sensor_type_to_code(sensor_type_raw, motion, door)
        sensor_type = code_to_sensor_type(sensor_code)

        # optional client time
        ts_in = data.get("time")
        if ts_in:
            try:
                dt = datetime.fromisoformat(str(ts_in).replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now()
        else:
            dt = datetime.now()

        event_ts = dt.isoformat(timespec="seconds")
        hour = int(dt.hour)

        prediction = classify_event(hour, motion, door, armed, sensor_code)
        
        # Trace the reason
        reason = "ML Model"
        if should_alarm_by_rule(hour, armed, motion, door):
            reason = "Safety Rule Override"
        
        with get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO events
                (device_id, ts, hour, motion, door, sensor_type, sensor_code, system_armed, prediction, label, source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'model', ?)
                """,
                (device_id, event_ts, hour, motion, door, sensor_type, sensor_code, armed, prediction, prediction, f"Reason: {reason}"),
            )
            event_id = int(cur.lastrowid)
            conn.commit()

        if prediction == "intrusion":
            if can_trigger_alarm(device_id):
                queue_command(device_id, "ALARM_ON_30")

            send_telegram(
                " Intrusion Detected\n"
                f"Device: {device_id}\n"
                f"Time: {event_ts}\n"
                f"Motion: {motion}, Door: {door}, Armed: {armed}",
                event_id=event_id,
            )

        return jsonify({"ok": True, "event_id": event_id, "prediction": prediction})
    except Exception as e:
        app.logger.error(f"Event processing failed: {str(e)}")
        return jsonify({"ok": False, "error": "Processing failure"}), 500


@app.get("/command")
def get_command():
    if not is_authorized(request):
        return jsonify({"command": "NONE"}), 401

    device_id = request.args.get("device_id", DEFAULT_DEVICE_ID)
    cmd = pop_next_command(device_id)
    return jsonify({"command": cmd})

@app.post("/api/simulate")
def api_simulate():
    """
    Utility endpoint to simulate events for presentation/testing.
    Allows specifying hour to test day vs night.
    """
    data = request.get_json(silent=True) or {}
    motion = int(data.get("motion", 1))
    door = int(data.get("door", 0))
    hour_val = data.get("hour") # "02", "12", etc.
    
    device_id = "virtual_simulator"
    armed = get_armed()
    
    # Generate timestamp
    if hour_val:
        dt = datetime.now().replace(hour=int(hour_val), minute=0, second=0)
    else:
        dt = datetime.now()
    
    event_ts = dt.isoformat(timespec="seconds")
    hour = int(dt.hour)
    sensor_code = sensor_type_to_code("motion" if motion else "door", motion, door)
    
    prediction = classify_event(hour, motion, door, armed, sensor_code)
    
    reason = "ML Model"
    if should_alarm_by_rule(hour, armed, motion, door):
        reason = "Safety Rule Override"

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO events
            (device_id, ts, hour, motion, door, sensor_type, sensor_code, system_armed, prediction, label, source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'model', ?)
            """,
            (device_id, event_ts, hour, motion, door, code_to_sensor_type(sensor_code), sensor_code, armed, prediction, prediction, f"Simulator: {reason}"),
        )
        event_id = int(cur.lastrowid)
        conn.commit()

    if prediction == "intrusion":
        queue_command(device_id, "ALARM_ON_30")

    return jsonify({
        "ok": True, 
        "event_id": event_id, 
        "prediction": prediction, 
        "reason": reason,
        "time_simulated": event_ts
    })


# =====================================================
# Initialization
# =====================================================
# Run initialization on startup (even when imported by Vercel)
try:
    init_db()
    migrate_db()
    # Deep learning/ML training can be slow, but we need it for predictions
    # if it's the first time and model is None.
    if model is None:
        _train_model_worker() # Run synchronously once on start
except Exception as e:
    print(f"Startup initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
