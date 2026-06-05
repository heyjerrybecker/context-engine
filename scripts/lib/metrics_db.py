import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_metrics (
    session_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    session_type TEXT NOT NULL,
    agent TEXT NOT NULL,
    total_messages INTEGER,
    total_tokens INTEGER,
    context_messages INTEGER,
    context_tokens INTEGER,
    warmup_message_count INTEGER,
    correction_count INTEGER,
    first_new_work_message_index INTEGER,
    cor REAL,
    cr REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS message_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    label TEXT NOT NULL,
    token_count INTEGER,
    is_correction BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (session_id) REFERENCES session_metrics(session_id)
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_session(conn: sqlite3.Connection, row: dict):
    conn.execute(
        """INSERT OR REPLACE INTO session_metrics
        (session_id, date, session_type, agent, total_messages, total_tokens,
         context_messages, context_tokens, warmup_message_count, correction_count,
         first_new_work_message_index, cor, cr, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row["session_id"], row["date"], row["session_type"], row["agent"],
            row["total_messages"], row["total_tokens"], row["context_messages"],
            row["context_tokens"], row["warmup_message_count"], row["correction_count"],
            row["first_new_work_message_index"], row["cor"], row["cr"], row["created_at"],
        ),
    )
    conn.commit()


def insert_labels(conn: sqlite3.Connection, session_id: str, labels: list):
    conn.executemany(
        """INSERT INTO message_labels (session_id, message_index, role, label, token_count, is_correction)
        VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (session_id, l["index"], l["role"], l["label"], l["token_count"], l["is_correction"])
            for l in labels
        ],
    )
    conn.commit()
