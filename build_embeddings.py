#!/usr/bin/env python3
"""
build_embeddings.py

Builds/updates semantic embeddings for each question in survey_questions.sqlite.

Run:
    python build_embeddings.py --db survey_questions.sqlite
"""

import argparse
import json
import sqlite3
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env sitting next to this script
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file with OPENAI_API_KEY=... or set it in your environment."
    )

client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"  # good default for semantic search

# -----------------------------
# CONFIG / PLACEHOLDERS
# -----------------------------

QUESTION_TYPE_MAP = {
    "radio": "Single Select",
    "checkbox": "Multi-Select",
    "select": "Dropdown",
}


def friendly_question_type(raw: str | None) -> str:
    """
    Map raw XML/DB question_type to a human-friendly label.

    - radio    -> Single Select
    - checkbox -> Multi-Select
    - select   -> Dropdown
    - anything else: returned unchanged
    """
    if not raw:
        return ""
    key = raw.strip().lower()
    return QUESTION_TYPE_MAP.get(key, raw)


def get_embedding(text: str) -> List[float]:
    """
    Return an embedding vector for the given text using OpenAI's embeddings API.

    Requires:
      - `pip install openai`
      - OPENAI_API_KEY environment variable set.

    Uses the same model name here and in the QA app so retrieval is consistent.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    # We assume a single input string, so we take index 0
    return response.data[0].embedding


# -----------------------------
# DB helpers
# -----------------------------

CREATE_EMBED_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS question_embeddings (
    question_label TEXT PRIMARY KEY,
    embedding TEXT NOT NULL
);
"""

def init_embedding_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(CREATE_EMBED_TABLE_SQL)
    conn.commit()


def load_questions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Load all questions from survey_questions table.
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            question_label,
            question_text,
            question_type,
            response_options,
            frequency,
            months_asked,
            first_seen,
            last_seen,
            currently_active,
            notes
        FROM survey_questions
    """)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def build_description(row: Dict[str, Any]) -> str:
    """
    Build a textual description of a question for embedding.
    """
    label = row["question_label"] or ""
    text = row["question_text"] or ""
    qtype_raw = row["question_type"] or ""
    qtype = friendly_question_type(qtype_raw)
    freq = row["frequency"] or ""
    first_seen = row["first_seen"] or ""
    last_seen = row["last_seen"] or ""
    active = "yes" if row["currently_active"] else "no"

    # response_options is JSON-encoded list
    try:
        options = json.loads(row["response_options"] or "[]")
    except Exception:
        options = []
    options_str = "; ".join(str(o) for o in options)

    months_asked = row.get("months_asked") or ""

    # notes is JSON-encoded list of {wave, description}
    notes_raw = row.get("notes") or "[]"
    try:
        notes = json.loads(notes_raw)
    except Exception:
        notes = []
    notes_str = "; ".join(
        f"{n.get('wave', 'unknown')}: {n.get('description', '')}" for n in notes
    )

    return (
        f"Label: {label}\n"
        f"Type: {qtype}\n"
        f"Question: {text}\n"
        f"Options: {options_str}\n"
        f"Frequency: {freq}\n"
        f"Months asked: {months_asked}\n"
        f"First seen: {first_seen}\n"
        f"Last seen: {last_seen}\n"
        f"Currently active: {active}\n"
        f"Change history: {notes_str}"
    )


def upsert_embedding(conn: sqlite3.Connection, label: str, embedding: List[float]) -> None:
    """
    Insert or replace embedding for a question_label.
    """
    emb_json = json.dumps(embedding)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO question_embeddings (question_label, embedding)
        VALUES (?, ?)
        ON CONFLICT(question_label) DO UPDATE SET embedding=excluded.embedding
        """,
        (label, emb_json),
    )


# -----------------------------
# Main
# -----------------------------

def build_all_embeddings(db_path: str) -> None:
    if not os.path.isfile(db_path):
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        init_embedding_table(conn)
        questions = load_questions(conn)

        print(f"Loaded {len(questions)} questions from survey_questions.")

        for i, row in enumerate(questions, start=1):
            label = row["question_label"]
            desc = build_description(row)
            emb = get_embedding(desc)
            upsert_embedding(conn, label, emb)
            if i % 20 == 0:
                print(f"Processed {i} questions...")

        conn.commit()
        print("Embedding build complete.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic embeddings for survey questions."
    )
    parser.add_argument(
        "--db",
        default="survey_questions.sqlite",
        help="Path to the SQLite database.",
    )
    args = parser.parse_args()
    build_all_embeddings(args.db)


if __name__ == "__main__":
    main()