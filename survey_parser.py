#!/usr/bin/env python3
"""
Survey XML parser and questionnaire history tracker.

- Parses Decipher-style survey XML exports (radio, checkbox, select, number).
- Extracts question label, text, response options, and type.
- Tracks which months each question is asked.
- Maintains a SQLite database with change tracking.

Usage:
    python survey_parser.py path/to/July_2025_XML.txt --db survey_questions.sqlite
"""

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from build_embeddings import build_all_embeddings


# Load .env sitting next to this script
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file with OPENAI_API_KEY=... or set it in your environment."
    )

llm_client = OpenAI(api_key=api_key)
LLM_MODEL = "gpt-4.1-mini"  # or another GPT-4.* / GPT-4o-mini model you’re using


# -----------------------------
# Helpers: text and date parsing
# -----------------------------

MONTH_NAME_MAP = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12',
    # short forms
    'jan': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'sept': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12',
}



def parse_month_from_filename(filename: str) -> str:
    """
    Parse a month identifier (YYYY-MM) from the filename.

    Supports patterns like:
      - SAAT_2025-07.xml
      - SAAT_202507.xml
      - July 2025 XML.txt
      - 2025_07_SAAT.xml

    Returns:
        A string like '2025-07'.

    Raises:
        ValueError if a suitable month/year cannot be found.
    """
    name = os.path.basename(filename)
    lower = name.lower()

    # First: look for patterns like 2025-07, 2025_07, 2025 07
    m = re.search(r'(20\d{2})[-_ ]?(0[1-9]|1[0-2])', lower)
    if m:
        year, month = m.group(1), m.group(2)
        return f"{year}-{month}"

    # Second: look for month name + year, e.g., "july 2025"
    year_match = re.search(r'(20\d{2})', lower)
    if year_match:
        year = year_match.group(1)
        # search for any month name token
        for token in re.split(r'[^a-z]+', lower):
            if token in MONTH_NAME_MAP:
                month = MONTH_NAME_MAP[token]
                return f"{year}-{month}"

    raise ValueError(f"Could not parse year/month from filename: {filename}")


TAG_RE = re.compile(r'<.*?>', re.DOTALL)


def clean_text(text: str) -> str:
    """
    Decode HTML entities and strip basic HTML tags.
    """
    if text is None:
        return ""
    # Decode HTML entities (&lt;strong&gt; etc.)
    unescaped = html.unescape(text)
    # Remove simple tags like <strong>...</strong>
    without_tags = TAG_RE.sub('', unescaped)
    return without_tags.strip()


def compute_question_hash(label: str, q_type: str, text: str, options: List[str]) -> str:
    """
    Compute a stable hash for a question based on label, type, text, and options.
    """
    canonical = {
        "label": label,
        "type": q_type,
        "text": text,
        "options": options,
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------
# Database helpers
# -----------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS survey_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_label TEXT NOT NULL,
    question_text TEXT,
    question_type TEXT,
    response_options TEXT,           -- JSON-encoded list of strings
    frequency TEXT DEFAULT 'Always', -- 'Always', 'Quarterly', 'Bi-Annually', etc.
    months_asked TEXT,               -- JSON-encoded list of 'YYYY-MM'
    first_seen TEXT,                 -- 'YYYY-MM'
    last_seen TEXT,                  -- 'YYYY-MM' (last month the question was asked)
    currently_active INTEGER DEFAULT 1,  -- 1=true, 0=false
    change_flag INTEGER DEFAULT 0,       -- 1 if question disappeared but frequency='Always'
    question_hash TEXT,
    notes TEXT                           -- JSON-encoded list of {wave, description}
);
"""


CREATE_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_survey_questions_label
ON survey_questions (question_label);
"""

CREATE_WAVES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processed_waves (
    wave TEXT PRIMARY KEY,         -- 'YYYY-MM'
    source_file TEXT,              -- original file name
    processed_at TEXT              -- ISO datetime
);
"""

def is_wave_processed(conn: sqlite3.Connection, wave: str) -> bool:
    """
    Return True if this wave (YYYY-MM) has already been processed
    and recorded in processed_waves.
    """
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM processed_waves WHERE wave = ?", (wave,))
    return cur.fetchone() is not None


def mark_wave_processed(conn: sqlite3.Connection, wave: str, source_file: str) -> None:
    """
    Record that a wave (YYYY-MM) has been processed from a given file.
    If the row already exists, we update the source_file and processed_at.
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO processed_waves (wave, source_file, processed_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(wave) DO UPDATE SET
            source_file = excluded.source_file,
            processed_at = excluded.processed_at
        """,
        (wave, os.path.basename(source_file)),
    )
    conn.commit()


def append_note(notes: List[Dict[str, Any]], wave: str, description: str) -> None:
    """
    Append a note if an identical (wave + description) note doesn't already exist.
    """
    for n in notes:
        if n.get("wave") == wave and n.get("description") == description:
            return
    notes.append({"wave": wave, "description": description})


def generate_change_note(
    old_row: Dict[str, Any],
    new_text: str,
    new_type: str,
    new_options: List[str],
    wave: str,
) -> str:
    """
    Use OpenAI to describe changes between the previous and current versions
    of a question. Focus on text and response option differences.
    """
    label = old_row["question_label"]
    old_text = old_row["question_text"] or ""
    old_type = old_row["question_type"] or ""
    old_options = json_load_or_empty(old_row["response_options"])

    old_options_str = "; ".join(old_options) if old_options else "(none)"
    new_options_str = "; ".join(new_options) if new_options else "(none)"

    system_prompt = (
        "You help maintain a change log for a tracking survey questionnaire.\n"
        "Given an old version and a new version of a question, describe what changed in "
        "a clear, concise, human-readable way.\n\n"
        "Focus on:\n"
        "- Response options that were added or removed (name them explicitly).\n"
        "- Response options whose wording changed (explain old → new).\n"
        "- Any changes to the question text (new phrases added, removed, or reworded).\n"
        "- Any change in question type if relevant (e.g., radio → checkbox).\n\n"
        "If the differences are only cosmetic or trivial, say 'No material changes.'.\n"
        "Keep your answer to 1–3 short bullet points or sentences."
    )

    user_prompt = (
        f"Wave: {wave}\n"
        f"Question label: {label}\n\n"
        f"Previous version:\n"
        f"- Type: {old_type}\n"
        f"- Question text: {old_text}\n"
        f"- Response options: {old_options_str}\n\n"
        f"New version:\n"
        f"- Type: {new_type}\n"
        f"- Question text: {new_text}\n"
        f"- Response options: {new_options_str}\n\n"
        "Describe all meaningful changes between the previous and new versions."
    )

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        note = response.choices[0].message.content.strip()
        if not note:
            note = (
                f"{wave}: Question wording or response options changed "
                f"compared with the previous wave."
            )
        return note
    except Exception as e:
        # Don't crash the parser if the API call fails
        print(f"Warning: could not generate change note for {label}: {e}", file=sys.stderr)
        return (
            f"{wave}: Question wording or response options changed "
            f"compared with the previous wave."
        )


def init_db(conn: sqlite3.Connection) -> None:
    """
    Ensure the survey_questions table and indexes exist.
    """
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    cur.execute(CREATE_UNIQUE_INDEX_SQL)
    cur.execute(CREATE_WAVES_TABLE_SQL)  # NEW
    conn.commit()


def load_existing_questions(conn: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    """
    Load all existing questions into a dict keyed by question_label.
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM survey_questions")
    rows = cur.fetchall()
    return {row["question_label"]: row for row in rows}


def json_load_or_empty(value: Any) -> List[Any]:
    """
    Safely load JSON list from a DB field, returning [] on error or None.
    """
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


# -----------------------------
# XML parsing
# -----------------------------

QUESTION_TAGS = ("radio", "checkbox", "select", "number")


def extract_questions_from_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Parse the XML and extract all questions of interest.

    Returns:
        List of dicts:
          {
            "label": str,
            "text": str,
            "type": str,           # 'radio', 'checkbox', 'select', 'number'
            "options": List[str],  # cleaned option texts
          }
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise SystemExit(f"Error parsing XML file {xml_path}: {e}")

    questions: List[Dict[str, Any]] = []

    # Iterate over supported question tag types
    for tag in QUESTION_TAGS:
        for elem in root.iter(tag):
            label = elem.get("label")
            if not label:
                # Skip questions without a label; they can't be tracked over time
                continue

            # Extract title/question text
            title_el = elem.find("title")
            raw_title = title_el.text if title_el is not None else ""
            question_text = clean_text(raw_title)

            # Extract response options depending on type
            options: List[str] = []
            if tag in ("radio", "checkbox", "number"):
                for row in elem.findall("row"):
                    opt_text = clean_text(row.text or "")
                    if opt_text:
                        options.append(opt_text)
            elif tag == "select":
                for choice in elem.findall("choice"):
                    opt_text = clean_text(choice.text or "")
                    if opt_text:
                        options.append(opt_text)

            questions.append(
                {
                    "label": label,
                    "text": question_text,
                    "type": tag,
                    "options": options,
                }
            )

    return questions


# -----------------------------
# Upsert logic
# -----------------------------

def upsert_questions(
    conn: sqlite3.Connection,
    questions: List[Dict[str, Any]],
    month: str,
) -> Dict[str, int]:
    """
    Insert/update questions in the DB for the given month.

    Args:
        conn: SQLite connection.
        questions: list of question dicts from extract_questions_from_xml.
        month: string 'YYYY-MM' representing the fielding month.

    Returns:
        Stats dict: {"inserted": int, "updated": int, "marked_inactive": int}
    """
    existing = load_existing_questions(conn)
    cur = conn.cursor()

    inserted = 0
    updated = 0
    marked_inactive = 0

    # Track which labels we saw in this wave
    seen_labels = set()

    for q in questions:
        label = q["label"]
        text = q["text"]
        q_type = q["type"]
        options = q.get("options", [])

        seen_labels.add(label)

        q_hash = compute_question_hash(label, q_type, text, options)

        if label not in existing:
            # -----------------------------
            # New question
            # -----------------------------
            months_asked_sorted = [month]
            months_json = json.dumps(months_asked_sorted)
            first_seen = month
            last_seen = month

            # Fresh question starts with empty notes list
            notes_json = json.dumps([], ensure_ascii=False)

            cur.execute(
                """
                INSERT INTO survey_questions (
                    question_label,
                    question_text,
                    question_type,
                    response_options,
                    frequency,
                    months_asked,
                    first_seen,
                    last_seen,
                    currently_active,
                    change_flag,
                    question_hash,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label,
                    text,
                    q_type,
                    json.dumps(options, ensure_ascii=False),
                    "Always",                # default frequency
                    months_json,
                    first_seen,
                    last_seen,
                    1,                       # currently_active
                    0,                       # change_flag
                    q_hash,
                    notes_json,
                ),
            )
            inserted += 1

        else:
            # -----------------------------
            # Existing question
            # -----------------------------
            row = existing[label]

            # Update months_asked
            months_asked = json_load_or_empty(row["months_asked"])
            if month not in months_asked:
                months_asked.append(month)

            months_asked_sorted = sorted(set(months_asked))  # 'YYYY-MM' sorts correctly
            months_json = json.dumps(months_asked_sorted)
            first_seen = months_asked_sorted[0]
            last_seen = months_asked_sorted[-1]

            # Detect content changes
            old_options = json_load_or_empty(row["response_options"])
            old_text = row["question_text"] or ""
            old_type = row["question_type"] or ""

            content_changed = (
                old_text != text
                or old_type != q_type
                or old_options != options
                or row["question_hash"] != q_hash
            )
            reactivated = (row["currently_active"] == 0)

            # Load existing notes (if any)
            # sqlite3.Row doesn't support .get(), so we use indexing.
            if "notes" in row.keys():
                notes_raw = row["notes"] or "[]"
            else:
                notes_raw = "[]"
            existing_notes = json_load_or_empty(notes_raw)

            # If the content changed, ask the LLM to describe the change
            if content_changed:
                desc = generate_change_note(row, text, q_type, options, month)
                append_note(existing_notes, wave=month, description=desc)
            elif reactivated:
                prev_last_seen = row["last_seen"] or "an earlier wave"
                desc = (
                    f"Question reintroduced in this wave after being inactive; "
                    f"previously last seen in {prev_last_seen}."
                )
                append_note(existing_notes, wave=month, description=desc)

            # Always keep notes sorted chronologically
            existing_notes_sorted = sorted(
                existing_notes,
                key=lambda n: n.get("wave", "0000-00"),
            )
            notes_json = json.dumps(existing_notes_sorted, ensure_ascii=False)

            needs_update = (
                    old_text != text
                    or old_type != q_type
                    or old_options != options
                    or row["months_asked"] != months_json
                    or row["first_seen"] != first_seen
                    or row["last_seen"] != last_seen
                    or row["currently_active"] != 1
                    or row["change_flag"] != 0
                    or row["question_hash"] != q_hash
                    or (notes_raw or "[]") != notes_json
            )

            if needs_update:
                cur.execute(
                    """
                    UPDATE survey_questions
                    SET question_text = ?,
                        question_type = ?,
                        response_options = ?,
                        months_asked = ?,
                        first_seen = ?,
                        last_seen = ?,
                        currently_active = 1,
                        change_flag = 0,
                        question_hash = ?,
                        notes = ?
                    WHERE question_label = ?
                    """,
                    (
                        text,
                        q_type,
                        json.dumps(options, ensure_ascii=False),
                        months_json,
                        first_seen,
                        last_seen,
                        q_hash,
                        notes_json,
                        label,
                    ),
                )
                updated += 1

    # -----------------------------
    # Mark questions inactive if missing this wave
    # -----------------------------
    for label, row in existing.items():
        if row["currently_active"] and label not in seen_labels:
            # If this wave is earlier than when the question first appears,
            # don't interpret the absence as a drop — the question simply
            # didn't exist yet.
            first_seen = row["first_seen"] or ""
            if first_seen and month < first_seen:
                continue

            frequency = row["frequency"] or "Always"
            change_flag = 1 if frequency == "Always" else 0

            # Load existing notes (sqlite3.Row -> use indexing)
            if "notes" in row.keys():
                notes_raw = row["notes"] or "[]"
            else:
                notes_raw = "[]"
            existing_notes = json_load_or_empty(notes_raw)

            prev_last_seen = row["last_seen"] or "an earlier wave"
            if change_flag:
                desc = (
                    f"Question not present in this wave; marked inactive "
                    f"(frequency='{frequency}'). Last seen in {prev_last_seen}."
                )
            else:
                desc = (
                    f"Question not present in this wave and deactivated "
                    f"(frequency='{frequency}'). Last seen in {prev_last_seen}."
                )

            # Store the note for this wave
            append_note(existing_notes, wave=month, description=desc)

            # Keep notes sorted chronologically
            existing_notes_sorted = sorted(
                existing_notes,
                key=lambda n: n.get("wave", "0000-00"),
            )
            notes_json = json.dumps(existing_notes_sorted, ensure_ascii=False)

            cur.execute(
                """
                UPDATE survey_questions
                SET currently_active = 0,
                    change_flag = ?,
                    notes = ?
                WHERE question_label = ?
                """,
                (change_flag, notes_json, label),
            )
            marked_inactive += 1


    conn.commit()
    return {"inserted": inserted, "updated": updated, "marked_inactive": marked_inactive}


# -----------------------------
# Main orchestration
# -----------------------------


def process_directory(dir_path: str, db_path: str) -> Dict[str, int]:
    """
    Process all XML / TXT files in a directory (e.g., 'QRE').

    - For each file, parse the wave from its filename.
    - Sort waves chronologically (YYYY-MM) so questionnaire changes are
      applied in time order.
    - For each wave, skip processing if it is already recorded in processed_waves.
    - Otherwise, process the file just like process_file.

    Returns:
      Aggregate stats across all processed files.
    """
    if not os.path.isdir(dir_path):
        raise SystemExit(f"Directory not found: {dir_path}")

    print(f"Scanning directory: {dir_path}")

    totals = {"inserted": 0, "updated": 0, "marked_inactive": 0}

    entries = sorted(os.listdir(dir_path))
    if not entries:
        print("Directory is empty; nothing to do.")
        return totals

    # Collect (wave, full_path) pairs
    files_with_waves: List[tuple[str, str]] = []
    for name in entries:
        full_path = os.path.join(dir_path, name)
        if not os.path.isfile(full_path):
            continue

        # Only handle .xml and .txt (adjust if you have other extensions)
        if not (name.lower().endswith(".xml") or name.lower().endswith(".txt")):
            continue

        try:
            wave = parse_month_from_filename(full_path)
        except ValueError as e:
            print(f"Skipping {full_path}: {e}")
            continue

        files_with_waves.append((wave, full_path))

    if not files_with_waves:
        print("No valid wave files found; nothing to do.")
        return totals

    # Sort by wave (YYYY-MM) so we always process in chronological order
    files_with_waves.sort(key=lambda x: x[0])

    for wave, full_path in files_with_waves:
        try:
            file_stats = process_file(full_path, db_path)
        except SystemExit as e:
            # If a single file fails to parse or is already processed, just warn and move on
            print(f"Skipping {full_path}: {e}")
            continue

        if file_stats:
            for key in totals:
                totals[key] += file_stats.get(key, 0)

    print("Directory processing complete.")
    print(
        f"Total inserted: {totals['inserted']}, "
        f"updated: {totals['updated']}, "
        f"marked inactive: {totals['marked_inactive']}"
    )
    print()

    return totals


def process_file(xml_path: str, db_path: str) -> Dict[str, int] | None:
    """
    High-level function:
      - Determine month from filename.
      - Skip if this wave is already recorded in the DB.
      - Extract questions.
      - Upsert into DB.
      - Record the wave as processed.
      - Print a short summary.

    Returns:
      stats dict, or None if the wave was skipped.
    """
    if not os.path.isfile(xml_path):
        raise SystemExit(f"XML file not found: {xml_path}")

    try:
        month = parse_month_from_filename(xml_path)
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"=== Processing file: {xml_path}")
    print(f"Detected fielding month: {month}")

    # Connect to DB
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)

        # Check if this wave has already been processed
        if is_wave_processed(conn, month):
            print(f"Wave {month} already processed for this database. Skipping file.")
            return None

        questions = extract_questions_from_xml(xml_path)
        print(f"Extracted {len(questions)} questions (types: {', '.join(QUESTION_TAGS)}).")

        stats = upsert_questions(conn, questions, month)

        # Mark this wave as processed
        mark_wave_processed(conn, month, xml_path)

    finally:
        conn.close()

    print("Done.")
    print(f"  Inserted new questions : {stats['inserted']}")
    print(f"  Updated existing       : {stats['updated']}")
    print(f"  Marked inactive        : {stats['marked_inactive']}")
    print()

    return stats


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse survey XML and update the questionnaire history database.\n\n"
            "You can pass either a single file path OR a directory path (e.g., 'QRE').\n"
            "If a directory is given, all .xml/.txt files in that directory will be processed.\n"
            "Waves that have already been processed for the given DB will be skipped.\n"
            "After updates (if any), semantic embeddings will be rebuilt automatically."
        )
    )
    parser.add_argument(
        "path",
        help="Path to an XML/.txt file OR a directory (e.g., 'QRE') containing many exports.",
    )
    parser.add_argument(
        "--db",
        default="survey_questions.sqlite",
        help="Path to the SQLite database file (will be created if it does not exist).",
    )

    args = parser.parse_args(argv)

    if os.path.isdir(args.path):
        # Directory mode (e.g., QRE folder)
        stats = process_directory(args.path, args.db)
    else:
        # Single-file mode (backwards compatible)
        stats = process_file(args.path, args.db)

    # Decide whether anything actually changed
    changed = bool(
        stats
        and (
            stats.get("inserted", 0)
            or stats.get("updated", 0)
            or stats.get("marked_inactive", 0)
        )
    )

    if changed:
        print("Changes detected in questionnaire history. Rebuilding embeddings...")
        try:
            build_all_embeddings(args.db)
            print("Embedding rebuild complete.")
        except Exception as e:
            # Don't kill the whole run if embedding rebuild fails
            print(f"Warning: failed to rebuild embeddings: {e}", file=sys.stderr)
    else:
        print("No DB changes detected; skipping embedding rebuild.")


if __name__ == "__main__":
    main()