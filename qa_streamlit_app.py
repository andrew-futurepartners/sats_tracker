#!/usr/bin/env python3
"""
qa_streamlit_app.py

Streamlit-based QA interface on top of survey_questions.sqlite
and question_embeddings built by build_embeddings.py.

Run:
    streamlit run qa_streamlit_app.py
"""

import json
import sqlite3
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)
EMBEDDING_MODEL = "text-embedding-3-small"  # good default for semantic search
CHAT_MODEL = "gpt-4.1-mini"   # or "gpt-4o-mini", depending on your account

import re

# Very small stopword list so we don't search on trivial words
STOPWORDS = {
    "the", "and", "or", "for", "with", "about", "what", "which", "when", "where",
    "how", "why", "are", "is", "was", "were", "do", "does", "did",
    "of", "to", "in", "on", "at", "a", "an", "any", "all",
    "list", "show", "give", "please",
    "question", "questions"
}

# Maximum number of question records we will send to the LLM in one answer.
# This is an internal safety cap; you never have to set it.
MAX_CONTEXT_QUESTIONS = 300

MONTH_NAME_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

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


# -----------------------------
# Embedding + LLM placeholders
# -----------------------------

def get_embedding(text: str) -> List[float]:
    """
    Get an embedding for the user question using the same model as build_embeddings.py.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def call_llm(user_question: str, context_rows: List[Dict[str, Any]]) -> str:
    """
    Use OpenAI's chat completions API to generate a natural language answer
    based on the most relevant survey questions.

    The model is instructed to ONLY use the provided context_rows and not hallucinate.
    """
    if not context_rows:
        return "I couldn’t find any relevant questions in the database."

    context_blocks = []
    for row in context_rows:
        label = row["question_label"]
        text = row["question_text"] or ""
        qtype_raw = row["question_type"] or ""
        qtype = friendly_question_type(qtype_raw)
        first_seen = row["first_seen"] or "unknown"
        last_seen = row["last_seen"] or "unknown"
        active = "yes" if row["currently_active"] else "no"
        freq = row["frequency"] or "Unknown"

        # Try to decode response options nicely
        try:
            options = json.loads(row["response_options"] or "[]")
        except Exception:
            options = []

        months_asked = row["months_asked"] or "[]"

        # Change history / notes (if present)
        # sqlite3.Row has no .get(), so use indexing
        if "notes" in row.keys():
            notes_raw = row["notes"] or "[]"
        else:
            notes_raw = "[]"

        try:
            notes = json.loads(notes_raw)
        except Exception:
            notes = []


        context_blocks.append(
            {
                "label": label,
                "question_text": text,
                "question_type": qtype,
                "frequency": freq,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "currently_active": active,
                "months_asked": months_asked,
                "response_options": options,
                "insights_explorer": bool(row.get("insights_explorer", 0)),
                "insights_section": row.get("insights_section") or "",
                "insights_page": row.get("insights_page") or "",
                "notes": notes,
            }
        )


    # Serialize context as JSON-ish text so the model can scan it
    context_text_lines = []
    for block in context_blocks:
        options_str = "; ".join(block["response_options"])

        # Format notes / change history nicely
        notes = block.get("notes") or []
        if notes:
            notes_lines = []
            for note in notes:
                wave = note.get("wave", "unknown")
                desc = note.get("description", "")
                notes_lines.append(f"    - {wave}: {desc}")
            notes_str = "\n".join(notes_lines)
            notes_section = f"  Change history:\n{notes_str}\n"
        else:
            notes_section = "  Change history: none\n"

        context_text_lines.append(
            f"Label: {block['label']}\n"
            f"  Question: {block['question_text']}\n"
            f"  Type: {block['question_type']}\n"
            f"  Frequency: {block['frequency']}\n"
            f"  First seen: {block['first_seen']}\n"
            f"  Last seen: {block['last_seen']}\n"
            f"  Currently active: {block['currently_active']}\n"
            f"  Months asked: {block['months_asked']}\n"
            f"  Options: {options_str}\n"
            f"  Insights Explorer: {block['insights_explorer']}, "
            f"Section: {block['insights_section']}, Page: {block['insights_page']}\n"
            f"{notes_section}"
        )

    context_text = "\n".join(context_text_lines)

    system_prompt = (
        "You are an assistant that answers questions about the State of the "
        "American Traveler survey questionnaire history.\n\n"
        "You are given:\n"
        "- A user question\n"
        "- A set of survey question records with metadata such as first_seen, "
        "last_seen, currently_active, frequency, response options, and sometimes "
        "a 'Change history' / 'notes' section describing how the question has changed.\n\n"
        "Rules:\n"
        "- Base your answer ONLY on the provided records.\n"
        "- If you are not sure or the data is missing, say so explicitly.\n"
        "- When you mention dates or labels, be precise (use the YYYY-MM format and label names).\n"
        "- Prefer concise answers followed by bullet points for details.\n"
        "- If the user asks about how a question has changed over time, rely first on "
        "the 'Change history' / notes for each question.\n"
        "- Do NOT add a separate summary section at the end (such as a list titled "
        "'Summary of labels and dates used'); instead, incorporate any labels and "
        "dates directly into your main answer."
    )

    user_prompt = (
        f"User question:\n{user_question}\n\n"
        f"Survey question records:\n{context_text}\n\n"
        "Please answer the user question using only these records. "
        "When helpful, mention question labels and date ranges inline in your explanation. "
        "Do not add a separate summary section; your answer should be a single, "
        "coherent explanation (with bullet points if useful), but no extra 'Summary' block."
    )


    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()



# -----------------------------
# DB and retrieval helpers
# -----------------------------

def connect_db(db_path: str) -> sqlite3.Connection:
    # Allow usage across threads (Streamlit reruns, callbacks, etc.)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn



def load_embeddings(conn: sqlite3.Connection) -> Dict[str, np.ndarray]:
    """
    Load embeddings from question_embeddings as a dict:
      question_label -> np.array
    """
    cur = conn.cursor()
    cur.execute("SELECT question_label, embedding FROM question_embeddings")
    rows = cur.fetchall()
    emb_map: Dict[str, np.ndarray] = {}
    for r in rows:
        label = r["question_label"]
        try:
            emb_list = json.loads(r["embedding"])
            emb_map[label] = np.array(emb_list, dtype=float)
        except Exception:
            continue
    return emb_map


def load_question_rows(conn: sqlite3.Connection, labels: List[str]) -> List[Dict[str, Any]]:
    if not labels:
        return []

    placeholders = ",".join("?" for _ in labels)
    sql = f"""
        SELECT *
        FROM survey_questions
        WHERE question_label IN ({placeholders})
    """
    cur = conn.cursor()
    cur.execute(sql, labels)
    rows = cur.fetchall()
    return [dict(r) for r in rows]

def extract_wave_tokens_from_question(text: str) -> List[str]:
    """
    Look for expressions like '2025-07', '2025/07', 'July 2025', 'Jul 2025'
    and normalize them to wave tokens 'YYYY-MM'.
    """
    lower = text.lower()
    waves: List[str] = []

    # Direct numeric YYYY-MM or YYYY/MM
    for m in re.finditer(r"(20\d{2})[-/](0[1-9]|1[0-2])", lower):
        year, month = m.group(1), m.group(2)
        waves.append(f"{year}-{month}")

    # Month name + year, e.g. "July 2025"
    for m in re.finditer(r"(20\d{2})", lower):
        year = m.group(1)
        window_start = max(0, m.start() - 20)
        window_end = min(len(lower), m.end() + 20)
        window = lower[window_start:window_end]
        for token in re.split(r"[^a-z]+", window):
            if token in MONTH_NAME_MAP:
                month = MONTH_NAME_MAP[token]
                waves.append(f"{year}-{month}")

    # Deduplicate, preserve order
    waves = list(dict.fromkeys(waves))
    return waves


def keyword_filter_questions(
    conn: sqlite3.Connection,
    user_question: str,
    max_results: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Keyword-based filter over question_text, response_options, notes,
    and date metadata (months_asked, first_seen, last_seen).

    Returns a list of question rows (as dicts) where at least one non-trivial
    keyword OR normalized wave token from the user question appears.
    """
    tokens = re.findall(r"\b\w+\b", user_question.lower())
    keywords = [
        t for t in tokens
        if len(t) > 2 and t not in STOPWORDS and not t.isdigit()
    ]

    # Add normalized wave tokens like '2025-07' if the user mentioned months/years
    wave_tokens = extract_wave_tokens_from_question(user_question)
    keywords.extend(wave_tokens)

    # Deduplicate keywords
    keywords = list(dict.fromkeys(keywords))

    if not keywords:
        return []

    like_clauses = []
    params: List[str] = []
    for kw in keywords:
        pattern = f"%{kw}%"
        like_clauses.append(
            "("
            "LOWER(question_text)    LIKE ? "
            "OR LOWER(response_options) LIKE ? "
            "OR LOWER(notes)           LIKE ? "
            "OR LOWER(months_asked)    LIKE ? "
            "OR LOWER(first_seen)      LIKE ? "
            "OR LOWER(last_seen)       LIKE ?"
            ")"
        )
        params.extend([pattern, pattern, pattern, pattern, pattern, pattern])

    where_clause = " OR ".join(like_clauses)
    sql = f"""
        SELECT *
        FROM survey_questions
        WHERE {where_clause}
    """

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    results = [dict(r) for r in rows]

    # Deduplicate by label, preserving order
    seen_labels = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        lbl = r.get("question_label")
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        deduped.append(r)

    if max_results is not None:
        return deduped[:max_results]
    return deduped

def extract_labels_from_answer(answer: str) -> List[str]:
    """
    Find label-like tokens (e.g., Q550, Q30) in the model's answer.

    Returns a de-duplicated list like ["Q550", "Q30"].
    """
    labels = re.findall(r"\bQ\d+\b", answer)
    # Deduplicate, preserve order
    return list(dict.fromkeys(labels))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_relevant_questions(
    user_question: str,
    emb_map: Dict[str, np.ndarray],
    max_k: int = 50,
    min_k: int = 1,
    base_threshold: float = 0.50,
    relative_margin: float = 0.05,
) -> List[Tuple[str, float]]:
    """
    Retrieve labels + similarity scores for questions relevant to the user query.

    - Compute similarity to every question embedding.
    - Compute a dynamic threshold: max(base_threshold, best_score - relative_margin).
    - Keep all questions at or above that threshold.
    - If that yields fewer than `min_k`, fall back to the top `min_k` matches.
    - Cap results at `max_k` to avoid extreme contexts.
    """
    if not emb_map:
        return []

    q_emb = np.array(get_embedding(user_question), dtype=float)
    scores: List[Tuple[str, float]] = []
    for label, emb in emb_map.items():
        score = cosine_similarity(q_emb, emb)
        scores.append((label, score))

    if not scores:
        return []

    # sort highest similarity first
    scores.sort(key=lambda x: x[1], reverse=True)
    best_score = scores[0][1]
    dynamic_threshold = max(base_threshold, best_score - relative_margin)

    filtered = [(lbl, s) for (lbl, s) in scores if s >= dynamic_threshold]

    # If too few, fall back to minimum top-k
    if len(filtered) < min_k:
        filtered = scores[:min_k]

    if max_k is not None:
        filtered = filtered[:max_k]

    return filtered


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="SATS.ai")

    st.title("SATS.ai")
    st.markdown(
        "Ask questions about the **history and status** of questions in The "
        "State of the American Traveler survey."
    )

    # Sidebar config
    st.sidebar.header("File")
    db_path = st.sidebar.text_input(
        "DB path",
        value="survey_questions.sqlite",
        help="Path to your DB (created by survey_parser.py).",
    )

    if "conn" not in st.session_state or st.session_state.get("db_path") != db_path:
        try:
            conn = connect_db(db_path)
            emb_map = load_embeddings(conn)
            st.session_state["conn"] = conn
            st.session_state["emb_map"] = emb_map
            st.session_state["db_path"] = db_path
        except Exception as e:
            st.error(f"Error connecting to DB: {e}")
            return

    conn = st.session_state["conn"]
    emb_map = st.session_state["emb_map"]

    if not emb_map:
        st.warning(
            "No embeddings found in question_embeddings table.\n\n"
            "Run `build_embeddings.py --db survey_questions.sqlite` first."
        )

    user_question = st.text_area(
        "Your question",
        placeholder="e.g., When did we first ask about travel insurance, and is it still active?",
        height=100,
    )

    if st.button("Ask") and user_question.strip():
        if not emb_map:
            st.error("Cannot answer: embeddings are missing. Run build_embeddings.py first.")
            return

        question_text = user_question.strip()

        with st.spinner("Finding an answer..."):
            # 1) Semantic retrieval over all embeddings (no explicit max_k)
            emb_matches = retrieve_relevant_questions(
                question_text,
                emb_map,
                max_k=None,           # no hard cap here
                min_k=1,
                base_threshold=0.50,
                relative_margin=0.05,
            )
            emb_labels = [m[0] for m in emb_matches]

            # 2) Keyword-based retrieval directly from SQLite
            keyword_rows = keyword_filter_questions(
                conn,
                question_text,
                max_results=None,     # no explicit cap here either
            )
            keyword_labels = [r["question_label"] for r in keyword_rows]

            # 3) Combine labels, preserving order: embeddings first, then keyword-only
            all_labels_ordered: List[str] = []
            seen = set()
            for lbl in emb_labels + keyword_labels:
                if lbl in seen:
                    continue
                seen.add(lbl)
                all_labels_ordered.append(lbl)

            # 4) Internal safety cap so we don't blow up the LLM context
            if emb_map:
                max_context = min(MAX_CONTEXT_QUESTIONS, len(emb_map))
            else:
                max_context = MAX_CONTEXT_QUESTIONS

            if len(all_labels_ordered) > max_context:
                all_labels_ordered = all_labels_ordered[:max_context]

            # 5) Load rows for combined labels and call LLM
            rows = load_question_rows(conn, all_labels_ordered)
            answer = call_llm(question_text, rows)

        # --- Only keep questions actually referenced in the answer ---
        used_label_bases = extract_labels_from_answer(answer)

        if used_label_bases:
            def is_used(row: Dict[str, Any]) -> bool:
                lbl = row["question_label"] or ""
                # Keep row if any base label (e.g. "Q550") appears in the full label
                return any(base in lbl for base in used_label_bases)

            rows = [r for r in rows if is_used(r)]
        # If no labels were found in the answer, we leave rows as-is (fallback).

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Question Details")
        if not rows:
            st.write("No questions found.")
        else:
            for r in rows:
                label = r["question_label"]
                qtext = r["question_text"] or ""

                # Short preview for the outer expander title
                if qtext:
                    preview = qtext.strip().replace("\n", " ")
                    if len(preview) > 110:
                        preview = preview[:110].rstrip() + "…"
                    header = f"{label}: {preview}"
                else:
                    header = label

                # Single-level expander per question
                with st.expander(header):
                    # Label + full question text
                    st.markdown(f"**Label:** `{label}`")
                    if qtext:
                        st.markdown(f"**Question text:** {qtext}")

                    st.markdown("---")
                    st.markdown("**Attributes & metadata**")
                    st.markdown(f"- Type: `{friendly_question_type(r['question_type'])}`")
                    st.markdown(f"- Frequency: `{r['frequency']}`")
                    st.markdown(f"- First seen: `{r['first_seen']}`")
                    st.markdown(f"- Last seen: `{r['last_seen']}`")
                    st.markdown(
                        f"- Currently active: "
                        f"`{'yes' if r['currently_active'] else 'no'}`"
                    )
                    st.markdown(
                        f"- Change flag: "
                        f"`{'yes' if r['change_flag'] else 'no'}`"
                    )
                    st.markdown(
                        f"- Insights Explorer: "
                        f"`{'yes' if r['insights_explorer'] else 'no'}` "
                        f"(Section: `{r['insights_section'] or ''}`, "
                        f"Page: `{r['insights_page'] or ''}`)"
                    )

                    # Response options
                    try:
                        opts = json.loads(r["response_options"] or "[]")
                    except Exception:
                        opts = []
                    if opts:
                        st.markdown("**Response options:**")
                        for o in opts:
                            st.markdown(f"- {o}")

                    # Change history / notes
                    notes_raw = r.get("notes") or "[]"
                    try:
                        notes = json.loads(notes_raw)
                    except Exception:
                        notes = []

                    if notes:
                        st.markdown("**Change history / notes:**")
                        for note in notes:
                            wave = note.get("wave", "unknown")
                            desc = note.get("description", "")
                            st.markdown(f"- `{wave}`: {desc}")


if __name__ == "__main__":
    main()