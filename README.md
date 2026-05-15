# SATS Tracker

**SATS Tracker** is a survey questionnaire management and AI-powered Q&A system. It ingests Decipher-style survey XML exports, tracks question history across monthly waves, builds semantic embeddings, and provides an interactive Streamlit interface for exploring and querying survey data using natural language.

## Features

- **Survey Parsing** — Automatically parses Decipher XML exports (radio, checkbox, select, number question types) and extracts question labels, text, response options, and types.
- **Wave Tracking** — Tracks which months each question appears in, detects when questions are added, removed, or modified between waves, and records a full change history.
- **LLM-Powered Change Notes** — When a question's wording or options change between waves, GPT generates a human-readable summary of what changed.
- **Semantic Embeddings** — Builds vector embeddings for every question using OpenAI's `text-embedding-3-small` model, enabling similarity-based retrieval.
- **AI Q&A Interface** — A Streamlit app lets users ask natural-language questions about the survey. Hybrid retrieval (cosine similarity + keyword filtering) grounds GPT answers in actual questionnaire data.
- **Questionnaire Matrix** — A full visual matrix of questions by month, with expandable detail panels showing options, change history, and metadata.

## Architecture

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  QRE/ folder     │────▶│  survey_parser.py    │────▶│  SQLite database     │
│  (XML exports)   │     │  (ingest & track)    │     │  survey_questions    │
└──────────────────┘     └────────┬────────────┘     │  processed_waves     │
                                  │                   └──────────┬───────────┘
                                  ▼                              │
                         ┌─────────────────────┐                │
                         │ build_embeddings.py  │◀───────────────┘
                         │ (vectorize questions)│
                         └────────┬────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ question_embeddings  │
                         │ (SQLite table)       │
                         └────────┬────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ qa_streamlit_app.py  │
                         │ (Q&A + Matrix UI)    │
                         └─────────────────────┘
```

## Project Structure

```
SATS_Tracker/
├── survey_parser.py        # XML ingestion pipeline and change tracker
├── build_embeddings.py     # Generates OpenAI embeddings for all questions
├── qa_streamlit_app.py     # Streamlit app (Q&A chat + questionnaire matrix)
├── requirements.txt        # Python dependencies
├── survey_questions.sqlite # SQLite database (generated)
├── .streamlit/
│   ├── config.toml         # Streamlit server/browser settings
│   └── secrets.toml        # Local secrets (not committed)
├── QRE/                    # Monthly survey XML exports (.txt files)
│   ├── SATS January 2025.txt
│   ├── SATS February 2025.txt
│   └── ...
├── .env                    # Environment variables — local fallback
├── .gitignore              # Ignores secrets, DB, cache, IDE files
└── notes.txt               # Quick-reference run commands
```

## Prerequisites

- **Python 3.10+**
- An **OpenAI API key** with access to embeddings and chat completions

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd SATS_Tracker
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key** (pick one method):

   **Option A — `.env` file** (traditional):

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

   **Option B — Streamlit secrets** (works locally and on Community Cloud):

   Create `.streamlit/secrets.toml`:

   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

   The app checks `st.secrets` first, then falls back to the `.env` file.

## Usage

### 1. Parse Survey Exports

Process a directory of XML/TXT survey exports to populate the database:

```bash
python survey_parser.py QRE --db survey_questions.sqlite
```

Or process a single file:

```bash
python survey_parser.py "QRE/SATS July 2025.txt" --db survey_questions.sqlite
```

The parser will:
- Extract questions from each file and determine the survey wave (`YYYY-MM`) from the filename.
- Insert new questions, update changed ones, and mark missing questions as inactive.
- Skip waves that have already been processed.
- Automatically rebuild embeddings if any changes were detected.

### 2. Rebuild Embeddings (standalone)

If you need to regenerate embeddings without re-parsing:

```bash
python build_embeddings.py --db survey_questions.sqlite
```

### 3. Launch the Streamlit App

```bash
streamlit run qa_streamlit_app.py
```

The app opens in your browser with two tabs:

- **Q&A** — Type a natural-language question (e.g., *"Which questions ask about brand awareness?"*). The system retrieves relevant questions via hybrid search (semantic similarity + keyword matching) and generates a grounded answer using GPT.
- **Full Question List** — Browse the complete questionnaire matrix showing which questions appeared in which months. Select any question to view its full details, response options, and change history.

## Deployment

### Local

Run the app directly from your machine:

```bash
streamlit run qa_streamlit_app.py
```

The API key can be provided via either a `.env` file or `.streamlit/secrets.toml` (see Installation step 3).

### Streamlit Community Cloud

1. **Push your repo to GitHub.** Make sure the following are committed:
   - `qa_streamlit_app.py`, `survey_parser.py`, `build_embeddings.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `survey_questions.sqlite` (the database must be in the repo since Community Cloud has no persistent filesystem)

   The `.gitignore` already prevents `.env`, `.streamlit/secrets.toml`, and other local artifacts from being committed.

2. **Connect your repo** at [share.streamlit.io](https://share.streamlit.io):
   - Set the main file path to `qa_streamlit_app.py`.

3. **Add your API key** in the deployment's **Advanced Settings > Secrets** field:

   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

4. **Deploy.** The app will install dependencies from `requirements.txt` automatically.

> **Note:** Because Community Cloud does not provide persistent storage, the SQLite database must be committed to the repository. To update the database, run the parser locally, then commit and push the updated `survey_questions.sqlite`.

## Database Schema

### `survey_questions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `question_label` | TEXT | Unique question identifier (e.g., Q1, Q2a) |
| `question_text` | TEXT | Full question wording |
| `question_type` | TEXT | radio, checkbox, select, or number |
| `response_options` | TEXT | JSON array of option strings |
| `frequency` | TEXT | How many waves the question has appeared in |
| `months_asked` | TEXT | JSON array of `YYYY-MM` strings |
| `first_seen` | TEXT | First wave the question appeared |
| `last_seen` | TEXT | Most recent wave the question appeared |
| `currently_active` | INTEGER | 1 if present in the latest wave, 0 otherwise |
| `change_flag` | INTEGER | 1 if modified since last wave |
| `question_hash` | TEXT | SHA-256 hash for change detection |
| `notes` | TEXT | JSON array of `{wave, description}` change notes |

Note: Legacy columns `insights_explorer`, `insights_section`, and `insights_page` may still exist in older database files but are no longer used.

### `processed_waves`

| Column | Type | Description |
|--------|------|-------------|
| `wave` | TEXT | Wave identifier (`YYYY-MM`), primary key |
| `source_file` | TEXT | Filename that was processed |
| `processed_at` | TEXT | Timestamp of processing |

### `question_embeddings`

| Column | Type | Description |
|--------|------|-------------|
| `question_label` | TEXT | Primary key, references survey_questions |
| `embedding` | TEXT | JSON array of floats (vector from `text-embedding-3-small`) |

## Configuration

All configuration is handled through environment variables and in-code constants:

| Setting | Location | Default |
|---------|----------|---------|
| `OPENAI_API_KEY` | `.streamlit/secrets.toml`, Cloud dashboard, or `.env` | *(required)* |
| Embedding model | `build_embeddings.py`, `qa_streamlit_app.py` | `text-embedding-3-small` |
| Chat model | `survey_parser.py`, `qa_streamlit_app.py` | `gpt-4.1-mini` |
| Max context questions | `qa_streamlit_app.py` | 300 |
| Default database path | All scripts | `survey_questions.sqlite` |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Database | SQLite |
| AI/ML | OpenAI API (embeddings + chat completions) |
| Similarity Search | NumPy (cosine similarity) |
| Data Display | Pandas |
| XML Parsing | xml.etree.ElementTree |
| Secrets Management | Streamlit secrets + python-dotenv (fallback) |
