"""
Sentiment pipeline: merge scraper CSVs → clean → score with StanceBERTa.

Stage 1 – Preprocess
  Reads every CSV in an input directory, detects source by filename prefix
  (4chan_, reddit_, telegram_), normalises each row into a shared schema,
  and writes a single merged CSV.

Stage 2 – Score
  Loads the merged CSV, runs StanceBERTa in batches, writes a scored CSV
  with a final `sentiment` column in [-1, +1].

Output columns (merged CSV)
───────────────────────────
source         – origin platform (4chan / reddit / telegram)
text_cleaned   – post text with URLs and extra whitespace stripped
url            – link back to the original thread or message
author         – who wrote it (Anonymous for 4chan)
created_utc    – unix timestamp when available
score          – upvotes / reactions (0 when unavailable)
is_question    – 1 if the text ends with '?', 0 otherwise
thread_id      – groups replies with their parent thread
is_op          – 1 for the opening post, 0 for a reply
"""

import csv
import re
from pathlib import Path
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pycld2 as cld2
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────────────────────────
# Constants
# ───────────────────────────────────────────

STANCEBERTA = "eevvgg/StanceBERTa"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPLY_SEP = " ||| "

FIELDNAMES = [
    "source",
    "text_cleaned",
    "url",
    "author",
    "created_utc",
    "score",
    "is_question",
    "thread_id",
    "is_op",
]

# ───────────────────────────────────────────
# Stage 1: text helpers
# ───────────────────────────────────────────

def detect_english_cld2(text: str) -> bool:
    try:
        for lang_code, _pct, _, _ in cld2.detect(text):
            if lang_code == "en":
                return True
    except Exception as e:
        print(e)
    return False


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _is_question(text: str) -> bool:
    return bool(re.search(r"\?\s*$", text))


# ───────────────────────────────────────────
# Stage 1: per-platform processors
# ───────────────────────────────────────────

def _process_4chan(file_path: str) -> List[Dict]:
    rows: List[Dict] = []
    df = pd.read_csv(file_path)
    for _, r in df.iterrows():
        thread_url = str(r.get("url", ""))
        thread_id = thread_url.rstrip("/").split("/")[-1] if thread_url else ""
        op_text = str(r.get("post", ""))
        rows.append({
            "source": "4chan",
            "text_cleaned": clean_text(op_text),
            "url": thread_url,
            "author": "Anonymous",
            "created_utc": None,
            "score": 0,
            "is_question": int(_is_question(op_text)),
            "thread_id": thread_id,
            "is_op": 1,
        })
        replies_raw = str(r.get("replies", ""))
        if replies_raw and replies_raw != "nan":
            for reply in replies_raw.split(REPLY_SEP):
                reply = reply.strip()
                if not reply:
                    continue
                rows.append({
                    "source": "4chan",
                    "text_cleaned": clean_text(reply),
                    "url": thread_url,
                    "author": "Anonymous",
                    "created_utc": None,
                    "score": 0,
                    "is_question": int(_is_question(reply)),
                    "thread_id": thread_id,
                    "is_op": 0,
                })
    return rows


def _process_reddit(file_path: str) -> List[Dict]:
    rows: List[Dict] = []
    df = pd.read_csv(file_path)
    for _, r in df.iterrows():
        post_id = str(r.get("id", ""))
        post_text = str(r.get("post", ""))
        post_url = str(r.get("url", ""))
        rows.append({
            "source": "reddit",
            "text_cleaned": clean_text(post_text),
            "url": post_url,
            "author": str(r.get("author", "")) if pd.notna(r.get("author")) else "",
            "created_utc": r.get("created_utc") if pd.notna(r.get("created_utc")) else None,
            "score": r.get("score") if pd.notna(r.get("score")) else 0,
            "is_question": int(_is_question(post_text)),
            "thread_id": post_id,
            "is_op": 1,
        })
        replies_raw = str(r.get("replies", ""))
        if replies_raw and replies_raw != "nan":
            for reply in replies_raw.split(REPLY_SEP):
                reply = reply.strip()
                if not reply:
                    continue
                rows.append({
                    "source": "reddit",
                    "text_cleaned": clean_text(reply),
                    "url": post_url,
                    "author": None,
                    "created_utc": None,
                    "score": None,
                    "is_question": int(_is_question(reply)),
                    "thread_id": post_id,
                    "is_op": 0,
                })
    return rows


def _process_telegram(file_path: str) -> List[Dict]:
    rows: List[Dict] = []
    fname = Path(file_path).name
    channel = re.sub(r"^telegram_", "", fname)
    channel = re.sub(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}\.csv$", "", channel)

    df = pd.read_csv(file_path)
    for _, r in df.iterrows():
        msg_text = str(r.get("text", "")) if pd.notna(r.get("text")) else ""
        msg_date = str(r.get("date", "")) if pd.notna(r.get("date")) else None
        created_utc = None
        if msg_date:
            try:
                created_utc = datetime.fromisoformat(msg_date).timestamp()
            except (ValueError, TypeError):
                pass
        rows.append({
            "source": "telegram",
            "text_cleaned": clean_text(msg_text),
            "url": str(r.get("urls", "")) if pd.notna(r.get("urls")) else "",
            "author": str(r.get("sender_id", "")) if pd.notna(r.get("sender_id")) else None,
            "created_utc": created_utc,
            "score": 0,
            "is_question": int(_is_question(msg_text)) if msg_text else 0,
            "thread_id": channel,
            "is_op": 1,
        })
    return rows


_PROCESSORS = {
    "4chan_": _process_4chan,
    "reddit_": _process_reddit,
    "telegram_": _process_telegram,
}


def preprocess(input_dir: str, output_dir: str, filename: str) -> str:
    """Merge all today's scraper CSVs in *input_dir* into one cleaned CSV."""
    date = datetime.now().strftime("%Y_%m_%d")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = str(Path(output_dir) / filename)
    all_rows: List[Dict] = []

    for fname in sorted(p.name for p in Path(input_dir).iterdir()):
        if not fname.endswith(".csv"):
            continue
        filepath = str(Path(input_dir) / fname)
        print(f"Processing {fname}…", file=sys.stderr)
        matched = False
        for prefix, processor in _PROCESSORS.items():
            if fname.startswith(prefix) and fname.endswith(f"{date}.csv"):
                rows = processor(filepath)
                print(f"  -> {len(rows)} rows from {prefix.rstrip('_')}", file=sys.stderr)
                all_rows.extend(rows)
                matched = True
                break
        if not matched:
            print("  -> Skipping (unknown prefix)", file=sys.stderr)

    print(f"\nTotal rows: {len(all_rows)}", file=sys.stderr)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV written to {output}", file=sys.stderr)
    return output


# ───────────────────────────────────────────
# Stage 2: scoring helpers
# ───────────────────────────────────────────

def _preprocess_for_stance(text: str) -> str:
    """StanceBERTa was trained with @user and http tokens."""
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"https?://\S+|www\.\S+", "http", text).strip()


def _safe_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "."
    return text.strip()


# ───────────────────────────────────────────
# Stage 2: scorer
# ───────────────────────────────────────────

class SentimentScorer:
    """Load a HF model once and batch-score texts into [-1, +1]."""

    def __init__(self, model_name: str, max_length: int = 512):
        print(f"Loading {model_name} …")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE).eval()
        self._preprocess = (
            _preprocess_for_stance if "stance" in model_name.lower() else lambda t: t
        )

    def score_one(self, text: str) -> float:
        text = self._preprocess(_safe_text(text))
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True,
            max_length=self.max_length,
        ).to(DEVICE)
        with torch.no_grad():
            probs = (
                torch.nn.functional.softmax(self.model(**inputs).logits, dim=-1)
                .squeeze().cpu().tolist()
            )
        return self._probs_to_score(probs)

    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        texts = [self._preprocess(_safe_text(t)) for t in texts]
        scores: List[float] = []
        for start in tqdm(range(0, len(texts), batch_size),
                          desc=self.model_name.split("/")[-1], leave=False):
            batch = texts[start: start + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True,
                max_length=self.max_length,
            ).to(DEVICE)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(
                    self.model(**inputs).logits, dim=-1
                ).cpu().tolist()
            for prob_row in probs:
                scores.append(self._probs_to_score(prob_row))
        return scores

    def _probs_to_score(self, probs: list) -> float:
        """[+1 × P(pos) + -1 × P(neg)] × [1 − P(neu)]  (StanceBERTa label order)"""
        if "stance" in self.model_name.lower():
            return (probs[1] - probs[2]) * (1 - probs[0])
        return 0.0


# ───────────────────────────────────────────
# Stage 2: score entry point
# ───────────────────────────────────────────

def score(
    input_file: str,
    output_dir: str,
    filename: str,
    models: List[str] = None,
) -> str:
    if models is None:
        models = [STANCEBERTA]

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    df = df.dropna(subset=["text_cleaned"])
    df = df[df["text_cleaned"].str.strip().astype(bool)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"{len(df)} rows after dropping empty text")

    texts = df["text_cleaned"].tolist()
    score_cols: List[str] = []

    for model_name in models:
        scorer = SentimentScorer(model_name)
        col = f"{model_name}_score"
        df[col] = scorer.score_batch(texts)
        score_cols.append(col)
        del scorer
        torch.cuda.empty_cache()

    if score_cols:
        df["sentiment"] = df[score_cols].mean(axis=1).round(6)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / filename)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} scored rows → {out_path}")

    print("\n── Score summary ──")
    for col in score_cols + ["sentiment"]:
        if col in df.columns:
            print(f"  {col:>15s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")
    if "source" in df.columns:
        print("\n── By source ──")
        print(df.groupby("source")["sentiment"].agg(["mean", "std", "count"]).to_string())

    return out_path


# ───────────────────────────────────────────
# CLI: run both stages end-to-end
# ───────────────────────────────────────────

if __name__ == "__main__":
    DATE = datetime.now().strftime("%Y_%m_%d")

    merged_file = preprocess(
        input_dir="./datasets",
        output_dir="./datasets/sentiment",
        filename=f"sentiment_{DATE}.csv",
    )

    score(
        input_file=merged_file,
        output_dir="./datasets/scores",
        filename=f"scored_sentiment_{DATE}.csv",
    )
