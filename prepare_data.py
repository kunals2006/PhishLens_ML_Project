"""Create a compact, labeled sample from the Kaggle URL CSV."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw" / "phishing_site_urls.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "phishing_small.csv"
MAX_ROWS = 15000


def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Download the Kaggle CSV and save it here: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    url_col = find_column(df.columns, ["url", "website", "link"])
    label_col = find_column(df.columns, ["label", "labels", "type", "class", "result", "prediction"])
    if url_col is None or label_col is None:
        raise ValueError(f"Could not identify URL and label columns. Found: {list(df.columns)}")
    compact = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"}).dropna()
    compact["url"] = compact["url"].astype(str).str.strip()
    original_labels = compact["label"].astype(str).str.strip().str.lower()
    label_map = {"good": 0, "legitimate": 0, "benign": 0, "0": 0,
                 "bad": 1, "phishing": 1, "malicious": 1, "1": 1}
    compact = compact[original_labels.isin(label_map)].copy()
    compact["label"] = original_labels[original_labels.isin(label_map)].map(label_map).astype(int)
    compact = compact.drop_duplicates(subset=["url"])
    if compact["label"].nunique() != 2:
        raise ValueError("Expected both benign and phishing labels (for example, 'good' and 'bad').")
    if len(compact) > MAX_ROWS:
        compact, _ = train_test_split(compact, train_size=MAX_ROWS, random_state=42, stratify=compact["label"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(compact):,} unique URLs to {OUTPUT_PATH}")
    print("Labels: 0 = legitimate, 1 = phishing")
    print(compact["label"].value_counts().rename(index={0: "legitimate", 1: "phishing"}).to_string())


if __name__ == "__main__":
    main()
