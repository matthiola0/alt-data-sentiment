"""Generate the Kaggle-ready ipynb file used to offload FinBERT
scoring to a free T4 GPU session:

    score_leukipp_on_kaggle.ipynb

The notebook is self-contained (S&P 500 whitelist + BLACKLIST +
entity linking + FinBERT loop inlined), so it runs with internet OFF
on Kaggle and only needs the dataset attached as an input.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "scripts" / "kaggle"
TICKERS_CSV = REPO_ROOT / "data" / "raw" / "sp500_tickers.csv"

# --- shared pieces -----------------------------------------------------

def _whitelist_literal() -> str:
    syms = sorted(
        pd.read_csv(TICKERS_CSV)["symbol"]
        .str.upper()
        .str.replace(".", "-", regex=False)
        .unique()
    )
    # Wrap as a frozenset literal, 8 per line for readability
    chunks = [", ".join(repr(s) for s in syms[i : i + 8]) for i in range(0, len(syms), 8)]
    body = ",\n    ".join(chunks)
    return f"SP500_WHITELIST = frozenset({{\n    {body}\n}})"


BLACKLIST_CODE = '''BLACKLIST = frozenset({
    "AN","OR","AT","TO","IS","OF","IN","IF","BE","AM","AS","DO","GO",
    "UP","US","WE","HE","NO","OK","ANY","FOR","NOT","BUT","CAN","ONE",
    "TWO","WAY","NEW","OUR","OUT","HOW","WHY","WHO","HIS","HER","HAD",
    "GET","GOT","OWN","SEE","USE","PUT","DAY","END","LOT","MAY","TOP",
    "BIG","OFF","MUCH","MORE","MOST","SOME","ONLY","JUST","EVEN","ALSO",
    "GOOD","BEST","MAKE","SAID","BACK","OVER","LIKE","THAT","THIS","WITH",
    "WHAT","WHEN","YEAR","TIME","NEXT","LAST","WEEK","STILL","GONNA",
    "WANNA","DONT","CANT","WONT","ISNT","ITS","HES","SHES",
    # WSB slang tickers (deliberately drop these companies)
    "DD","IT","COO",
    "WSB","YOLO","FOMO","EV","OP","PT","ATH","ATL","ITM","OTM","IV","HV",
    "SL","TP","CE","PE","MC","CC","HOLD",
    # finance / macro acronyms
    "USD","EUR","GBP","JPY","CEO","CFO","CTO","IPO","ETF","EPS","PB",
    "ROE","ROI","ROIC","FCF","GDP","CPI","PPI","PMI","FOMC","FED","SEC",
    "IRS","IRA","SSN","LLC","INC","NYSE","NASDAQ","SPY","QQQ","DIA",
    "IWM","VIX","FY","Q1","Q2","Q3","Q4","H1","H2","YOY","YTD","MOM","QOQ",
    # calendars — MAR intentionally drops Marriott
    "JAN","FEB","MAR","APR","JUN","JUL","AUG","SEP","OCT","DEC",
    "MON","TUE","WED","THU","FRI","SAT","SUN",
    # loud interjections
    "LOL","LMAO","WTF","OMG","IMO","IMHO","TLDR","AFAIK","IIRC","EDIT",
    "UPDATE","NEWS","POST","LINK","HERE","REAL",
})
'''

COMMON_IMPORTS_AND_LINKING = '''
import os, re, time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("torch.cuda.is_available() =", torch.cuda.is_available())
print("device  =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

CANDIDATE_RE = re.compile(r"\\$?([A-Z]{2,5})\\b")

def extract_tickers(text):
    if not isinstance(text, str) or not text:
        return []
    seen, out = set(), []
    for m in CANDIDATE_RE.finditer(text):
        sym = m.group(1)
        if sym in BLACKLIST or sym not in SP500_WHITELIST or sym in seen:
            continue
        seen.add(sym); out.append(sym)
    return out
'''

FINBERT_LOAD = '''
# CUDA compatibility guard: Kaggle sometimes assigns P100 (sm_60), which
# the preinstalled PyTorch no longer compiles kernels for. Fall back to
# CPU rather than error out mid-run.
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap < (7, 0):
        print(f"GPU {torch.cuda.get_device_name(0)} compute capability "
              f"{cap} < 7.0 — not supported by installed torch; using CPU.")
        device = "cpu"
    else:
        device = "cuda"
else:
    device = "cpu"
print("final device =", device)

MODEL_ID = "ProsusAI/finbert"
print("Loading FinBERT ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
model.to(device)
print(f"FinBERT ready on {device} in {time.time()-t0:.1f}s")

# Column reorder to [pos, neg, neu] regardless of HF label order
id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
PERM = [next(i for i, l in id2label.items() if l.startswith(want[:3]))
        for want in ["positive", "negative", "neutral"]]
print("id2label =", id2label, "perm =", PERM)

@torch.inference_mode()
def score_batch(texts, batch_size=64, max_length=512):
    texts = list(texts)
    n = len(texts); out = np.full((n, 3), np.nan, dtype=np.float32)
    usable = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not usable: return out
    for b0 in range(0, len(usable), batch_size):
        idx = usable[b0:b0+batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, PERM]
        for r, dst in enumerate(idx):
            out[dst] = probs[r]
    return out
'''

LEUKIPP_CELL = '''
# leukipp/reddit-finance-data (submissions only, 2021)

import glob
def _dump_tree(root, max_depth=4):
    for dirpath, dirs, files in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []; continue
        print(dirpath + "/", [f for f in files][:10])

_candidates = glob.glob("/kaggle/input/**/wallstreetbets/submissions_reddit.csv", recursive=True)
if not _candidates:
    print("FATAL: no submissions_reddit.csv under /kaggle/input/. Tree:")
    _dump_tree("/kaggle/input")
    raise SystemExit("Attach dataset leukipp/reddit-finance-data via +Add data")
DATASET_ROOT = str(Path(_candidates[0]).parent.parent)  # <...>/<subreddit>/<file>.csv -> root
print("DATASET_ROOT =", DATASET_ROOT)

SUBREDDITS = ("wallstreetbets", "stocks", "investing", "options")
OUT_DIR    = Path("/kaggle/working/scored_leukipp"); OUT_DIR.mkdir(parents=True, exist_ok=True)
USECOLS    = ["id", "created", "title", "selftext", "score"]
CHUNK      = 100_000
BATCH      = 64

def combine_text(t, b):
    parts = []
    if isinstance(t, str) and t.strip(): parts.append(t.strip())
    if isinstance(b, str) and b.strip(): parts.append(b.strip())
    return " ".join(parts)

t_run = time.time(); total_docs = 0; chunk_gi = 0
for sub in SUBREDDITS:
    csv = f"{DATASET_ROOT}/{sub}/submissions_reddit.csv"
    print(f"--- {sub}: {csv}")
    for ch in pd.read_csv(csv, usecols=USECOLS, chunksize=CHUNK):
        t0 = time.time()
        texts = [combine_text(t, b) for t, b in zip(ch["title"], ch["selftext"])]
        df = pd.DataFrame({
            "doc_id": f"leukipp/{sub}/" + ch["id"].astype(str),
            "subreddit": sub,
            "created_utc": pd.to_datetime(ch["created"], errors="coerce"),
            "text": texts,
            "score": pd.to_numeric(ch["score"], errors="coerce"),
            "is_comment": False,
        })
        df = df[df["text"].str.len() > 0]
        df = df[df["created_utc"].notna()]
        df["_tickers"] = [extract_tickers(t) for t in df["text"]]
        linked = df[df["_tickers"].str.len() > 0].reset_index(drop=True)
        if not linked.empty:
            probs = score_batch(linked["text"].tolist(), batch_size=BATCH)
            linked["score_pos"] = probs[:, 0]
            linked["score_neg"] = probs[:, 1]
            linked["score_neu"] = probs[:, 2]
            fanned = linked.explode("_tickers", ignore_index=True).rename(columns={"_tickers": "ticker"})
            out = fanned[["doc_id","subreddit","created_utc","ticker","is_comment",
                          "score","score_pos","score_neg","score_neu"]].rename(columns={"score":"doc_score"})
            out.to_parquet(OUT_DIR / f"part_{chunk_gi:05d}.parquet", index=False)
            total_docs += linked["doc_id"].nunique()
        dt = time.time() - t0
        print(f"  {sub:16s} chunk_gi={chunk_gi:04d}  in={len(ch):>6}  linked={len(linked):>5}  {dt:6.1f}s", flush=True)
        chunk_gi += 1

print(f"\\nDONE leukipp: {total_docs:,} unique linked docs in {(time.time()-t_run)/60:.1f} min")
print("Parts written to /kaggle/working/scored_leukipp/")
'''


def make_notebook(title: str, dataset_cell: str, out_name: str, input_dataset_slug: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            "**Run this on Kaggle with a T4 GPU accelerator** — "
            "it offloads FinBERT scoring that is impractical on CPU.\n\n"
            f"**Required input dataset (Add Data):** `{input_dataset_slug}`\n\n"
            "**Accelerator:** Settings → Accelerator → **GPU T4 x2** (or T4 x1).\n\n"
            "**Internet:** can be OFF. The S&P 500 whitelist is inlined below.\n\n"
            "Output goes to `/kaggle/working/scored_{dataset}/part_*.parquet`. "
            "After Run All, click the Output pane on the right to download "
            "the folder as a zip; then place the parts under "
            "`alt-data-sentiment/data/processed/scored_{dataset}/` locally."
        ),
        nbf.v4.new_code_cell(_whitelist_literal() + "\nprint(len(SP500_WHITELIST), 'S&P 500 symbols')"),
        nbf.v4.new_code_cell(BLACKLIST_CODE + "\nprint(len(BLACKLIST), 'blacklist entries')"),
        nbf.v4.new_code_cell(COMMON_IMPORTS_AND_LINKING),
        nbf.v4.new_code_cell(FINBERT_LOAD),
        nbf.v4.new_code_cell(dataset_cell),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = OUT_DIR / out_name
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path}  ({path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    make_notebook(
        "Score leukipp/reddit-finance-data on Kaggle GPU",
        LEUKIPP_CELL,
        "score_leukipp_on_kaggle.ipynb",
        "leukipp/reddit-finance-data",
    )
