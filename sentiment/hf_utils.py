try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

import pandas as pd


def load_hf_dataset(dataset_id="tweet_eval", subset="sentiment"):
    """Load a HuggingFace dataset and return a pandas DataFrame with `text` and `label` (0/1).

    This function handles common label schemes:
      - numeric {0,1,2} where 0=neg,1=neu,2=pos -> keeps only 0 and 2
      - numeric {0,4} (sentiment140) -> 0 negative, 4 positive
      - string labels -> filters for 'negative'/'positive'

    Returns (df, removed): df has columns `text` and `label` mapped to 0/1, removed = rows dropped.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available. Install with `pip install datasets`")

    if subset:
        ds = load_dataset(dataset_id, subset)
    else:
        ds = load_dataset(dataset_id)

    # try to find the split that contains text and label
    # prefer 'train' split
    if isinstance(ds, dict):
        if "train" in ds:
            df = ds["train"].to_pandas()
        else:
            # take first split
            df = next(iter(ds.values())).to_pandas()
    else:
        df = ds.to_pandas()

    # normalize column names
    if "text" not in df.columns:
        # try common alternatives
        for col in ["tweet", "content", "sentence", "text_raw"]:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break

    if "text" not in df.columns or "label" not in df.columns:
        # try label alternatives
        for col in ["label", "labels", "sentiment"]:
            if col in df.columns:
                df = df.rename(columns={col: "label"})
                break

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Could not find 'text' and 'label' columns in the HuggingFace dataset")

    original_len = len(df)

    # handle numeric labels
    if pd.api.types.is_integer_dtype(df["label"]):
        unique = sorted(df["label"].unique())
        # common: 0/4 -> map 4->1, 0->0
        if set(unique) <= {0, 4}:
            df = df[df["label"].isin([0, 4])].copy()
            df["label"] = df["label"].map({0: 0, 4: 1})
        # common: 0/1/2 -> keep 0 and 2
        elif set(unique) & {0, 1, 2}:
            df = df[df["label"].isin([0, 2])].copy()
            df["label"] = df["label"].map({0: 0, 2: 1})
        else:
            # fallback: drop anything that's not 0 or 1
            df = df[df["label"].isin([0, 1])].copy()
    else:
        # string labels: filter for positive/negative
        df = df[df["label"].isin(["positive", "negative"])].copy()
        df["label"] = df["label"].map({"negative": 0, "positive": 1})

    removed = original_len - len(df)

    # ensure only required columns
    df = df[["text", "label"]].copy()

    return df, removed
