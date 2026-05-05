import pandas as pd
import os


def load_and_prepare_data(data_path, valid_labels=("positive", "negative"), label_mapping=None):
    """Load CSV, remove unnamed columns, filter valid labels and map them.

    Returns:
        data (pd.DataFrame): cleaned dataframe with `text` and `label` (mapped to ints if mapping provided)
        removed (int): number of rows removed due to invalid labels
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    data = pd.read_csv(data_path)
    # drop typical unnamed index cols produced by pandas csv exports
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Validate required columns
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain columns: 'text' and 'label'")

    original_len = len(data)
    data = data[data["label"].isin(valid_labels)].copy()
    removed = original_len - len(data)

    if label_mapping is not None:
        data["label"] = data["label"].map(label_mapping)

    return data, removed
