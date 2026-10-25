"""Load the WANDS dataset (github.com/wayfair/WANDS, `dataset/` folder).

Files are tab-separated:
  product.csv: product_id, product_name, product_class, ..., product_description, product_features, ...
  query.csv:   query_id, query, query_class
  label.csv:   id, query_id, product_id, label   (Exact | Partial | Irrelevant)
"""
from pathlib import Path
import pandas as pd

GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def load_wands(path: str):
    p = Path(path)
    products = pd.read_csv(p / "product.csv", sep="\t")
    queries = pd.read_csv(p / "query.csv", sep="\t")
    labels = pd.read_csv(p / "label.csv", sep="\t")
    labels["gain"] = labels["label"].map(GAIN).fillna(0).astype(int)
    return products, queries, labels


def build_documents(products: pd.DataFrame, max_desc_chars: int = 600) -> list[str]:
    """One text per product: name + class + (truncated) description + features."""
    def col(name):
        if name not in products:
            return pd.Series([""] * len(products), index=products.index)
        return products[name].fillna("").astype(str)

    docs = (
        col("product_name") + ". "
        + col("product_class") + ". "
        + col("product_description").str.slice(0, max_desc_chars) + ". "
        + col("product_features").str.slice(0, 300)
    )
    return docs.tolist()


def relevance_by_query(labels: pd.DataFrame) -> dict[int, dict[int, int]]:
    """{query_id: {product_id: gain}} for judged pairs. Unjudged pairs count as 0."""
    out: dict[int, dict[int, int]] = {}
    for qid, pid, gain in labels[["query_id", "product_id", "gain"]].itertuples(index=False):
        out.setdefault(int(qid), {})[int(pid)] = int(gain)
    return out
