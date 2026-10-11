"""The Random Forest as a feature extractor: a verdict AND a per-feature breakdown of it.

For one link, the forest returns
    P   mean phishing probability over the trees
    u   spread of the trees' probabilities
    c_<feature>   how many probability points each input feature added to or took from P

The c_ values come from the trees themselves (the Saabas path decomposition): walking a tree from
the root to a leaf, every split changes the node's phishing fraction; that change is credited to the
feature that was split on. Summed over a tree's path and averaged over the trees,

    P = bias + sum_f c_f          (bias = the phishing fraction at the roots, averaged)

holds exactly (tests/test_rf_feature_extractor.py). Properties that matter for the head:
  * The URL-spelling features and the page / domain features sit in the SAME forest, so they
    share one verdict and one breakdown.
  * c_ values are in probability points (between -1 and 1), already on the same scale as P and u.
  * A feature the forest never saw, or never split on for this link, contributes exactly 0
    ("no evidence"), so the url_only forest (14 inputs) and the full forest (31 inputs) fill the same
    31-slot vector and the page slots are simply 0 when no page was reached.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.features.model.rf_extractor import VARIANTS, p_and_u

FULL_FEATURES = VARIANTS["full"]
CONTRIBUTION_COLUMNS = [f"c_{f}" for f in FULL_FEATURES]
OUTPUT_COLUMNS = ["P", "u"] + CONTRIBUTION_COLUMNS


def tree_contributions(tree, values: np.ndarray, n_features: int) -> tuple[np.ndarray, float]:
    """(n_rows, n_features) contributions of one fitted decision tree, and its root phishing fraction."""
    t = tree.tree_
    counts = t.value[:, 0, :]
    p1 = counts[:, 1] / counts.sum(axis=1)  # phishing fraction at every node
    internal = np.flatnonzero(t.children_left != -1)
    children = np.concatenate([t.children_left[internal], t.children_right[internal]])
    parents = np.concatenate([internal, internal])
    gain = csr_matrix((p1[children] - p1[parents], (children, t.feature[parents])),
                      shape=(t.node_count, n_features))
    return (tree.decision_path(values) @ gain).toarray(), float(p1[0])


def path_contributions(rf, values: np.ndarray) -> tuple[np.ndarray, float]:
    """Forest-level contributions: mean over trees. Returns (contributions, bias)."""
    total = np.zeros((len(values), values.shape[1]))
    bias = 0.0
    for tree in rf.estimators_:
        contribution, root = tree_contributions(tree, values, values.shape[1])
        total += contribution
        bias += root
    n_trees = len(rf.estimators_)
    return total / n_trees, bias / n_trees


def extract_vector(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """P, u and the 31 contributions (FULL_FEATURES order) for each row, from one extractor.

    `features` holds the extractor's input columns plus has_url. Rows with no link, or with every
    input at -1 (nothing known), get P = u = -1 and all contributions 0, like extract_p_u."""
    out = pd.DataFrame(0.0, index=features.index, columns=OUTPUT_COLUMNS)
    out[["P", "u"]] = -1.0
    unknown = (features[bundle["features"]] == -1).all(axis=1)
    scored = ((features["has_url"] == 1) & ~unknown).to_numpy()
    if scored.any():
        rows = features.loc[scored, bundle["features"]]
        p, u = p_and_u(bundle["model"], rows)
        contribution, _ = path_contributions(bundle["model"], rows.to_numpy(dtype=np.float32))
        out.loc[scored, "P"], out.loc[scored, "u"] = p, u
        for i, name in enumerate(bundle["features"]):
            out.loc[scored, f"c_{name}"] = contribution[:, i]
    return out


def route_vector(url_only: dict, full: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Like rf_extractor.route_p_u: the full forest where the page was reached (page_fetched == 1),
    the url_only forest everywhere else."""
    fetched = ((features["has_url"] == 1) & (features["page_fetched"] == 1)).to_numpy()
    out = extract_vector(url_only, features)
    if fetched.any():
        out.loc[fetched] = extract_vector(full, features.loc[fetched]).to_numpy()
    return out


def extractor_gates(importance: pd.Series) -> np.ndarray:
    """Gates (0.1 to 3.0) for the 31 contribution slots from a forest's permutation importances,
    indexed by feature name. Same min-max rule as the CNN and link-feature gates."""
    from src.features.model.rf_gating_module import gate_transform

    return gate_transform(importance.reindex(FULL_FEATURES).to_numpy(dtype=float)).astype(np.float32)
