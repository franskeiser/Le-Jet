import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imblearn")
pytest.importorskip("nltk")
pytest.importorskip("urlextract")

from src.unified import voting_baseline as vb  # noqa: E402


def test_strip_links_removes_the_link_only():
    out = vb.strip_links("BDO: verify now at http://bdo-secure.xyz/login today")
    assert "bdo-secure" not in out and "verify now at" in out and out.endswith("today")


def _rows(n):
    spam = ["urgent your account is locked verify now {u}", "you won a prize claim now {u}", "payment failed confirm details {u}"]
    ham = ["see you at lunch tomorrow {u}", "your parcel arrives today thanks {u}", "meeting moved to noon {u}"]
    texts, labels = [], []
    for i in range(n):
        texts.append((spam if i % 2 else ham)[i % 3].format(u=f"http://site{i}.com/a") + f" ref{i}")
        labels.append("smishing" if i % 2 else "legitimate")
    return pd.DataFrame({"sms_text": texts, "label": labels})


def test_fit_and_score_shapes_and_ranges_and_is_deterministic():
    train, test = _rows(60), _rows(10)
    p1, pred1 = vb.fit_and_score(train, test, seed=42)
    p2, pred2 = vb.fit_and_score(train, test, seed=42)
    assert p1.shape == pred1.shape == (10,) and set(np.unique(pred1)) <= {0, 1}
    assert ((p1 >= 0) & (p1 <= 1)).all() and np.array_equal(pred1, pred2) and np.array_equal(p1, p2)
    assert ((p1 >= 0.5).astype(int) == pred1).all()      # the majority of three voters is the 0.5 cut on their share
