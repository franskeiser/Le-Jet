"""Tests for the KNN-RF-ETC baseline (Jain et al. 2022) with its URL routing: src/unified/voting_baseline.py."""
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


def _rows(n, start=0):
    """Every third message has no link; the links of smishing messages look different from legitimate ones."""
    spam = ["urgent your account is locked verify now", "you won a prize claim now", "payment failed confirm details"]
    ham = ["see you at lunch tomorrow", "your parcel arrives today thanks", "meeting moved to noon"]
    rows = []
    for i in range(start, start + n):
        smish = i % 2 == 1
        linked = i % 3 != 0
        link = (f"http://secure-login-verify{i}.xyz/account" if smish else f"https://www.shop{i}.com/orders") if linked else ""
        text = (spam if smish else ham)[i % 3] + (f" {link}" if linked else "") + f" ref{i}"
        rows.append({"message_id": f"m{i}", "sms_text": text, "label": "smishing" if smish else "legitimate",
                     "has_url": int(linked), "link": link if linked else "-1"})
    return pd.DataFrame(rows)


def test_fit_and_score_shapes_and_ranges_and_is_deterministic():
    train, test = _rows(90), _rows(12, start=500)
    p1, pred1 = vb.fit_and_score(train, test, seed=42)
    p2, pred2 = vb.fit_and_score(train, test, seed=42)
    assert p1.shape == pred1.shape == (12,) and set(np.unique(pred1)) <= {0, 1}
    assert ((p1 >= 0) & (p1 <= 1)).all() and np.array_equal(pred1, pred2) and np.array_equal(p1, p2)
    assert ((p1 > 0.5).astype(int) == pred1).all()     # LR argmax and the majority of three voters are both a cut at 0.5


def test_messages_are_routed_by_has_url_like_the_paper():
    detector = vb.fit_detector(_rows(90), seed=42)
    test = _rows(12, start=500)
    p, pred, route = vb.score(detector, test)
    linked = test["has_url"].to_numpy() == 1
    assert list(route) == ["url" if x else "text" for x in linked]
    assert set(np.round(p[~linked] * 3, 6)) <= {0, 1, 2, 3}                 # text route: share of three voters
    U = detector.url_vec.transform([vb.transform_url(u) for u in test.loc[linked, "link"]])
    assert np.allclose(p[linked], detector.url_clf.predict_proba(U)[:, 1])  # URL route: Logistic Regression on the link


def test_a_linked_message_is_judged_on_its_link_only_and_a_no_link_message_on_its_words():
    detector = vb.fit_detector(_rows(90), seed=42)
    test = _rows(12, start=500)
    reworded = test.copy()
    reworded["sms_text"] = "completely different words here"
    p, _, route = vb.score(detector, test)
    q, _, _ = vb.score(detector, reworded)
    url = route == "url"
    assert np.array_equal(p[url], q[url])               # the text is not read on the URL route (as in the paper)


def test_both_classifiers_are_fitted_on_the_training_rows_only():
    train = _rows(90)
    detector = vb.fit_detector(train, seed=42)
    assert detector.url_info["n_unique_urls"] == len(set(zip(train.loc[train.has_url == 1, "link"], train.loc[train.has_url == 1, "label"])))
    assert detector.text_info["n_rows_after_cleaning"] <= len(train)
    assert all("xyz" not in w and "http" not in w for w in detector.tfidf.vocabulary_)   # links are cut out of the text classifier's words
