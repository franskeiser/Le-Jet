"""
Tests for src/data/split_crowd.py.

Run with:  python -m pytest tests/test_split_crowd.py
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import split_crowd as sc


def _frame(n_domains=40, per_domain=3, chain_templates=False):
    rows = []
    for d in range(n_domains):
        for i in range(per_domain):
            # chain_templates: domain d shares a template with domain d+1, so everything joins up.
            template = f"t{d + i % 2}" if chain_templates else f"t{d}_{i}"
            rows.append({"message_id": f"pilot_{len(rows):06d}", "sms_text": f"BDO notice {i} https://d{d}.xyz",
                         "label": "smishing" if d % 2 else "legitimate",
                         "link_domain": f"d{d}.xyz", "template_id": template})
    return pd.DataFrame(rows)


def test_joined_groups_link_domain_and_template():
    df = pd.DataFrame({"link_domain": ["a", "b", "c", "d"], "template_id": ["t1", "t1", "t2", "t3"]})
    g = sc.joined_groups(df)
    assert g[0] == g[1] and len({g[0], g[2], g[3]}) == 3


def test_strict_rule_refuses_one_giant_group():
    with pytest.raises(ValueError, match="more than one fold"):
        sc.make_groups(_frame(chain_templates=True), "domain+template")


@pytest.mark.parametrize("rule", sc.GROUP_RULES)
def test_run_writes_disjoint_folds(tmp_path, rule):
    src = tmp_path / "crowd_pilot_synthetic.csv"
    _frame().to_csv(src, index=False)
    report, files = sc.run(src, tmp_path / "out", rule, seeds=[42, 123], n_folds=5)
    assert len(files) == 10 and len(report["folds"]) == 10
    for seed in (42, 123):
        tested = pd.concat([files[(seed, f)] for f in range(5)]).query("part == 'test'")
        assert tested["message_id"].is_unique and len(tested) == 120
    df = _frame()
    for (seed, fold), parts in files.items():
        merged = df.merge(parts, on="message_id")
        domains = merged.groupby("link_domain")["part"].nunique()
        assert (domains == 1).all(), "a domain is in two parts"
        assert set(merged["part"]) == {"train", "val", "test"}
    assert (tmp_path / "out" / "seed42_fold0.csv").exists()
    saved = json.loads((tmp_path / "out" / "split_report.json").read_text())
    assert saved["group_by"] == rule and saved["folds"][0]["test_rows_with_train_domain"] == 0


def test_seq_len_is_95th_percentile():
    texts = pd.Series(["one two three"] * 19 + [" ".join(["w"] * 40)])
    # Sorted, the 95th percentile sits 5% of the way from 3 to 40: 4.85, rounded up.
    assert sc.seq_len(texts) == 5


def test_no_link_messages_are_not_one_group():
    """A message with no link has link_domain -1; the 'domain -1' must not tie all of them into one giant group."""
    df = pd.DataFrame({"message_id": [f"m{i}" for i in range(6)], "link_domain": ["-1"] * 4 + ["a", "a"],
                       "template_id": ["t1", "t2", "t3", "t4", "t5", "t6"]})
    g = sc.joined_groups(df)
    assert g.iloc[:4].nunique() == 4          # four no-link messages, four groups
    assert g.iloc[4] == g.iloc[5]             # the same domain still joins
    assert sc.make_groups(df, "domain").iloc[:4].nunique() == 4


def test_no_link_messages_with_one_template_stay_together():
    df = pd.DataFrame({"message_id": ["m0", "m1", "m2"], "link_domain": ["-1"] * 3, "template_id": ["t1", "t1", "t2"]})
    g = sc.joined_groups(df)
    assert g.iloc[0] == g.iloc[1] and g.iloc[0] != g.iloc[2]


def test_dup_group_joins_rows_with_different_domain_and_template():
    df = pd.DataFrame({"message_id": ["m0", "m1", "m2"], "link_domain": ["a", "b", "c"], "template_id": ["t1", "t2", "t3"],
                       "dup_group": ["g1", "g1", "g2"]})
    g = sc.joined_groups(df)
    assert g.iloc[0] == g.iloc[1] and g.iloc[0] != g.iloc[2]
    without = sc.joined_groups(df.drop(columns="dup_group"))
    assert without.nunique() == 3             # the column is what joins them
