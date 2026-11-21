"""Tests for src/unified/film_inputs.py (plan docs/film_architecture_plan.md, step 1.2).

Run with:  python -m pytest tests/test_film_inputs.py
"""
import numpy as np
import pandas as pd
import pytest

from src.unified import film_inputs as fi


def rows(n=6):
    """Text rows of one part: ids, labels and whether the page was fetched."""
    return pd.DataFrame({"message_id": [f"m{i}" for i in range(n)],
                         "label": ["smishing", "legitimate"] * (n // 2),
                         "page_fetched": [1, 0, 1, 1, 0, 1][:n]})


def lane(text_rows, blank=False, p=0.7):
    frame = pd.DataFrame({"message_id": text_rows["message_id"], "P": p, "u": 0.1,
                          "page_missing": 1 if blank else 1 - text_rows["page_fetched"]})
    frame["c_length_url"] = 0.0   # the lane also carries the 37 contributions; E2 ignores them
    return frame


def test_align_gives_one_array_per_number_in_text_row_order():
    text = rows()
    shuffled = lane(text).sample(frac=1, random_state=1).reset_index(drop=True)   # the lane's order may differ
    part = fi.align(text, shuffled, lane(text, blank=True))
    assert list(part.message_id) == list(text["message_id"])
    for numbers in (part.full, part.blank):
        assert set(numbers) == {"P", "u", "page_missing"}
        assert all(a.shape == (6,) for a in numbers.values())
    assert list(part.label) == [1, 0, 1, 0, 1, 0]
    assert part.stack("full").shape == (6, 3) and part.stack("full").dtype == np.float32
    assert (part.stack("blank")[:, 2] == 1).all()
    assert list(part.full["page_missing"]) == [0, 1, 0, 0, 1, 0]


def test_align_keeps_each_message_with_its_own_numbers():
    text = rows()
    full = lane(text)
    full["P"] = np.linspace(0.1, 0.6, 6)
    part = fi.align(text, full.sample(frac=1, random_state=3).reset_index(drop=True), lane(text, blank=True))
    assert np.allclose(part.full["P"], np.linspace(0.1, 0.6, 6))


def test_align_stops_on_a_missing_or_extra_or_duplicate_row():
    text = rows()
    with pytest.raises(ValueError, match="one to one"):
        fi.align(text, lane(text).iloc[:-1], lane(text, blank=True))                        # a message has no numbers
    extra = pd.concat([lane(text), lane(text).iloc[[0]].assign(message_id="m99")], ignore_index=True)
    with pytest.raises(ValueError, match="one to one"):
        fi.align(text, extra, lane(text, blank=True))                                        # a number has no message
    duplicated = pd.concat([lane(text).iloc[:-1], lane(text).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one to one"):
        fi.align(text, duplicated, lane(text, blank=True))
    with pytest.raises(ValueError, match="twice"):
        fi.align(pd.concat([text, text.iloc[[0]]], ignore_index=True), lane(text), lane(text, blank=True))


@pytest.mark.parametrize("column,value,message", [("P", 1.2, "outside"), ("P", -0.1, "outside"), ("u", -0.2, "negative"),
                                                  ("P", np.nan, "finite"), ("u", np.inf, "finite")])
def test_align_stops_on_bad_numbers(column, value, message):
    text = rows()
    full = lane(text)
    full.loc[2, column] = value
    with pytest.raises(ValueError, match=message):
        fi.align(text, full, lane(text, blank=True))


def test_align_checks_page_missing_against_the_dataset():
    text = rows()
    wrong = lane(text)
    wrong.loc[0, "page_missing"] = 1                              # the dataset says this page was fetched
    with pytest.raises(ValueError, match="page_fetched"):
        fi.align(text, wrong, lane(text, blank=True))
    not_blank = lane(text, blank=True)
    not_blank.loc[3, "page_missing"] = 0
    with pytest.raises(ValueError, match="blank"):
        fi.align(text, lane(text), not_blank)


def test_summary_rows_describe_each_part():
    text = rows()
    part = fi.align(text, lane(text), lane(text, blank=True))
    out = fi.summary_rows(42, 0, {"train": part})
    assert out[0]["n"] == 6 and out[0]["page_missing"] == round(2 / 6, 3) and out[0]["part"] == "train"


def test_pilot_files_align_for_one_cell():
    """Only where the (git-ignored) synthetic pilot files exist; skipped on a fresh checkout."""
    from src.config import CROWD_PILOT_PATH
    from src.unified.film_inputs import ll

    root = ll.ROOT
    if not (root / CROWD_PILOT_PATH).exists() or not (ll.out_dir(root / CROWD_PILOT_PATH) / "oof_seed42_fold0.csv").exists():
        pytest.skip("the synthetic pilot files are not on this machine")
    inputs = fi.link_inputs(42, 0, pilot=True)
    assert set(inputs) == {"train", "val", "test"}
    for part in inputs.values():
        n = len(part.message_id)
        assert n > 0 and all(a.shape == (n,) for a in part.full.values()) and all(a.shape == (n,) for a in part.blank.values())
