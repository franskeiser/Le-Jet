"""has_url through the E2 pipeline: the link lane, the inputs, the models and the app path (docs/film_architecture_plan.md, the diagram's
"has_url = 1?" branch). A message with no link must never reach the Link RF, must get the plain text CNN's numbers from E2, and must be
told apart by the concatenation control."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from src.features.model.concat_cnn import ConcatTextCNN
from src.features.model.film_cnn import FiLMTextCNN
from src.unified import concat_control as cc
from src.unified import film
from src.unified import film_inputs as fi
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37

VOCAB = {"<PAD>": 0, "verify": 1, "your": 2, "account": 3, "parcel": 4, "<URL>": 5}


# ------------------------------------------------------------------------------------------------------ link lane
def lane_rows(n=40, with_url=None):
    rng = np.random.default_rng(1)
    rows = pd.DataFrame(rng.random((n, len(FEATURES_37))), columns=FEATURES_37)
    rows[ll.PAGE] = rows[ll.PAGE] * 10
    rows["page_fetched"] = 1
    rows["label"] = np.where(rows.iloc[:, 0] + rows[ll.PAGE[0]] / 10 > 1, "smishing", "legitimate")
    rows["message_id"] = [f"m{i}" for i in range(n)]
    if with_url is not None:
        rows["has_url"] = with_url
        rows.loc[rows["has_url"] == 0, FEATURES_37] = -1
        rows.loc[rows["has_url"] == 0, "page_fetched"] = 0
    return rows


def test_has_link_reads_the_column_and_a_dataset_without_it_has_a_link_everywhere():
    assert ll.has_link(lane_rows(4)).all()
    assert list(ll.has_link(lane_rows(4, with_url=[1, 0, 1, 0]))) == [True, False, True, False]


def test_lane_numbers_give_zeros_to_a_message_with_no_link_and_score_the_rest():
    rows = lane_rows(40, with_url=[1, 1, 1, 0] * 10)
    linked = ll.has_link(rows)
    rf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(rows[linked][FEATURES_37], (rows[linked]["label"] == "smishing"))
    out = ll.lane_numbers(rf, rows)
    assert list(out.columns) == ll.LANE and len(out) == 40
    none = out[~linked.to_numpy()]
    assert (none[ll.OUTPUTS] == 0).all().all() and (none["page_missing"] == 1).all()          # the forest speaks only for linked messages
    assert (out[linked.to_numpy()]["P"] > 0).any() and (out[linked.to_numpy()]["u"] >= 0).all()
    assert np.allclose(out[linked.to_numpy()]["P"], rf.predict_proba(rows[linked][FEATURES_37])[:, 1])


def test_the_oof_numbers_cover_every_row_and_never_train_a_forest_on_a_no_link_message(monkeypatch):
    rows = lane_rows(60, with_url=[1, 1, 1, 0] * 15)
    rows["domain"] = [f"d{i}" for i in range(60)]
    seen = []
    real_fit = ll.fit_rf

    def spy(X, y, seed, **params):
        seen.append(X.copy())
        return real_fit(X, y, seed, **params)
    monkeypatch.setattr(ll, "fit_rf", spy)
    monkeypatch.setattr(ll, "make_groups", lambda df, rule, n: pd.Series(df["domain"].to_numpy(), index=df.index))
    full, blank = ll.oof_outputs(rows, "domain", 0, {"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}, n_splits=3)
    assert len(full) == len(blank) == 60 and full.index.equals(rows.index)
    assert all((X != -1).all().all() for X in seen), "a no-link message (every feature -1) reached a forest"
    no_link = ~ll.has_link(rows).to_numpy()
    assert (full[no_link][ll.OUTPUTS] == 0).all().all() and (blank[no_link][ll.OUTPUTS] == 0).all().all()


# ------------------------------------------------------------------------------------------------------- inputs
def test_align_carries_has_url_and_stack4_puts_it_last():
    rows = pd.DataFrame({"message_id": ["a", "b", "c"], "label": ["smishing", "legitimate", "legitimate"],
                         "page_fetched": [1, 0, 0], "has_url": [1, 1, 0]})
    lane = pd.DataFrame({"message_id": ["a", "b", "c"], "P": [0.9, 0.2, 0.0], "u": [0.1, 0.2, 0.0], "page_missing": [0, 1, 1]})
    blank = lane.assign(page_missing=1)
    part = fi.align(rows, lane, blank)
    assert list(part.has_url) == [1, 1, 0]
    four = part.stack4("full")
    assert four.shape == (3, fi.LINK_WIDTH) and np.allclose(four[:, 3], [1, 1, 0]) and np.allclose(four[:, :3], part.stack("full"))
    assert part.stack("full").shape == (3, 3)                                                  # the three-number view is unchanged


def test_a_dataset_without_has_url_has_a_link_in_every_message():
    rows = pd.DataFrame({"message_id": ["a", "b"], "label": ["smishing", "legitimate"], "page_fetched": [1, 1]})
    lane = pd.DataFrame({"message_id": ["a", "b"], "P": [0.9, 0.2], "u": [0.1, 0.2], "page_missing": [0, 0]})
    assert list(fi.align(rows, lane, lane.assign(page_missing=1)).stack4("full")[:, 3]) == [1, 1]


# ------------------------------------------------------------------------------------------------------ models
def embedding():
    torch.manual_seed(0)
    return torch.nn.Embedding(len(VOCAB), 300, padding_idx=0)


def batch(n=8):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, len(VOCAB), (n, 20), generator=g)
    link = torch.cat([torch.rand(n, 3, generator=g), torch.tensor([1, 0, 1, 0, 1, 0, 1, 0.])[:n, None]], dim=1)
    return ids, link


def active(model):
    with torch.no_grad():
        model.generator.weight.copy_(torch.randn_like(model.generator.weight))
        model.generator.bias.copy_(torch.randn_like(model.generator.bias))
    return model.eval()


def test_film_reads_has_url_from_the_last_column_of_a_width_four_link():
    model = active(FiLMTextCNN(embedding()))
    ids, link = batch()
    keep = link[:, 3].bool()
    gamma, beta = model.film_parameters(link, model.conv_features(ids)[1])
    assert (gamma[~keep] == 1).all() and (beta[~keep] == 0).all()
    assert not (gamma[keep] == 1).all()
    assert torch.equal(model(ids, link), model(ids, link[:, :3], has_url=link[:, 3]))          # the column and the argument agree
    assert torch.equal(model(ids[keep], link[keep][:, :3]), model(ids[keep], link[keep]))      # a width-3 link still means "has a link"


def test_a_width_four_link_with_no_link_gives_the_plain_cnn_numbers():
    model = active(FiLMTextCNN(embedding()))
    plain = FiLMTextCNN(embedding()).eval()                                                      # generator at zero
    plain.conv.load_state_dict(model.conv.state_dict())
    ids, link = batch()
    no_link = link.clone()
    no_link[:, 3] = 0
    assert torch.equal(model.encode(ids, no_link), plain.encode(ids, no_link))


def test_without_the_bypass_has_url_is_ignored_and_the_generator_reads_every_message():
    on, off = active(FiLMTextCNN(embedding())), FiLMTextCNN(embedding(), bypass_no_link=False).eval()
    off.load_state_dict(on.state_dict())
    ids, link = batch()
    keep = link[:, 3].bool()
    gamma, _ = off.film_parameters(link, off.conv_features(ids)[1])
    assert not (gamma[~keep] == 1).all()                                                         # no-link messages are FiLM-adjusted too
    assert torch.equal(off(ids, link), off(ids, link[:, :3]))                                   # the fourth column changes nothing
    assert not torch.equal(on(ids, link), off(ids, link))


def test_a_bad_width_is_refused_and_the_message_says_what_is_accepted():
    model = FiLMTextCNN(embedding())
    ids, link = batch()
    with pytest.raises(ValueError, match="has_url last"):
        model(ids, torch.cat([link, link[:, :1]], dim=1))                                      # width 5


def test_the_concatenation_control_is_built_with_four_inputs_and_has_the_extra_parameters():
    direct = ConcatTextCNN(embedding(), link_dim=fi.LINK_WIDTH)
    assert direct.dense.in_features == 64 + 4
    count = sum(p.numel() for p in direct.parameters() if p.requires_grad and p is not direct.embedding.weight)
    assert count == cc.EXPECTED_TRAINABLE == cc.expected_trainable(4) == 62210
    ids, link = batch()
    assert tuple(direct.eval()(ids, link).shape) == (8, 2)
    with pytest.raises(ValueError, match="link must be"):
        direct(ids, link[:, :3])


def test_the_control_can_tell_a_no_link_message_from_a_link_one():
    model = ConcatTextCNN(embedding(), link_dim=fi.LINK_WIDTH).eval()
    ids, link = batch()
    other = link.clone()
    other[:, 3] = 1 - link[:, 3]
    assert not torch.allclose(model(ids, link), model(ids, other))


# ------------------------------------------------------------------------------------------------------- app path
@pytest.fixture(scope="module")
def app():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.random((60, len(FEATURES_37))), columns=FEATURES_37)
    features[ll.PAGE] = features[ll.PAGE] * 10
    labels = (features.iloc[:, 0] + features[ll.PAGE[0]] / 10 > 1).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(features, labels)
    rows = features.iloc[:12].copy()
    rows.insert(0, "sms_text", ["verify your account <URL>", "your parcel <URL>", "your parcel"] * 4)
    rows["page_fetched"] = 1
    rows["has_url"] = [1, 1, 0] * 4
    rows.loc[rows["has_url"] == 0, FEATURES_37] = np.nan                    # the forest would crash on these: it must never see them
    rows.loc[rows["has_url"] == 0, "page_fetched"] = 0
    torch.manual_seed(0)
    model = active(FiLMTextCNN(torch.nn.Embedding(len(VOCAB), 300, padding_idx=0)))
    return model, rf, rows


def test_predict_messages_never_sends_a_no_link_message_to_the_forest(app):
    model, rf, rows = app
    out = film.predict_messages(model, rf, VOCAB, 12, rows)
    none = rows["has_url"].to_numpy() == 0
    assert list(out["has_url"]) == list(rows["has_url"]) and (out.loc[none, ["P", "u"]] == 0).all().all()
    assert (out.loc[none, "page_missing"] == 1).all() and np.isfinite(out["p_smishing"]).all()
    assert list(out.columns) == ["P", "u", "page_missing", "has_url", "p_smishing", "pred"]


def test_a_no_link_verdict_is_the_plain_text_cnns_and_ignores_whatever_the_features_hold(app):
    model, rf, rows = app
    none = rows[rows["has_url"] == 0]
    again = none.copy()
    again[FEATURES_37] = 7.0                                                # different junk, same message: the verdict must not move
    a = film.predict_messages(model, rf, VOCAB, 12, none)["p_smishing"].to_numpy()
    b = film.predict_messages(model, rf, VOCAB, 12, again)["p_smishing"].to_numpy()
    assert np.array_equal(a, b)
    forced = none.assign(has_url=1)
    forced[FEATURES_37] = 0.5
    c = film.predict_messages(model, rf, VOCAB, 12, forced)["p_smishing"].to_numpy()
    assert not np.allclose(a, c)                                            # a link WOULD have changed it: the bypass is doing the work


def test_predict_messages_without_the_column_means_every_message_has_a_link(app):
    model, rf, rows = app
    linked = rows[rows["has_url"] == 1].drop(columns="has_url")
    out = film.predict_messages(model, rf, VOCAB, 12, linked)
    assert (out["has_url"] == 1).all() and (out["u"] >= 0).all() and (out["P"] > 0).any()
