"""Tests for src/features/model/plain_cnn.py, the text-only baseline (plan docs/film_architecture_plan.md, step 3.6).

Run with:  python -m pytest tests/test_plain_cnn.py
"""
import pytest
import torch
import torch.nn.functional as F

from src.features.model.film_cnn import FiLMTextCNN
from src.features.model.plain_cnn import PlainTextCNN

FRESH = 100


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(FRESH, 300, padding_idx=0)


def batch(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, FRESH, (n, 50), generator=g), torch.rand(n, 3, generator=g)


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p is not model.embedding.weight)


def test_the_baseline_has_61954_trainable_parameters_without_the_embedding():
    model = PlainTextCNN(embedding())
    assert trainable(model) == 61954
    parts = {n: sum(p.numel() for p in layer.parameters()) for n, layer in
             (("conv", model.conv), ("dense", model.dense), ("output", model.output))}
    assert parts == {"conv": 57664, "dense": 4160, "output": 130}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 61954 + FRESH * 300     # the embedding is trainable too


def test_it_is_8704_parameters_smaller_than_e2():
    assert trainable(FiLMTextCNN(embedding())) - trainable(PlainTextCNN(embedding())) == 8704


def test_forward_gives_logits_and_ignores_the_link_and_text_keep():
    model = PlainTextCNN(embedding()).eval()
    ids, link = batch()
    with torch.no_grad():
        logits = model(ids)
        assert tuple(logits.shape) == (8, 2)
        assert torch.equal(logits, model(ids, link)) and torch.equal(logits, model(ids, torch.rand_like(link)))
        assert torch.equal(logits, model(ids, link, text_keep=torch.zeros(8)))


def test_e2_with_its_generator_at_zero_is_exactly_this_model():
    """The fair start: copy the baseline's weights into E2 and the two give the same numbers, bit for bit."""
    baseline = PlainTextCNN(embedding()).eval()
    model = FiLMTextCNN(embedding(seed=5)).eval()
    state = {k: v for k, v in baseline.state_dict().items()}
    result = model.load_state_dict(state, strict=False)
    assert result.unexpected_keys == [] and {k.split(".")[0] for k in result.missing_keys} == {"generator"}
    ids, link = batch()
    with torch.no_grad():
        assert torch.equal(model.encode(ids, link), baseline.encode(ids))
        assert torch.equal(model(ids, link), baseline(ids))


def test_the_embedding_can_be_frozen_or_trained():
    trained = PlainTextCNN(embedding())
    frozen = PlainTextCNN(embedding(), freeze_embedding=True)
    assert trained.embedding.weight.requires_grad and not frozen.embedding.weight.requires_grad
    before = frozen.embedding.weight.clone()
    ids, _ = batch()
    optimizer = torch.optim.Adam(frozen.parameters(), lr=1e-2)
    F.cross_entropy(frozen(ids), torch.tensor([0, 1] * 4)).backward()
    optimizer.step()
    assert torch.equal(frozen.embedding.weight, before)
    emb_before = trained.embedding.weight.clone()
    optimizer = torch.optim.Adam(trained.parameters(), lr=1e-2)
    F.cross_entropy(trained(ids), torch.tensor([0, 1] * 4)).backward()
    optimizer.step()
    assert not torch.equal(trained.embedding.weight, emb_before)


@pytest.mark.parametrize("freeze", [False, True])
def test_parameter_groups_cover_every_trainable_parameter_once(freeze):
    model = PlainTextCNN(embedding(), freeze_embedding=freeze)
    groups = model.parameter_groups(1e-4)
    listed = [p for g in groups for p in g["params"]]
    assert len(listed) == len({id(p) for p in listed}) and {id(p) for p in listed} == {id(p) for p in model.parameters() if p.requires_grad}
    decay = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert decay[id(model.dense.weight)] == 1e-4 and decay[id(model.conv.weight)] == 0.0 and decay[id(model.output.weight)] == 0.0
    assert any(p is model.embedding.weight for p in listed) is (not freeze)
    torch.optim.Adam(groups, lr=1e-3)
    assert model.parameter_groups(1e-4, generator_weight_decay=5e-4)[0]["weight_decay"] == 1e-4     # accepted and ignored
