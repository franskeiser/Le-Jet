"""Tests for src/features/model/film_cnn.py (plan docs/film_architecture_plan.md, step 2.3).

The tests use a small random embedding, so they run in seconds. One test at the end uses a real trained text CNN and the
(git-ignored) synthetic pilot files, and is skipped where they are missing.

Not tested: item (f) of step 2.3, "gamma after max-pool equals gamma before ReLU", belongs to variant V9, which is not built.

Run with:  python -m pytest tests/test_film.py
"""
import pytest
import torch
import torch.nn.functional as F

from src.features.model.cnn_encoder import TextEncoder
from src.features.model.film_cnn import FiLMTextCNN

FRESH = 100        # vocabulary size of the small test embedding
FILTERS = 64


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(FRESH, 300, padding_idx=0)


def batch(n=8, seed=1, length=50):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, FRESH, (n, length), generator=g), torch.rand(n, 3, generator=g)


def trainable(model):
    """Trainable parameters excluding the embedding: the plan's 70,658 and 66,306 do not include it (D6)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p is not model.embedding.weight)


def randomize_generator(model, std=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.generator.weight.copy_(torch.randn(model.generator.weight.shape, generator=g) * std)
        model.generator.bias.copy_(torch.randn(model.generator.bias.shape, generator=g) * std)


# ------------------------------------------------------------------ shapes
def test_forward_gives_logits_for_a_batch_and_stops_on_a_wrong_link_shape():
    model = FiLMTextCNN(embedding())
    ids, link = batch()
    assert tuple(model(ids, link).shape) == (8, 2)
    x, average = model.conv_features(ids)
    assert tuple(x.shape) == (8, FILTERS, 48) and tuple(average.shape) == (8, FILTERS)
    gamma, beta = model.film_parameters(link, average)
    assert tuple(gamma.shape) == tuple(beta.shape) == (8, FILTERS)
    with pytest.raises(ValueError, match="link must be"):
        model(ids, link[:, :2])                  # two numbers instead of three
    with pytest.raises(ValueError, match="link must be"):
        model(ids, link[:5])                     # a different batch from the text


# ------------------------------------------------------------------ (a) the start is the plain text CNN
def test_a_generator_at_zero_gives_gamma_one_beta_zero():
    model = FiLMTextCNN(embedding())
    ids, link = batch()
    gamma, beta = model.film_parameters(link, model.conv_features(ids)[1])
    assert (gamma == 1).all() and (beta == 0).all()
    assert (model.generator.weight == 0).all() and (model.generator.bias == 0).all()


def test_a_at_the_start_the_output_equals_the_network_without_a_generator():
    model = FiLMTextCNN(embedding()).eval()
    ids, link = batch()
    with torch.no_grad():
        scores = model.conv(model.embedding(ids).transpose(1, 2))
        pooled = F.relu(scores).max(dim=2).values                       # no generator, no FiLM
        reference = model.output(F.relu(model.dense(pooled)))
        assert torch.equal(model(ids, link), reference)
        assert torch.equal(model(ids, torch.rand_like(link)), reference)  # any link numbers: the same output


def test_a_at_the_start_the_features_equal_the_text_encoders():
    """The same weights in the existing TextEncoder give the same 64 numbers, so a trained text CNN can start E2."""
    emb = embedding()
    encoder = TextEncoder(emb, FILTERS).eval()
    model = FiLMTextCNN(emb).eval()
    model.conv.load_state_dict(encoder.conv.state_dict())
    ids, link = batch()
    with torch.no_grad():
        assert torch.equal(model.encode(ids, link), encoder.encode(ids))


def test_a_a_text_encoder_state_dict_loads_into_embedding_and_conv():
    emb = embedding()
    state = TextEncoder(emb, FILTERS).state_dict()
    model = FiLMTextCNN(embedding(seed=5))
    result = model.load_state_dict({k: v for k, v in state.items() if k.startswith(("embedding.", "conv."))}, strict=False)
    assert result.unexpected_keys == []
    assert {k.split(".")[0] for k in result.missing_keys} == {"generator", "dense", "output"}
    assert torch.equal(model.conv.weight, state["conv.weight"]) and torch.equal(model.embedding.weight, state["embedding.weight"])


# ------------------------------------------------------------------ (b) size
def test_b_trainable_parameters_are_exactly_70658():
    model = FiLMTextCNN(embedding())
    assert trainable(model) == 70658
    parts = {name: sum(p.numel() for p in layer.parameters()) for name, layer in
             (("conv", model.conv), ("generator", model.generator), ("dense", model.dense), ("output", model.output))}
    assert parts == {"conv": 57664, "generator": 8704, "dense": 4160, "output": 130}


def test_b_gamma_only_has_66306_and_half_the_generator():
    model = FiLMTextCNN(embedding(), use_beta=False)
    assert trainable(model) == 66306
    assert sum(p.numel() for p in model.generator.parameters()) == 4352


def test_b_the_embedding_is_trainable_by_default_and_counted_apart():
    """D6, changed 2026-10-20: the embedding is fine-tuned, as in the diagram."""
    model = FiLMTextCNN(embedding())
    assert model.freeze_embedding is False and model.embedding.weight.requires_grad
    all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all_trainable == 70658 + FRESH * 300 and trainable(model) == 70658      # the vocabulary x 300 comes on top


def test_b_freeze_embedding_keeps_it_at_its_fasttext_values():
    emb = embedding()
    model = FiLMTextCNN(emb, freeze_embedding=True)
    assert model.freeze_embedding and not model.embedding.weight.requires_grad and not emb.weight.requires_grad
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 70658          # the original count, exactly
    before = model.embedding.weight.clone()
    ids, link = batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    F.cross_entropy(model(ids, link), torch.zeros(8, dtype=torch.long)).backward()
    optimizer.step()
    assert torch.equal(model.embedding.weight, before)


def test_b_a_trainable_embedding_moves_after_one_step_and_the_padding_row_does_not():
    emb = embedding()
    with torch.no_grad():
        emb.weight[0].zero_()                                          # the padding row starts at zero, as in the real matrix
    model = FiLMTextCNN(emb)
    before = model.embedding.weight.clone()
    ids, link = batch()
    ids[:, 40:] = 0                                                    # some padding in every message
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    F.cross_entropy(model(ids, link), torch.tensor([0, 1] * 4)).backward()
    assert model.embedding.weight.grad is not None and model.embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    used = torch.unique(ids[ids != 0])
    assert not torch.equal(model.embedding.weight[used], before[used])        # the words in the batch moved
    assert torch.equal(model.embedding.weight[0], before[0])                  # padding stays where it was
    unused = torch.tensor(sorted(set(range(1, FRESH)) - set(used.tolist())))
    if len(unused):
        assert torch.equal(model.embedding.weight[unused], before[unused])    # words not in the batch did not move


def test_b_the_embedding_trains_through_the_film_path_too():
    """The embedding gets its gradient through the generator's text summary as well, once the generator is non-zero."""
    model = FiLMTextCNN(embedding())
    randomize_generator(model, std=0.5)
    ids, link = batch()
    F.cross_entropy(model(ids, link), torch.tensor([0, 1] * 4)).backward()
    assert model.embedding.weight.grad.abs().sum() > 0 and model.generator.weight.grad.abs().sum() > 0


# ------------------------------------------------------------------ (c) the generator learns
def test_c_the_generator_gets_a_gradient_and_moves_after_one_step():
    model = FiLMTextCNN(embedding()).train()
    ids, link = batch()
    labels = torch.tensor([0, 1] * 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    F.cross_entropy(model(ids, link), labels).backward()
    assert model.generator.weight.grad is not None and model.generator.weight.grad.abs().sum() > 0
    assert model.generator.bias.grad.abs().sum() > 0
    optimizer.step()
    assert model.generator.weight.abs().sum() > 0 and model.generator.bias.abs().sum() > 0   # no longer all zero


# ------------------------------------------------------------------ (d) the link changes gamma and beta
def test_d_with_a_non_zero_generator_the_link_changes_gamma_and_beta_and_nothing_else_does():
    model = FiLMTextCNN(embedding()).eval()
    ids, link = batch()
    other_link = link.clone()
    other_link[:, 0] = 1 - link[:, 0]                                        # a different P, the text is the same
    average = model.conv_features(ids)[1]

    g0, b0 = model.film_parameters(link, average)
    g1, b1 = model.film_parameters(other_link, average)
    assert torch.equal(g0, g1) and torch.equal(b0, b1)                       # at the start the link has no effect

    randomize_generator(model)
    g0, b0 = model.film_parameters(link, average)
    g1, b1 = model.film_parameters(other_link, average)
    assert not torch.allclose(g0, g1) and not torch.allclose(b0, b1)         # now it does
    g2, b2 = model.film_parameters(link, average)
    assert torch.equal(g0, g2) and torch.equal(b0, b2)                       # the same link gives the same numbers


def test_d_the_text_summary_also_changes_gamma():
    """The same link with different words gives different gamma: the generator reads the Conv average (design E2, not E)."""
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model)
    (ids_a, link), (ids_b, _) = batch(seed=1), batch(seed=2)
    gamma_a, _ = model.film_parameters(link, model.conv_features(ids_a)[1])
    gamma_b, _ = model.film_parameters(link, model.conv_features(ids_b)[1])
    assert not torch.allclose(gamma_a, gamma_b)


def test_d_the_link_changes_the_prediction_features_once_the_generator_is_non_zero():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    other_link = 1 - link
    with torch.no_grad():
        assert not torch.allclose(model.encode(ids, link), model.encode(ids, other_link))


# ------------------------------------------------------------------ (e) the gamma rule
def test_e_bounded_gamma_stays_between_zero_and_two_and_free_gamma_does_not():
    ids, link = batch(n=32)
    free, bounded = FiLMTextCNN(embedding(), gamma_mode="free"), FiLMTextCNN(embedding(), gamma_mode="bounded")
    for model in (free, bounded):
        randomize_generator(model, std=5.0)
    g_free = free.film_parameters(link, free.conv_features(ids)[1])[0]
    g_bounded = bounded.film_parameters(link, bounded.conv_features(ids)[1])[0]
    assert g_bounded.min() >= 0 and g_bounded.max() <= 2
    assert g_free.min() < 0 and g_free.max() > 2


def test_e_the_default_gamma_is_bounded_and_unbounded_is_the_option():
    """D2, changed 2026-10-20: bounded (1 + tanh) is the default; the unbounded rule (1 + delta) is an ablation."""
    default = FiLMTextCNN(embedding())
    assert default.gamma_mode == "bounded"
    ids, link = batch(n=32)
    randomize_generator(default, std=5.0)
    gamma = default.film_parameters(link, default.conv_features(ids)[1])[0]
    assert gamma.min() >= 0 and gamma.max() <= 2
    free = FiLMTextCNN(embedding(), gamma_mode="free")
    randomize_generator(free, std=5.0)
    g_free = free.film_parameters(link, free.conv_features(ids)[1])[0]
    assert g_free.min() < 0 or g_free.max() > 2


def test_e_bounded_gamma_also_starts_at_one_and_a_bad_mode_is_refused():
    ids, link = batch()
    model = FiLMTextCNN(embedding(), gamma_mode="bounded")
    gamma, beta = model.film_parameters(link, model.conv_features(ids)[1])
    assert (gamma == 1).all() and (beta == 0).all()
    with pytest.raises(ValueError, match="gamma_mode"):
        FiLMTextCNN(embedding(), gamma_mode="tanh")


def test_gamma_only_keeps_beta_at_zero():
    model = FiLMTextCNN(embedding(), use_beta=False)
    randomize_generator(model)
    ids, link = batch()
    gamma, beta = model.film_parameters(link, model.conv_features(ids)[1])
    assert (beta == 0).all() and not (gamma == 1).all()
    assert tuple(gamma.shape) == tuple(beta.shape) == (8, FILTERS)
    assert tuple(model(ids, link).shape) == (8, 2)


# ------------------------------------------------------------------ (g) modality dropout
def test_g_text_keep_zero_hides_the_text_numbers():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    (ids_a, link), (ids_b, _) = batch(seed=1), batch(seed=2)
    with torch.no_grad():
        hidden = model.encode(ids_a, link, text_keep=torch.zeros(8))
        assert (hidden == 0).all()
        # with the text hidden the head sees zeros, so different messages (same link) get the same logits
        assert torch.equal(model(ids_a, link, text_keep=torch.zeros(8)), model(ids_b, link, text_keep=torch.zeros(8)))


def test_g_text_keep_works_per_message_and_one_means_unchanged():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    keep = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1.0])
    with torch.no_grad():
        plain, masked = model.encode(ids, link), model.encode(ids, link, text_keep=keep)
        assert torch.equal(masked[keep == 0], torch.zeros(3, FILTERS))
        assert torch.equal(masked[keep == 1], plain[keep == 1])
        assert torch.equal(model.encode(ids, link, text_keep=torch.ones(8)), plain)


# ------------------------------------------------------------------ the Conv average options
def test_the_masked_average_counts_only_positions_with_a_real_word():
    model = FiLMTextCNN(embedding(), mask_padding=True)
    message = torch.zeros(1, 50, dtype=torch.long)
    message[0, :5] = torch.tensor([7, 8, 9, 10, 11])                  # 5 real words, 45 padding
    x, average = model.conv_features(message)
    counted = sum(1 for j in range(48) if (message[0, j:j + 3] != 0).any())
    assert counted == 5                                               # windows starting at 0..4 hold a real word
    assert torch.allclose(average[0], x[0, :, :counted].mean(dim=1), atol=1e-6)
    plain = FiLMTextCNN(embedding())
    assert not torch.allclose(plain.conv_features(message)[1], average)


def test_a_message_of_only_padding_gives_a_finite_average_of_zero():
    model = FiLMTextCNN(embedding(), mask_padding=True)
    _, average = model.conv_features(torch.zeros(1, 50, dtype=torch.long))
    assert torch.isfinite(average).all() and (average == 0).all()


def test_the_average_after_relu_is_never_negative():
    model = FiLMTextCNN(embedding(), average_after_relu=True)
    ids, _ = batch()
    assert (model.conv_features(ids)[1] >= 0).all()


def test_masking_needs_a_padding_index():
    with pytest.raises(ValueError, match="padding_idx"):
        FiLMTextCNN(torch.nn.Embedding(FRESH, 300), mask_padding=True)


@pytest.mark.parametrize("options", [{}, {"mask_padding": True}, {"average_after_relu": True},
                                     {"gamma_mode": "bounded"}, {"gamma_mode": "free"}, {"use_beta": False}])
def test_every_option_runs_and_keeps_the_size_of_the_outputs(options):
    model = FiLMTextCNN(embedding(), **options)
    ids, link = batch()
    assert tuple(model(ids, link).shape) == (8, 2)


# ------------------------------------------------------------------ weight-decay groups
@pytest.mark.parametrize("freeze", [False, True])
def test_parameter_groups_cover_every_trainable_parameter_once_with_the_right_decay(freeze):
    model = FiLMTextCNN(embedding(), freeze_embedding=freeze)
    groups = model.parameter_groups(1e-4)
    listed = [p for g in groups for p in g["params"]]
    wanted = [p for p in model.parameters() if p.requires_grad]
    assert len(listed) == len(wanted) and {id(p) for p in listed} == {id(p) for p in wanted}
    in_groups = any(p is model.embedding.weight for p in listed)
    assert in_groups is (not freeze)                                   # a frozen embedding is in no group
    if not freeze:
        own = [g for g in groups if any(p is model.embedding.weight for p in g["params"])]
        assert len(own) == 1 and own[0]["params"] == [model.embedding.weight] and own[0]["weight_decay"] == 0.0
    decay = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert decay[id(model.dense.weight)] == 1e-4 and decay[id(model.generator.weight)] == 1e-4
    for layer, name in ((model.dense.bias, "dense bias"), (model.generator.bias, "generator bias"), (model.conv.weight, "conv"),
                        (model.output.weight, "output")):
        assert decay[id(layer)] == 0.0, name
    torch.optim.Adam(groups, lr=1e-3)                                 # the optimizer accepts them


def test_the_generator_can_have_its_own_decay():
    groups = FiLMTextCNN(embedding()).parameter_groups(1e-4, generator_weight_decay=1e-3)
    assert [g["weight_decay"] for g in groups[:2]] == [1e-4, 1e-3]


def test_gamma_only_groups_also_cover_everything():
    model = FiLMTextCNN(embedding(), use_beta=False)
    listed = {id(p) for g in model.parameter_groups(1e-4) for p in g["params"]}
    assert listed == {id(p) for p in model.parameters() if p.requires_grad}


# ------------------------------------------------------------------ the real thing
@pytest.mark.parametrize("options", [{}, {"mask_padding": True}, {"average_after_relu": True}, {"gamma_mode": "bounded"},
                                     {"gamma_mode": "free"}, {"use_beta": False}])
def test_at_the_start_a_real_trained_text_cnn_is_reproduced_exactly(options):
    """Seed 42, fold 0 of the synthetic pilot: the generator-at-zero model gives the trained CNN's 64 numbers, bit for bit."""
    from src.config import CROWD_PILOT_PATH
    from src.data.split_crowd import load_fold
    from src.unified import text_cnn as tc
    from src.unified.film_inputs import link_inputs

    dataset = tc.dataset_path(True)
    checkpoint = tc.model_dir(dataset) / "phase_a_seed42_fold0.pt"
    if not (tc.ROOT / CROWD_PILOT_PATH).exists() or not checkpoint.exists() or not checkpoint.with_suffix(".json").exists():
        pytest.skip("the synthetic pilot files are not on this machine")
    cnn, vocab, _ = tc.load_phase_a(checkpoint)
    _, _, seq_len = tc.fold_text_setup(dataset, 42, 0)
    token_ids, _ = tc.tensors(load_fold(42, 0, pilot=True)["val"], vocab, seq_len)
    link = torch.tensor(link_inputs(42, 0, pilot=True)["val"].stack("full"))

    model = FiLMTextCNN(cnn.embedding, **options)
    state = {k: v for k, v in cnn.state_dict().items() if k.startswith(("embedding.", "conv."))}
    assert model.load_state_dict(state, strict=False).unexpected_keys == []
    model.eval()
    cnn.eval()
    with torch.no_grad():
        assert torch.equal(model.encode(token_ids, link), cnn.encode(token_ids))


# ------------------------------------------------------------------ step 2.4: explanations
def adjusted_scores(model, ids, link):
    x, average = model.conv_features(ids)
    gamma, beta = model.film_parameters(link, average)
    return gamma.unsqueeze(2) * x + beta.unsqueeze(2)


def test_encode_with_positions_gives_the_same_pooled_values_as_forward():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    with torch.no_grad():
        pooled, position = model.encode_with_positions(ids, link)
        assert torch.equal(pooled, model.encode(ids, link))
        logits = model.output(model.dropout_head(F.relu(model.dense(pooled))))
        assert torch.equal(logits, model(ids, link))                      # the same numbers the verdict is made from
    assert tuple(pooled.shape) == tuple(position.shape) == (8, FILTERS) and position.dtype == torch.long
    assert position.min() >= 0 and position.max() <= 47


def test_the_position_points_at_the_window_that_gave_the_value():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    with torch.no_grad():
        pooled, position = model.encode_with_positions(ids, link)
        picked = adjusted_scores(model, ids, link).gather(2, position.unsqueeze(2)).squeeze(2)
    assert torch.allclose(F.relu(picked), pooled)


def test_encode_with_positions_keeps_the_graph_for_gradient_explanations():
    model = FiLMTextCNN(embedding()).train()
    ids, link = batch()
    pooled, _ = model.encode_with_positions(ids, link)
    assert pooled.requires_grad
    pooled.sum().backward()
    assert model.conv.weight.grad is not None


def test_explain_at_the_start_shows_no_effect_of_the_link():
    model = FiLMTextCNN(embedding()).eval()
    ids, link = batch()
    e = model.explain(ids, link)
    assert (e.gamma == 1).all() and (e.beta == 0).all() and (e.lift == 0).all()
    assert torch.equal(e.pooled, e.plain_pooled) and torch.equal(e.position, e.plain_position)
    assert torch.equal(e.pooled, model.encode(ids, link))


def test_explain_reports_the_same_gamma_beta_and_pooled_values_the_model_uses():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    e = model.explain(ids, link)
    gamma, beta = model.film_parameters(link, model.conv_features(ids)[1])
    assert torch.equal(e.gamma, gamma) and torch.equal(e.beta, beta)
    assert torch.equal(e.pooled, model.encode(ids, link))
    assert torch.equal(e.lift, e.pooled - e.plain_pooled)
    assert torch.equal(e.position, model.encode_with_positions(ids, link)[1])
    assert not torch.equal(e.pooled, e.plain_pooled)                       # the link did change something


def test_explain_gives_the_token_ids_of_each_winning_window():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    e = model.explain(ids, link)
    assert tuple(e.window_ids.shape) == (8, FILTERS, 3)
    for b, f in ((0, 0), (3, 17), (7, 63), (5, 40)):
        start = int(e.position[b, f])
        assert torch.equal(e.window_ids[b, f], ids[b, start:start + 3])


def test_lift_is_positive_when_the_link_raises_filters_and_negative_when_it_lowers_them():
    ids, link = batch()
    for shift, sign in ((1.0, 1), (-1.0, -1)):
        model = FiLMTextCNN(embedding()).eval()
        with torch.no_grad():
            model.generator.bias[FILTERS:] = shift                         # beta = shift for every filter, gamma stays 1
        e = model.explain(ids, link)
        assert (sign * e.lift >= 0).all() and (sign * e.lift > 0).any()
        assert (e.gamma == 1).all() and (e.beta == shift).all()


def test_a_filter_raised_by_the_link_alone_is_flagged_as_having_no_phrase():
    """On a message with no words, a large beta lifts filters although there is no phrase to point at."""
    model = FiLMTextCNN(embedding()).eval()
    with torch.no_grad():
        model.generator.bias[FILTERS:] = 5.0
    nothing = torch.zeros(1, 50, dtype=torch.long)
    e = model.explain(nothing, torch.rand(1, 3))
    assert (e.pooled > 0).all() and not e.window_has_words.any()
    words = torch.randint(1, FRESH, (1, 50))                               # a full message: every window has words
    assert model.explain(words, torch.rand(1, 3)).window_has_words.all()


def test_explain_is_read_only():
    model = FiLMTextCNN(embedding()).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    before = model(ids, link).detach().clone()
    e = model.explain(ids, link)
    assert not any(t.requires_grad for t in e if isinstance(t, torch.Tensor))
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(model(ids, link), before)


@pytest.mark.parametrize("options", [{"mask_padding": True}, {"average_after_relu": True}, {"gamma_mode": "bounded"},
                                     {"gamma_mode": "free"}, {"use_beta": False}])
def test_explain_works_with_every_option(options):
    model = FiLMTextCNN(embedding(), **options).eval()
    randomize_generator(model, std=0.5)
    ids, link = batch()
    e = model.explain(ids, link)
    assert tuple(e.lift.shape) == (8, FILTERS) and torch.equal(e.pooled, model.encode(ids, link))
    if not options.get("use_beta", True):
        assert (e.beta == 0).all()


def test_at_the_start_positions_and_values_match_the_text_encoders_on_a_real_fold():
    """The existing explanation code reads (pooled, position) from TextEncoder.encode_with_positions; E2 gives the same."""
    from src.config import CROWD_PILOT_PATH
    from src.data.split_crowd import load_fold
    from src.unified import text_cnn as tc
    from src.unified.film_inputs import link_inputs

    dataset = tc.dataset_path(True)
    checkpoint = tc.model_dir(dataset) / "phase_a_seed42_fold0.pt"
    if not (tc.ROOT / CROWD_PILOT_PATH).exists() or not checkpoint.exists() or not checkpoint.with_suffix(".json").exists():
        pytest.skip("the synthetic pilot files are not on this machine")
    cnn, vocab, _ = tc.load_phase_a(checkpoint)
    _, _, seq_len = tc.fold_text_setup(dataset, 42, 0)
    token_ids, _ = tc.tensors(load_fold(42, 0, pilot=True)["val"], vocab, seq_len)
    link = torch.tensor(link_inputs(42, 0, pilot=True)["val"].stack("full"))
    model = FiLMTextCNN(cnn.embedding)
    model.load_state_dict({k: v for k, v in cnn.state_dict().items() if k.startswith(("embedding.", "conv."))}, strict=False)
    model.eval()
    cnn.eval()
    with torch.no_grad():
        mine, mine_position = model.encode_with_positions(token_ids, link)
        theirs, theirs_position = cnn.encode_with_positions(token_ids)
    assert torch.equal(mine, theirs) and torch.equal(mine_position, theirs_position)
