# src/features/model/rf_gating_module.py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.features.model.train import LABEL_TO_IDX

# Investigated 2026-09-08 (see docs/DEV_TEAM_BUILD_PLAN.md Section 0's gate-neutrality
# risk): fitting the RF on X,y and then measuring permutation_importance on that SAME
# X,y made every one of 203 gates come back exactly neutral (1.0), across all 3 seeds.
# Root cause, confirmed by holding out a slice and comparing: this RF (200 trees, no
# depth limit, min_samples_leaf=1) memorizes its own training rows (in-sample F1 was
# exactly 1.000000). Once a model has memorized a row, shuffling one feature and
# re-scoring against that SAME row barely changes anything -- it has redundant paths
# back to the memorized answer. Measuring on a held-out slice the RF never fit on
# recovered real, non-zero, distinguishing importance for the CNN features. This
# fraction controls how much of the RF-fitting split (Task 1.9's rf_fit slice) is
# reserved for scoring rather than fitting -- both pieces stay inside rf_fit; Train and
# Test are never touched by this.
PERM_IMPORTANCE_HOLDOUT_FRACTION = 0.3

# Parallelism for the RF fit and permutation importance (the gated model's slowest training
# step). Verified 2026-10-05 on the deployed Phase A checkpoint, all 3 seeds: n_jobs=-1
# produces gates bit-for-bit identical to the single-core run (both take random_state, so
# the result doesn't depend on worker count) while cutting this step ~2.5-3.4x on 16 cores.
# n_repeats stays at 30 -- the same test showed 10-20 repeats measurably shift the CNN gates.
RF_N_JOBS = -1

# Columns that always keep a fixed gate of 1.0 and are never scored (docs/dataset_and_training_plan.md
# 7.2): has_url, and the extractor's P and u. P and u already ARE a forest's output, so ranking
# them against raw link features would be circular. Only the names present in a run are used.
FIXED_GATE_COLUMNS = ("has_url", "P", "u")


@torch.no_grad()
def extract_cnn_features(model, token_ids: torch.Tensor) -> np.ndarray:
    # Frozen forward pass. model.eval() disables Dropout; @torch.no_grad() stops
    # gradient bookkeeping entirely -- this task only ever READS the CNN's
    # output, it never trains it further.
    model.eval()
    features = model.encode(token_ids)  # (batch, num_filters) -- NOT .forward()
    return features.numpy()


def build_url_feature_matrix(url_feature_dicts: list[dict]) -> tuple[np.ndarray, list[str]]:
    # pandas aligns columns by NAME, not by dict insertion order, so this stays
    # correct even if Task 1.7 later drops/combines a URL feature -- no code
    # change needed here, only fewer columns out.
    df = pd.DataFrame(url_feature_dicts)
    return df.values.astype(np.float32), df.columns.tolist()


def build_gating_matrix(
    cnn_features: np.ndarray, url_features: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    assert cnn_features.shape[0] == url_features.shape[0], "row count mismatch"
    X = np.concatenate([cnn_features, url_features], axis=1)
    return X, cnn_features.shape[1], url_features.shape[1]


def fit_gating_rf(
    X: np.ndarray, y: np.ndarray, seed: int,
    n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1,
    class_weight: str | None = None,
) -> RandomForestClassifier:
    # This is the ONLY call site for RandomForestClassifier(...).fit(...) in the
    # whole codebase -- one shot, outside any epoch/training loop.
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=seed, class_weight=class_weight, n_jobs=RF_N_JOBS,
    )
    rf.fit(X, y)
    return rf


def compute_importances(
    rf: RandomForestClassifier, X: np.ndarray, y: np.ndarray, seed: int,
    scoring: str = "f1", n_repeats: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Primary metric.
    result = permutation_importance(
        rf, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=RF_N_JOBS,
    )
    perm_mean, perm_std = result.importances_mean, result.importances_std

    # Comparison value ONLY -- never fed into gate_transform.
    gini = rf.feature_importances_
    return perm_mean, perm_std, gini


def gate_transform(importance_group: np.ndarray) -> np.ndarray:
    # Appendix A's formula, with its near-zero-variance fallback.
    lo, hi = importance_group.min(), importance_group.max()
    if hi - lo < 1e-8:
        return np.ones_like(importance_group)
    return 0.1 + (importance_group - lo) / (hi - lo) * (3.0 - 0.1)


def select_top_k(ranking, k: int) -> np.ndarray:
    # C-prune (docs/c_prune_architecture.md Phase 5): `ranking` lists every filter id once, best first
    # (src/unified/gate_rf.py, step 4). keep_idx = the first k, returned in ascending id order, so the
    # pruned Conv keeps the filters in their original order and kept position i maps back to keep_idx[i].
    ranking = np.asarray(ranking, dtype=int)
    if sorted(ranking.tolist()) != list(range(len(ranking))):
        raise ValueError("ranking must list every filter id 0..n-1 exactly once")
    if not 1 <= k <= len(ranking):
        raise ValueError(f"k must be between 1 and {len(ranking)}, got {k}")
    return np.sort(ranking[:k])


# Maps a permutation_importance `scoring` string to the plain (y_true, y_pred) function
# used for the group's own in-sample/held-out/baseline scores below -- keeps both
# measurements on the same metric instead of silently mixing f1 and balanced_accuracy.
_SCORE_FUNCS = {"f1": f1_score, "balanced_accuracy": balanced_accuracy_score}


def majority_class_score(y_fit: np.ndarray, y_score: np.ndarray, scoring: str = "f1") -> float:
    majority_label = np.bincount(y_fit).argmax()
    predictions = np.full_like(y_score, majority_label)
    return _SCORE_FUNCS[scoring](y_score, predictions)


def fit_and_score_group(
    X: np.ndarray, y: np.ndarray, idx_fit: np.ndarray, idx_score: np.ndarray, seed: int,
    scoring: str = "f1", n_repeats: int = 30, class_weight: str | None = None,
) -> dict:
    # Investigated 2026-09-21: fitting ONE joint RF over CNN's 64 dims + URL's 11 dims
    # structurally starves the URL block. sklearn's default max_features='sqrt' samples
    # ~14 of 203 features per split, so a URL feature is a split CANDIDATE on well under
    # 1 split in most nodes -- and even when it is, CNN's much more discriminative
    # continuous features almost always win the greedy impurity-reduction comparison.
    # Measured: across 200 trees / 7177 total splits, only 13 (0.18%) ever used ANY URL
    # feature, so permutation importance on the joint model comes back at machine-precision
    # 0.0 for all 11 -- not because URL features lack signal, but because the fitted trees
    # never reference them. Fitting each feature group as its OWN separate RF removes this
    # cross-group masking entirely.
    X_fit, X_score = X[idx_fit], X[idx_score]
    y_fit, y_score = y[idx_fit], y[idx_score]
    score_fn = _SCORE_FUNCS[scoring]

    rf = fit_gating_rf(X_fit, y_fit, seed, class_weight=class_weight)
    in_sample_score = score_fn(y_fit, rf.predict(X_fit))
    held_out_score = score_fn(y_score, rf.predict(X_score))
    baseline_score = majority_class_score(y_fit, y_score, scoring)
    perm_mean, perm_std, gini = compute_importances(rf, X_score, y_score, seed, scoring, n_repeats)

    # Permutation importance only reflects genuine per-feature reliance when the group's
    # OWN isolated model actually generalizes. When a group is weak (e.g. URL features
    # alone: held-out F1 0.55 vs a 0.73 majority-class baseline, confirmed across every
    # regularization setting tested), the fitted RF is effectively overfit noise, and
    # permutation importance on it comes back INVERTED relative to which features the
    # trees actually used -- e.g. near-zero-Gini features (barely used) score as "most
    # important" while the highest-Gini feature (most used) scores as "least important".
    # Gini importance doesn't depend on held-out generalization, so it stays a stable,
    # non-inverted (if more limited) signal even when the group's model is too weak to
    # beat a trivial baseline. This is a general fallback rule, not URL-specific: any
    # feature group whose isolated model can't beat guessing the majority class gets
    # scored by Gini instead of permutation importance.
    used_gini_fallback = held_out_score < baseline_score
    importance = gini if used_gini_fallback else perm_mean
    gates = gate_transform(importance)

    return {
        "gates": gates, "perm_mean": perm_mean, "perm_std": perm_std, "gini": gini,
        "in_sample_score": in_sample_score, "held_out_score": held_out_score,
        "baseline_score": baseline_score, "scoring": scoring,
        "used_gini_fallback": used_gini_fallback,
        # Backward-compatible aliases -- notebooks/06_run_rf_gating.py and older callers
        # read in_sample_f1/held_out_f1 regardless of which scoring metric was actually used.
        "in_sample_f1": in_sample_score, "held_out_f1": held_out_score, "baseline_f1": baseline_score,
    }


def fit_and_score_url_group(
    url_features: np.ndarray, url_feature_names: list[str], y: np.ndarray,
    idx_fit: np.ndarray, idx_score: np.ndarray, seed: int, n_repeats: int = 30,
) -> dict:
    # Investigated 2026-09-21/22: URL features split into two tiers.
    #
    # 'has_url' is EXCLUDED from the gating multiplier entirely -- held fixed at
    # neutral (1.0), never scored by any importance method. It can only be measured
    # by fitting it alongside the other 10 URL features in one block (gate_transform
    # min-max normalizes WITHIN a group, so a single-feature group has no variance to
    # normalize against and always comes back neutral regardless of true importance --
    # confirmed: tried scoring it alone first, got exactly 1.0/std=0.0 every seed).
    # But that 11-feature combination never beats a trivial majority-class baseline
    # (confirmed every seed) -- exactly the regime where permutation importance is
    # proven unreliable (see fit_and_score_group's comment: it comes back INVERTED,
    # ranking barely-used features as "most important"). The project's methodology
    # is permutation importance with no exceptions, so rather than quietly falling
    # back to Gini for this one feature, it's held out of the multiplier altogether.
    # Its own signal isn't lost to the model -- has_url's raw value still reaches the
    # fusion classifier, just with a fixed 1.0 multiplier instead of a learned one,
    # same as if it were simply left ungated.
    #
    # The other 10 features are only DEFINED for rows that actually have a URL; the
    # rest carry sentinel placeholder values (see url_features.py's no_url_features()).
    # Scoring them over the full split -- ~75% sentinel duplicates, on a dataset
    # that's majority-smishing overall -- drowns the real signal: the isolated
    # 11-feature model can't even beat a trivial majority-class baseline (confirmed:
    # held-out F1 0.55 vs baseline 0.73, every regularization setting tested), so
    # fit_and_score_group's Gini fallback fires every time and permutation importance
    # never gets a chance. Restricting to has_url==1 rows recovers a completely
    # different dataset -- ~90% smishing among URL-containing messages -- so 'f1'
    # (which trivially rewards always guessing the majority class here) is swapped
    # for class_weight='balanced' + 'balanced_accuracy' scoring. That recovers real,
    # cross-seed-reproducible signal: held-out balanced_accuracy beat its own
    # majority-class baseline by 28-35 points in every one of 3 seeds tested, with
    # url_length and number_of_slashes as the two dominant, non-inverted predictors
    # every time.
    if "has_url" not in url_feature_names:
        # No has_url column to gate on (e.g. a synthetic/test feature set without the
        # real production schema) -- fall back to treating the whole block as one
        # group, same as before this two-tier split existed.
        result = fit_and_score_group(
            url_features, y, idx_fit, idx_score, seed, scoring="f1", n_repeats=n_repeats,
        )
        return {**result, "detail_group": result, "detail_feature_names": url_feature_names}

    has_url_idx = url_feature_names.index("has_url")
    fixed_idx = [i for i, name in enumerate(url_feature_names) if name in FIXED_GATE_COLUMNS]
    detail_idx = [i for i in range(len(url_feature_names)) if i not in fixed_idx]
    detail_names = [url_feature_names[i] for i in detail_idx]

    has_url_mask = url_features[:, has_url_idx] == 1
    sub_idx_fit = idx_fit[has_url_mask[idx_fit]]
    sub_idx_score = idx_score[has_url_mask[idx_score]]
    detail_result = fit_and_score_group(
        url_features[:, detail_idx], y, sub_idx_fit, sub_idx_score, seed,
        scoring="balanced_accuracy", n_repeats=n_repeats, class_weight="balanced",
    )

    # Reassemble into the ORIGINAL column order -- the fixed columns' positions are
    # whatever build_url_feature_matrix's pandas column ordering produced, not
    # necessarily last, so this can't assume a fixed layout. Their importance
    # slots are NaN, not 0.0 -- 0.0 would read as "measured, found unimportant";
    # NaN means "deliberately excluded, never measured."
    n = len(url_feature_names)
    gates, perm_mean, perm_std, gini = (np.empty(n) for _ in range(4))
    for orig_i in fixed_idx:
        gates[orig_i] = 1.0
        perm_mean[orig_i] = perm_std[orig_i] = gini[orig_i] = np.nan
    for local_i, orig_i in enumerate(detail_idx):
        gates[orig_i] = detail_result["gates"][local_i]
        perm_mean[orig_i] = detail_result["perm_mean"][local_i]
        perm_std[orig_i] = detail_result["perm_std"][local_i]
        gini[orig_i] = detail_result["gini"][local_i]

    return {
        "gates": gates, "perm_mean": perm_mean, "perm_std": perm_std, "gini": gini,
        "detail_group": detail_result, "detail_feature_names": detail_names,
        # in_sample_f1/held_out_f1/baseline_f1 kept as the detail sub-group's numbers --
        # despite the "_f1" name, these are on the balanced_accuracy scale for this
        # group (see detail_result["scoring"]). has_url has no group result of its own
        # to report here since it's never fit/scored at all.
        "in_sample_f1": detail_result["in_sample_score"], "held_out_f1": detail_result["held_out_score"],
        "baseline_f1": detail_result["baseline_score"], "used_gini_fallback": detail_result["used_gini_fallback"],
    }


def export_gates(gates: np.ndarray, seed: int, out_dir: str = "models") -> str:
    # Only the gate array is persisted -- the fitted RandomForestClassifier
    # object is deliberately NOT saved anywhere. It's never loaded in the
    # inference/serving path.
    path = f"{out_dir}/gates_seed{seed}.npy"
    np.save(path, gates)
    return path


def run_one_seed(
    model, texts: list[str], url_feature_dicts: list[dict], labels: list[str],
    word_to_idx: dict[str, int], seq_len: int, seed: int,
    scoring: str = "balanced_accuracy", n_repeats: int = 30,
    perm_holdout_fraction: float = PERM_IMPORTANCE_HOLDOUT_FRACTION,
    out_dir: str = "models", encode_fn=None, url_gates_override: np.ndarray | None = None,
) -> dict:
    # encode_fn defaults to train.py's encode_batch -- strict, crashes on any word not in
    # word_to_idx, correct for the real rf_fit slice, which Task 1.8 built the vocab from.
    # Override only for a slice that can contain text the vocab was never built from (e.g.
    # notebooks/06b_run_rf_gating_augmented.py's synthetic rows) -- pass
    # src.serving.predict.encode_for_serving there, the same OOV-tolerant (drop-unknown-word)
    # encoder live serving already uses, rather than corrupting or crashing on new vocabulary.
    # This only changes what CNN features get extracted (frozen, read-only forward pass,
    # @torch.no_grad() below) -- it can't affect the CNN's own weights either way.
    if encode_fn is None:
        from src.features.model.train import encode_batch
        encode_fn = encode_batch

    token_ids = encode_fn(texts, word_to_idx, seq_len)
    cnn_features = extract_cnn_features(model, token_ids)
    url_features, url_feature_names = build_url_feature_matrix(url_feature_dicts)
    X, cnn_dim, url_dim = build_gating_matrix(cnn_features, url_features)
    y = np.array([LABEL_TO_IDX[label] for label in labels])

    # The rf_fit slice is split AGAIN, into an rf-fit sub-slice (fits the RF) and a
    # perm-score sub-slice (measures importance) -- see PERM_IMPORTANCE_HOLDOUT_FRACTION's
    # comment above for why. Both pieces are still exclusively from Task 1.9's rf_fit
    # slice; Train and Test remain untouched by RF fitting either way. Both feature
    # groups share the same row split, so their scores are directly comparable.
    idx_fit, idx_score = train_test_split(
        np.arange(len(y)), test_size=perm_holdout_fraction, stratify=y, random_state=seed,
    )

    # CNN and URL groups are fit as two SEPARATE RFs -- see fit_and_score_group's
    # comment for why a single joint RF structurally starves the smaller URL block.
    # The URL group gets its own two-tier treatment on top of that -- see
    # fit_and_score_url_group's comment.
    # docs/training_rules.md rule 6: the rf_fit split is uneven, so class-balanced forests scored
    # with balanced accuracy (plain F1 shifts with class balance).
    cnn_result = fit_and_score_group(
        X[:, :cnn_dim], y, idx_fit, idx_score, seed, scoring, n_repeats, class_weight="balanced",
    )
    if url_gates_override is None:
        url_result = fit_and_score_url_group(X[:, cnn_dim:], url_feature_names, y, idx_fit, idx_score, seed, n_repeats)
    else:
        # The head's non-text inputs come from an RF extractor (rf_feature_extractor.py): their gates are
        # that forest's own importances, supplied by the caller, not fitted on the SMS rows.
        assert len(url_gates_override) == len(url_feature_names), "override length does not match the inputs"
        n_url = len(url_feature_names)
        url_result = {"gates": np.asarray(url_gates_override, dtype=float),
                      "perm_mean": np.full(n_url, np.nan), "perm_std": np.full(n_url, np.nan),
                      "gini": np.full(n_url, np.nan), "in_sample_f1": np.nan, "held_out_f1": np.nan}

    gates = np.concatenate([cnn_result["gates"], url_result["gates"]])
    perm_mean = np.concatenate([cnn_result["perm_mean"], url_result["perm_mean"]])
    perm_std = np.concatenate([cnn_result["perm_std"], url_result["perm_std"]])
    gini = np.concatenate([cnn_result["gini"], url_result["gini"]])

    # out_dir defaults to "models" to preserve notebooks/06_run_rf_gating.py's existing
    # real-usage behavior unchanged -- but MUST be overridable, or any test calling this
    # with seed=42 silently overwrites the real production models/gates_seed42.npy (this
    # is exactly what happened before this parameter existed -- see
    # tests/test_gating_module.py's fix, same commit).
    path = export_gates(gates, seed, out_dir=out_dir)

    feature_names = [f"cnn_{i}" for i in range(cnn_dim)] + url_feature_names
    return {
        "seed": seed, "gates": gates, "perm_importance_mean": perm_mean,
        "perm_importance_std": perm_std, "gini_importance": gini,
        "feature_names": feature_names, "cnn_dim": cnn_dim, "url_dim": url_dim,
        "gate_path": path, "rf_fit_n": len(idx_fit), "perm_score_n": len(idx_score),
        # Kept as top-level in_sample_f1/held_out_f1 for backward compatibility with the
        # CNN group's numbers (the historically-reported values) -- per-group detail,
        # including whether the Gini fallback fired, is in cnn_group/url_group below.
        "in_sample_f1": cnn_result["in_sample_f1"], "held_out_f1": cnn_result["held_out_f1"],
        "cnn_group": cnn_result, "url_group": url_result,
    }


def stability_report(per_seed_results: list[dict], out_path: str = "experiments/gate_stability_report.csv"):
    # Mean +/- SD per feature, across every seed we ran.
    feature_names = per_seed_results[0]["feature_names"]
    gates_by_seed = np.stack([r["gates"] for r in per_seed_results])  # (n_seeds, n_features)

    report = pd.DataFrame({
        "feature_name": feature_names,
        "mean_gate": gates_by_seed.mean(axis=0),
        "std_gate": gates_by_seed.std(axis=0),
        "n_seeds": len(per_seed_results),
    })
    report.to_csv(out_path, index=False)
    return report
