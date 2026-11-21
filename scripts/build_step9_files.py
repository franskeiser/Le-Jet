r"""Step 9 of the experimental testing guide: classification error analysis of the proposed model (E2).

    .venv\Scripts\python.exe scripts/build_step9_files.py

Reads the Step 5 predictions, the dataset and the Link RF's saved test scores; nothing is trained or scored again.

1. One outcome per message. Each message is classified once per seed (3 times). Its outcome is the majority over the seeds
   (TP / TN / FP / FN in at least 2 of 3); `consistent` = the same outcome in all 3 seeds.
2. Patterns, by comparison: within smishing, FN against TP; within legitimate, FP against TN. For each characteristic the share of
   messages that have it in each group, and the difference. Characteristics: the guide's (urgent wording, financial terms, OTP /
   verification phrases, neutral wording, missing URL evidence, inaccessible webpage) plus reward wording, category, language, source,
   shortened link, fetch status, message length and the Link RF's P (what the forest said about the link; mean over the 3 seeds).
   The word lists are simple English + Tagalog patterns (WORDS below): a description of the errors, not a model.
3. Representative messages: per outcome, up to 8 messages (consistent ones first; FP / FN the most confidently wrong; TP / TN spread over
   the message conditions), ANONYMISED: a link becomes [link: <domain>], phone numbers [PHONE], other numbers of 4+ digits [NUMBER],
   emails [EMAIL]. Names cannot be masked reliably by rules: REVIEW EVERY MESSAGE BY HAND before it goes into the manuscript.
   For each, how many seeds each other main model got it wrong.

Writes experiments/film/<dataset>/step9/Classification_Error_Analysis.xlsx:
    Outcome_Counts       per model: TP / TN / FP / FN by majority, and consistent; overlap of E2's errors with the other models
    Patterns_FN_vs_TP    within smishing
    Patterns_FP_vs_TN    within legitimate
    By_Group             outcome counts of E2 by category, language, source, message condition
    Representative       the anonymised examples
    Per_Message          every message: outcome per seed, majority, characteristics (for the group's own reading)
    Possible_Label_Issues  link domains labelled BOTH smishing and legitimate, and rows that look like lab test records: for the
                         annotators to review (this script changes no label)
    Method
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                         # noqa: E402
from src.features.text_preprocessing import mask_urls, text_tokens  # noqa: E402
from src.unified import fit_model as fm                             # noqa: E402
from src.unified import link_lane as ll                             # noqa: E402
from src.unified import text_cnn as tc                              # noqa: E402

MAIN = [("Proposed Model", "Proposed_Model"), ("Concatenation Fusion", "Concatenation"), ("CNN-only", "CNN_only"),
        ("KNN-RF-ETC", "KNN_RF_ETC")]
WORDS = {
    "urgent wording": r"\burgent|immediate|\basap\b|\btoday\b|\bnow\b|expir|suspend|\block(ed)?\b|\bblock(ed)?\b|deactivat|limited|"
                      r"last chance|final (?:notice|warning)|within \d+|\bagad\b|ngayon|bilisan|huling|\bhurry",
    "OTP / verification phrases": r"\botp\b|one[- ]?time|\bpin\b|\bcode\b|verif|confirm|i-?verify|update (?:your|ang)|log ?in\b|sign ?in\b|"
                                  r"\bkyc\b|password",
    "financial terms": r"\bbank|gcash|\bmaya\b|paymaya|account|payment|\bpay\b|\bloan|credit|\bcard\b|wallet|transfer|peso|\bphp\b|₱|"
                       r"\bbpi\b|\bbdo\b|metrobank|landbank|unionbank|deposit|withdraw|balance|utang|pera|bayad",
    "reward wording": r"\bwin\b|\bwon\b|winner|prize|bonus|\bfree\b|reward|\bcash\b|\bclaim|libre|panalo|premyo|jackpot|gift",
}
LABEL = {1: "smishing", 0: "legitimate"}


def anonymise(text: str, row: pd.Series) -> str:
    out = mask_urls(str(text)).replace("<URL>", f"[link: {row.link_domain}]" if str(row.link_domain) not in ("-1", "") else "[link]")
    out = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", out)
    out = re.sub(r"(\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}", "[PHONE]", out)
    out = re.sub(r"\d{4,}", "[NUMBER]", out)
    return out


def outcome(actual: str, predicted: str) -> str:
    return {("smishing", "smishing"): "TP", ("smishing", "legitimate"): "FN", ("legitimate", "smishing"): "FP",
            ("legitimate", "legitimate"): "TN"}[(actual, predicted)]


def per_model_outcomes(step5: Path, stem: str) -> pd.DataFrame:
    t = pd.read_csv(step5 / f"{stem}_Predictions.csv")
    t["outcome"] = [outcome(a, p) for a, p in zip(t["Actual Label"], t["Predicted Label"])]
    w = t.pivot(index="Message ID", columns="Seed", values="outcome")
    w.columns = [f"seed_{c}" for c in w.columns]
    w["majority"] = w.mode(axis=1)[0]
    w["seeds_wrong"] = w[[f"seed_{s}" for s in SEEDS]].isin(["FP", "FN"]).sum(axis=1)
    w["consistent"] = w[[f"seed_{s}" for s in SEEDS]].nunique(axis=1) == 1
    w["mean_p_smishing"] = t.groupby("Message ID")["Predicted Probability"].mean()
    return w


def rf_p(dataset: Path) -> pd.Series:
    """The Link RF's P for each message with a link, from the saved TEST scores of its run (mean over the 3 seeds)."""
    frames = []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            f = pd.read_csv(ll.out_dir(dataset) / f"final_seed{seed}_fold{fold}.csv")
            frames.append(f[f.part == "test"][["message_id", "P"]])
    return pd.concat(frames).groupby("message_id")["P"].mean()


def pattern_table(d: pd.DataFrame, error: str, correct: str) -> pd.DataFrame:
    a, b = d[d.majority == error], d[d.majority == correct]
    rows = []

    def add(name, mask_fn, kind="share"):
        rows.append({"characteristic": name, f"{error} (n={len(a)})": mask_fn(a), f"{correct} (n={len(b)})": mask_fn(b), "kind": kind})
    for w in WORDS:
        add(w, lambda x, w=w: x[w].mean())
    add("neutral wording (none of the four word lists)", lambda x: (~x[list(WORDS)].any(axis=1)).mean())
    add("no URL (missing URL evidence)", lambda x: (x.condition == "no URL").mean())
    add("inaccessible webpage", lambda x: (x.condition == "inaccessible webpage").mean())
    add("accessible webpage", lambda x: (x.condition == "accessible webpage").mean())
    add("shortened link", lambda x: (x.link_shortened == 1).mean())
    for c in ["none", "gambling", "banking", "e-wallet", "payments_billing", "lending", "investment_crypto"]:
        add(f"category: {c}", lambda x, c=c: (x.category == c).mean())
    for lg in ["en", "taglish", "tl"]:
        add(f"language: {lg}", lambda x, lg=lg: (x.language == lg).mean())
    for s in sorted(d.source.unique()):
        add(f"source: {s}", lambda x, s=s: (x.source == s).mean())
    add("message length (tokens), mean", lambda x: x.n_tokens.mean(), "mean")
    add("Link RF's P (messages with a URL), mean", lambda x: x.rf_P.mean(), "mean")
    add("Link RF's P < 0.5 (RF said legitimate), share of messages with a URL", lambda x: (x.rf_P.dropna() < 0.5).mean())
    t = pd.DataFrame(rows)
    t["difference"] = t[f"{error} (n={len(a)})"] - t[f"{correct} (n={len(b)})"]
    return t.round(4)


def main() -> int:
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    data = pd.read_csv(dataset, keep_default_na=False).set_index("message_id")
    data["condition"] = np.where(data.has_url == 0, "no URL", np.where(data.page_fetched == 1, "accessible webpage", "inaccessible webpage"))
    for w, pattern in WORDS.items():
        data[w] = data.sms_text.str.contains(pattern, case=False, regex=True)
    data["n_tokens"] = [len(text_tokens(s)) for s in data.sms_text]
    data["rf_P"] = rf_p(dataset)

    models = {name: per_model_outcomes(step5, stem) for name, stem in MAIN}
    e2 = models["Proposed Model"]
    d = data.join(e2, how="inner")
    assert len(d) == len(data)
    for name, w in models.items():
        if name != "Proposed Model":
            d[f"{name}: seeds wrong"] = w["seeds_wrong"]

    counts = []
    for name, w in models.items():
        row = {"model": name}
        for o in ["TP", "TN", "FP", "FN"]:
            row[f"{o} (majority)"] = int((w.majority == o).sum())
            row[f"{o} in all 3 seeds"] = int(((w.majority == o) & w.consistent).sum())
        row["messages wrong in at least one seed"] = int((w.seeds_wrong > 0).sum())
        counts.append(row)
    counts = pd.DataFrame(counts)
    overlap = []
    for o in ["FN", "FP"]:
        mine = set(e2.index[(e2.majority == o)])
        for name, w in models.items():
            if name != "Proposed Model":
                overlap.append({"E2 error": o, "E2 messages": len(mine), "other model": name,
                                "also wrong (majority) by the other model": len(mine & set(w.index[w.majority == o])),
                                "share": round(len(mine & set(w.index[w.majority == o])) / max(1, len(mine)), 3)})
    overlap = pd.DataFrame(overlap)

    by_group = []
    for col in ["condition", "category", "language", "source"]:
        g = d.groupby(col)["majority"].value_counts().unstack(fill_value=0).reindex(columns=["TP", "FN", "TN", "FP"], fill_value=0)
        g["error rate"] = ((g.FN + g.FP) / g.sum(axis=1)).round(4)
        g.insert(0, "group", col)
        by_group.append(g.reset_index().rename(columns={col: "value"}))
    by_group = pd.concat(by_group, ignore_index=True)

    reps = []
    rng = np.random.default_rng(42)
    for o in ["TP", "TN", "FP", "FN"]:
        pool = d[(d.majority == o)]
        if o in ("FP", "FN"):
            pool = pool[pool.consistent].assign(conf=lambda x: (x.mean_p_smishing - 0.5).abs()).sort_values("conf", ascending=False)
            picked = pool.groupby("condition", group_keys=False).head(3).head(8)
        else:
            pool = pool[pool.consistent]
            picked = pd.concat([g.iloc[rng.permutation(len(g))[:3]] for _, g in pool.groupby("condition")]).head(8)
        for mid, r in picked.iterrows():
            reps.append({"outcome": o, "message_id": mid, "message (anonymised)": anonymise(r.sms_text, r),
                         "actual": r.label, "E2 mean P(smishing)": round(r.mean_p_smishing, 3),
                         "Link RF's P": None if pd.isna(r.rf_P) else round(r.rf_P, 3), "condition": r.condition,
                         "category": r.category, "language": r.language, "source": r.source,
                         **{w: bool(r[w]) for w in WORDS},
                         **{f"{n}: seeds wrong": int(r[f"{n}: seeds wrong"]) for n, _ in MAIN[1:]},
                         "reviewed by hand (fill in)": ""})
    reps = pd.DataFrame(reps)

    per_message = d.reset_index()[["message_id", "label", "condition", "category", "language", "source", "link_domain", "link_shortened",
                                   "fetch_status", "n_tokens", "rf_P", *WORDS, "seed_42", "seed_123", "seed_2024", "majority",
                                   "consistent", "mean_p_smishing", *[f"{n}: seeds wrong" for n, _ in MAIN[1:]]]]
    method = pd.DataFrame({"item": ["model analysed", "outcome per message", "consistent", "pattern tables", "word lists",
                                    "Link RF's P", "representative messages", "anonymisation", "manual review"], "value": [
        "Proposed Model (E2); Outcome_Counts also lists the other main models",
        "majority over seeds 42, 123, 2024 of TP / TN / FP / FN (Step 5 predictions; smishing = positive)",
        "the same outcome in all 3 seeds",
        "within smishing FN vs TP, within legitimate FP vs TN: share of messages with each characteristic, and the difference",
        "; ".join(f"{k}: {v}" for k, v in WORDS.items()),
        "the forest of each message's own run, its saved test score (experiments/c_prune/.../link_lane/final_*.csv), mean over seeds",
        "up to 8 per outcome; consistent ones; FP / FN the most confidently wrong, at most 3 per message condition; TP / TN 3 per condition at random (seed 42)",
        "link -> [link: domain]; phone numbers -> [PHONE]; numbers of 4+ digits -> [NUMBER]; emails -> [EMAIL]",
        "names and other personal details are NOT masked by rules: review every representative message by hand"]})

    linked = data[data.has_url == 1]
    mixed = linked.groupby("link_domain").label.agg(messages="size", smishing=lambda x: int((x == "smishing").sum()))
    mixed["legitimate"] = mixed.messages - mixed.smishing
    mixed = mixed[(mixed.smishing > 0) & (mixed.legitimate > 0)].sort_values("messages", ascending=False).reset_index()
    e2_err = d.groupby("link_domain")["majority"].apply(lambda x: int(x.isin(["FP", "FN"]).sum()))
    mixed["E2 errors (majority) on this domain"] = mixed.link_domain.map(e2_err).fillna(0).astype(int)
    mixed["note"] = ""
    gambling_ops = mixed.link_domain.str.contains("casinoplus|applus|arenaplus|bingoplus|bet|play", regex=True)
    mixed.loc[gambling_ops, "note"] = "gambling operator: the same sender is labelled both ways; needs ONE annotation rule"
    platform = mixed.link_domain.isin(["onelink.me", "google.com", "facebook.com", "page.link", "x.com", "discord.com", "zoom.us",
                                       "wikipedia.org", "bbc.com", "amazon.com", "microsoftonline.com"])
    mixed.loc[platform, "note"] = "shared platform: both labels can be right (scams also use these hosts); check the minority label"
    lab = data[data.sms_text.str.contains(r"cloudapp\.azure\.com|lejet", case=False, regex=True)]
    lab_rows = pd.DataFrame({"link_domain": "(lab test record?)", "messages": len(lab), "smishing": int((lab.label == "smishing").sum()),
                             "legitimate": int((lab.label == "legitimate").sum()), "E2 errors (majority) on this domain": "",
                             "note": "message is only the group's own Azure lab URL: " + ", ".join(lab.index) + " (likely not an SMS)"},
                            index=[0]) if len(lab) else pd.DataFrame()
    label_issues = pd.concat([mixed, lab_rows], ignore_index=True)

    out = fm.out_dir(dataset, "step9")
    out.mkdir(parents=True, exist_ok=True)
    fn_tp, fp_tn = pattern_table(d[d.label == "smishing"], "FN", "TP"), pattern_table(d[d.label == "legitimate"], "FP", "TN")
    with pd.ExcelWriter(out / "Classification_Error_Analysis.xlsx", engine="openpyxl") as w:
        counts.to_excel(w, sheet_name="Outcome_Counts", index=False)
        overlap.to_excel(w, sheet_name="Outcome_Counts", index=False, startrow=len(counts) + 3)
        fn_tp.to_excel(w, sheet_name="Patterns_FN_vs_TP", index=False)
        fp_tn.to_excel(w, sheet_name="Patterns_FP_vs_TN", index=False)
        by_group.to_excel(w, sheet_name="By_Group", index=False)
        reps.to_excel(w, sheet_name="Representative", index=False)
        per_message.to_excel(w, sheet_name="Per_Message", index=False)
        label_issues.to_excel(w, sheet_name="Possible_Label_Issues", index=False)
        method.to_excel(w, sheet_name="Method", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 150)
    print(counts.to_string(index=False))
    print(overlap.to_string(index=False))
    print(fn_tp.to_string(index=False))
    print(fp_tn.to_string(index=False))
    print(by_group.to_string(index=False))
    print(reps[["outcome", "message (anonymised)", "E2 mean P(smishing)", "Link RF's P", "condition", "category"]].to_string(index=False))
    print(f"wrote {out / 'Classification_Error_Analysis.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
