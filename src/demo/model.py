"""The demo model behind the web page: SMS text in, verdict and evidence out.

PROTOTYPE ONLY: trained on synthetic data (src/demo/synthetic.py, train_demo.py).

The page-inspection step is SIMULATED. This code never opens or fetches a link. With
`inspect=True`, a link that appears in the demo data's stored page table is treated as a page the
inspector reached, and its stored (real, public-dataset) page features go to the `full`
extractor. Any other link is treated as unreachable and goes to `url_only`.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import DENSE_UNITS, NUM_FILTERS, SEQ_LEN_L
from src.demo.synthetic import DEMO_DATA, DEMO_MODEL_DIR, PAGE_COLUMNS
from src.features.link_features import LINK_TEXT_FEATURES, extract_links, link_text_features, no_link_features, normalize_link
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier
from src.features.model.fusion_data import aux_from_inputs, load_embedding_layer, load_vocab
from src.features.model.rf_feature_extractor import extract_vector
from src.features.text_preprocessing import pad_or_truncate, text_tokens
from src.serving.explain_text import explain_message

SMISHING = 1
URL_FEATURES_SHOWN = 8


class DemoModel:
    def __init__(self, model_dir: Path = DEMO_MODEL_DIR, data_path: Path = DEMO_DATA):
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.recipe = self.meta.get("recipe", "link_pu")
        self.aux_columns = self.meta["aux_columns"]
        self.word_to_idx = load_vocab()
        self.extractors = {v: joblib.load(model_dir / "extractors" / f"{v}.joblib") for v in ("url_only", "full")}
        gates = torch.from_numpy(np.load(model_dir / "gates_seed42.npy").astype(np.float32))

        embedding, _ = load_embedding_layer()
        self.cnn = TextEncoder(embedding, NUM_FILTERS)
        self.cnn.load_state_dict(torch.load(model_dir / "phase_a.pt", map_location="cpu"))
        self.cnn.eval()

        embedding, _ = load_embedding_layer()
        self.model = FusionClassifier(TextEncoder(embedding, NUM_FILTERS), gates[:NUM_FILTERS], gates[NUM_FILTERS:],
                                      url_dim=len(self.aux_columns), dense_units=DENSE_UNITS)
        self.model.load_state_dict(torch.load(model_dir / "phase_c.pt", map_location="cpu"))
        self.model.eval()

        data = pd.read_csv(data_path, keep_default_na=False)
        self.data = data
        fetched = data[data["page_fetched"] == 1]
        self.pages = {normalize_link(r["link"]): {c: r[c] for c in PAGE_COLUMNS} for _, r in fetched.iterrows()}

    def _tokens(self, text: str) -> list[str]:
        known = [t for t in text_tokens(text) if t in self.word_to_idx and t != "<PAD>"]
        return pad_or_truncate(known, SEQ_LEN_L)

    def analyze(self, text: str, inspect: bool = True) -> dict:
        links = extract_links(text)
        url = normalize_link(links[0]["raw"]) if links else None
        page = self.pages.get(url) if (inspect and url) else None
        row = {**(link_text_features(url) if url else no_link_features()),
               **{c: (page[c] if page else -1) for c in PAGE_COLUMNS},
               "has_url": int(url is not None), "page_fetched": int(page is not None)}
        frame = pd.DataFrame([row])

        outputs = {"url_only": extract_vector(self.extractors["url_only"], frame)}
        if page:
            outputs["full"] = extract_vector(self.extractors["full"], frame)
        used = "full" if page else "url_only"
        aux = torch.tensor(aux_from_inputs(frame, outputs[used], self.recipe).to_numpy(dtype="float32"))

        tokens = self._tokens(text)
        ids = torch.tensor([[self.word_to_idx[t] for t in tokens]], dtype=torch.long)
        with torch.no_grad():
            prob = self.model.predict_proba(ids, aux)[0, SMISHING].item()
            text_only = torch.softmax(self.cnn(ids), dim=1)[0, SMISHING].item()
        predicted = SMISHING if prob >= 0.5 else 0
        explanation = explain_message(self.model, ids, aux, tokens, predicted, max_reasons=3,
                                      raw_text=text, url_names=self.aux_columns)

        url_evidence = explanation["url_evidence"]
        shown = [f for f in url_evidence["features"] if f["feature"] != "has_url"][:URL_FEATURES_SHOWN]
        return {
            "verdict": "smishing" if predicted == SMISHING else "legitimate",
            "prob_smishing": round(prob, 4), "text_only_prob_smishing": round(text_only, 4),
            "link": url, "page_inspected": page is not None, "extractor_used": used if url else None,
            "extractors": {name: {"P": round(float(f.iloc[0]["P"]), 4), "u": round(float(f.iloc[0]["u"]), 4)}
                           for name, f in outputs.items()} if url else {},
            "rf_breakdown": self._breakdown(outputs[used].iloc[0], row) if url else [],
            "reasons": explanation["reasons"], "spans": explanation["spans"],
            "link_evidence": {"features": shown, "url_path_share": url_evidence["url_path_share"],
                              "url_total": url_evidence["url_total"], "text_total": url_evidence["text_total"]},
            "notes": self._notes(url, inspect, page),
        }

    @staticmethod
    def _breakdown(rf_row: pd.Series, row: dict, top: int = 8) -> list[dict]:
        """How the forest scored the link: each input's push on P, in probability points. A feature
        the forest did not read or did not split on for this link shows nothing."""
        parts = [(name[2:], rf_row[name] * 100) for name in rf_row.index if name.startswith("c_") and rf_row[name] != 0]
        parts.sort(key=lambda p: -abs(p[1]))
        return [{"feature": f, "value": float(row[f]), "points": round(float(points), 1)} for f, points in parts[:top]]

    @staticmethod
    def _notes(url, inspect, page) -> list[str]:
        if not url:
            return ["No link found: the message is judged on its text alone (link inputs are -1)."]
        if not inspect:
            return ["Page inspection is off: the link is scored from its spelling only (url_only extractor)."]
        if page is None:
            return ["Page not in the demo's stored table, so it counts as unreachable: url_only extractor."]
        return ["Simulated inspection: stored page and domain features for this link went to the full extractor."]

    def examples(self) -> list[dict]:
        test = self.data[self.data["split"] == "test"]
        wanted = [("ambiguous", "smishing", 1, 4), ("ambiguous", "legitimate", 1, 4),
                  ("class_specific", "smishing", 1, 2), ("class_specific", "legitimate", 1, 2),
                  ("ambiguous", "smishing", 0, 2), ("no_link", "smishing", 0, 1), ("no_link", "legitimate", 0, 1)]
        out = []
        for kind, label, fetched, n in wanted:
            pick = test[(test["text_kind"] == kind) & (test["label"] == label)
                        & ((test["page_fetched"] == fetched) | (kind == "no_link"))].head(n)
            out += [{"text": r["sms_text"], "label": label, "kind": kind, "page_fetched": int(r["page_fetched"])}
                    for _, r in pick.iterrows()]
        return out
