"""The Le-Jet serving layer: SMS text in, the frontend's /predict answer out, using the FINAL model
(CNN over the text + the two frozen RF extractors, routed by whether the page was inspected, gated, fused).

What this code does NOT do: it never opens, fetches or resolves a link. Page evidence comes only from a
store of links the lab already inspected in the isolated VMs (data/live/live_links.csv, keyed by the sha256
of the normalised link). A link that is not in the store counts as "page not inspected" and is scored by the
url_only extractor from its spelling alone. This is the same routing the thesis describes (page_fetched).

Models (ids match frontend/src/data/researchModels.js):
    film_cnn            the hybrid CNN–Random Forest model with Feature-wise Linear Modulation (FiLM) of the study (src/serving/film_model.py); served only when its files exist (the default then)
    fusion_gated        the proposed model: models/fusion/rf_vec/phase_c_seed<seed>.pt + gates
    cnn_only_baseline   the CNN alone: models/fusion/rf_vec/phase_a_seed<seed>.pt (Phase A of the same run)
Other ids answer "unknown_model" and the page shows them as "not available yet".
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ARTIFACTS_ROOT, DENSE_UNITS, NUM_FILTERS, SEQ_LEN_L
from src.features.link_features import (
    LINK_TEXT_FEATURES, extract_links, link_text_features, no_link_features, normalize_link,
)
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier
from src.features.model.fusion_data import RECIPE_COLUMNS, aux_from_inputs, load_embedding_layer, load_vocab
from src.features.model.rf_extractor import PAGE_FETCHED, VARIANTS, load_extractor
from src.features.model.rf_feature_extractor import extract_vector, route_vector
from src.features.text_preprocessing import pad_or_truncate, text_tokens
from src.serving.film_model import FilmModel
from src.serving.explain_text import explain_message
from src.serving.scope import assess_scope

SMISHING = 1
RECIPE = "rf_vec"
FUSION_ID, CNN_ID, FILM_ID = "fusion_gated", "cnn_only_baseline", "film_cnn"
MODEL_IDS = (FUSION_ID, CNN_ID)                       # always served; FILM_ID comes first in LejetService.model_ids when the FiLM model is loaded
PAGE_COLUMNS = VARIANTS["full"][len(LINK_TEXT_FEATURES):]          # page_rank, 12 webpage, 4 external
LIVE_LINKS = ARTIFACTS_ROOT / "data" / "live" / "live_links.csv"
DEFAULT_SHOT_DIRS = (Path(r"D:\LeJetData\clean\shots"),)
SHOT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
MAX_TEXT_CHARS = 2000

# Page features worth showing a person (name -> short plain label); -1 means "not available" and is hidden.
PAGE_FACTS = {
    "page_rank": "Open PageRank score (0-10)", "login_form": "Page asks for a login/password",
    "nb_hyperlinks": "Links on the page", "iframe": "Hidden frame on the page",
    "empty_title": "Page has no title", "domain_in_title": "Domain name appears in the page title",
    "external_favicon": "Icon is loaded from another site", "domain_age": "Domain age (days)",
    "domain_registration_length": "Registration length (days)", "dns_record": "Domain has NO DNS record (1 = no record, 0 = normal)",
    "whois_registered_domain": "Domain has NO registration record (1 = none, 0 = normal)",
}


def percent_floor(p: float) -> float:
    """Percent with 2 decimals, never rounded up, never 100 (the frontend's convention)."""
    return min(99.99, math.floor(max(0.0, p) * 10000) / 100)


def risk_level(prob_smishing: float) -> str:
    return "high" if prob_smishing >= 0.8 else "medium" if prob_smishing >= 0.5 else "low"


def link_key(normalised: str) -> str:
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def url_reasons(f: dict, raw_url: str) -> list[str]:
    """Plain sentences about the link's spelling. The wording starts with the prefixes the frontend maps
    to translated labels (frontend/src/data/backendClient.js URL_REASON_PATTERNS)."""
    out = []
    if f.get("https_presence") == 0:
        out.append("Link does not use HTTPS")
    if f.get("shortened_url_indicator") == 1:
        out.append("Link uses a shortened-URL service")
    if f.get("has_ip_address") == 1:
        out.append("Link uses a raw IP address")
    if f.get("number_of_hyphens", 0) >= 3:
        out.append("Domain contains multiple hyphens")
    if f.get("subdomain_count", 0) >= 3:
        out.append("Link has an unusually high number of subdomains")
    if f.get("suspicious_keyword_count", 0) >= 1:
        out.append("Link contains suspicious keywords")
    if f.get("suspicious_url_token_count", 0) >= 1:
        out.append("Link path contains suspicious wording")
    if f.get("special_char_count", 0) >= 6:
        out.append("Link contains an unusual number of special characters")
    return out


class LinkStore:
    """Links the lab already inspected: features, where the link led, a screenshot if one was taken."""

    def __init__(self, csv: Path = LIVE_LINKS, shot_dirs=DEFAULT_SHOT_DIRS, on_demand_csv: Path | None = None):
        self.rows: dict[str, dict] = {}
        self.shot_dirs = [Path(d) for d in shot_dirs]
        self._shots: dict[str, Path] = {}
        self._lock = threading.Lock()
        for path in (csv, on_demand_csv):                    # on-demand rows come from live inspections in the demo
            if path and Path(path).exists():
                frame = pd.read_csv(path, keep_default_na=False, dtype={"link_id": str})
                self.rows.update({r["key"]: r for r in frame.to_dict("records")})
        for shot_dir in self.shot_dirs:
            if shot_dir.exists():
                for png in shot_dir.glob("*/*.png"):
                    if SHOT_ID_RE.match(png.stem):
                        self._shots.setdefault(png.stem, png)

    def add(self, record: dict, csv: Path, shots_dir: Path | None = None) -> None:
        """Keep one freshly inspected link: in memory now, and in the on-demand file for the next start.
        Rows are normalised through the CSV so their types equal those of rows loaded at start-up."""
        with self._lock:
            csv = Path(csv)
            csv.parent.mkdir(parents=True, exist_ok=True)
            old = (pd.read_csv(csv, keep_default_na=False, dtype={"link_id": str})
                   if csv.exists() else pd.DataFrame())
            frame = pd.concat([old, pd.DataFrame([record])], ignore_index=True).drop_duplicates("key", keep="last")
            frame.to_csv(csv, index=False)
            fresh = pd.read_csv(csv, keep_default_na=False, dtype={"link_id": str})
            self.rows[record["key"]] = fresh[fresh["key"] == record["key"]].to_dict("records")[-1]
            if shots_dir and Path(shots_dir).exists():
                for png in Path(shots_dir).glob("*.png"):
                    if SHOT_ID_RE.match(png.stem):
                        self._shots[png.stem] = png

    def get(self, normalised_link: str) -> dict | None:
        return self.rows.get(link_key(normalised_link))

    def shot_path(self, link_id: str) -> Path | None:
        return self._shots.get(link_id) if SHOT_ID_RE.match(link_id or "") else None

    def has_shot(self, link_id: str) -> bool:
        return self.shot_path(link_id) is not None

    def __len__(self) -> int:
        return len(self.rows)


class LejetService:
    def __init__(self, seed: int = 42, fusion_dir: Path | None = None, extractor_dir: Path | None = None,
                 store: LinkStore | None = None, film: FilmModel | None = None):
        fusion_dir = Path(fusion_dir or ARTIFACTS_ROOT / "models" / "fusion" / RECIPE)
        extractor_dir = Path(extractor_dir or ARTIFACTS_ROOT / "models" / "rf_extractor")
        self.seed = seed
        self.columns = RECIPE_COLUMNS[RECIPE]
        self.word_to_idx = load_vocab()
        self.extractors = {v: load_extractor(v, seed, extractor_dir) for v in ("url_only", "full")}
        self.store = store if store is not None else LinkStore()
        self.lock = threading.Lock()           # the explanation pass writes .grad on the shared model
        self.film = film
        self.model_ids = ((FILM_ID,) if film else ()) + MODEL_IDS          # the first one answers when the request names none

        gates_csv = pd.read_csv(fusion_dir / f"gates_seed{seed}.csv")
        gates = torch.tensor(gates_csv["gate"].to_numpy(dtype="float32"))
        self.gate_table = gates_csv.iloc[NUM_FILTERS:].reset_index(drop=True)

        embedding, _ = load_embedding_layer()
        self.cnn = TextEncoder(embedding, NUM_FILTERS)
        self.cnn.load_state_dict(torch.load(fusion_dir / f"phase_a_seed{seed}.pt", map_location="cpu"))
        self.cnn.eval()
        embedding, _ = load_embedding_layer()
        self.model = FusionClassifier(TextEncoder(embedding, NUM_FILTERS), gates[:NUM_FILTERS], gates[NUM_FILTERS:],
                                      url_dim=len(self.columns), dense_units=DENSE_UNITS)
        self.model.load_state_dict(torch.load(fusion_dir / f"phase_c_seed{seed}.pt", map_location="cpu"))
        self.model.eval()

    # ------------------------------------------------------------------ helpers
    def _tokens(self, text: str) -> tuple[list[str], list[str]]:
        every = text_tokens(text)
        known = [t for t in every if t in self.word_to_idx and t != "<PAD>"]
        return every, pad_or_truncate(known, SEQ_LEN_L)

    def _row(self, normalised: str | None, stored: dict | None) -> dict:
        """The extractor inputs for one message: link spelling always; page evidence only from the store."""
        fetched = bool(stored and int(stored.get("page_fetched", 0)) == 1)
        row = dict(link_text_features(normalised) if normalised else no_link_features())
        for name in PAGE_COLUMNS:
            row[name] = stored[name] if fetched else -1
        row["has_url"] = int(normalised is not None)
        row[PAGE_FETCHED] = int(fetched)
        return row

    def _key_indicators(self, top: int = 8) -> list[dict]:
        table = self.gate_table.sort_values("gate", ascending=False)
        out = []
        for _, r in table.iterrows():
            name = str(r["feature"])
            if name in ("has_url", "P", "u"):
                continue
            plain = name[2:] if name.startswith("c_") else name
            out.append({"feature": plain, "label": plain.replace("_", " "), "gate": float(r["gate"]),
                        "gated": float(r["gate"]) > 0, "importance": float(r["gate"])})
            if len(out) == top:
                break
        return out

    def _inspection(self, normalised: str | None, stored: dict | None, row: dict, used: str,
                    outputs: dict, forest: str = "url_only extractor", page_used: bool = True) -> dict:
        if not normalised:
            return {"state": "no_link", "note": "No link in the message: judged on the text alone."}
        info = {"state": "not_inspected", "extractor": used,
                "extractors": {n: {"P": round(float(o.iloc[0]["P"]), 4), "u": round(float(o.iloc[0]["u"]), 4)}
                               for n, o in outputs.items()},
                "note": "This link is not in the lab's store of inspected pages, so no page was seen. "
                        f"It is scored from how the link is spelled ({forest})."}
        if not stored:
            return info
        info.update({"link_id": stored["link_id"], "fetch_status": stored.get("fetch_status", ""),
                     "batch": stored.get("batch_id", ""), "visited_at": stored.get("visited_at", ""),
                     "has_screenshot": self.store.has_shot(stored["link_id"])})
        if int(stored.get("page_fetched", 0)) == 1:
            facts = [{"feature": k, "label": PAGE_FACTS[k], "value": row[k]}
                     for k in PAGE_FACTS if row.get(k, -1) != -1]
            if used == "link_rf":
                seen = ("What it saw went to the Random Forest together with the link's spelling; the forest's score and certainty then "
                        "adjusted the CNN's filters." if page_used else
                        "This lab result lacks a column the Random Forest needs (safe_anchor), so the page was treated as not seen: "
                        "the forest judged the link from its spelling and the domain lookups alone.")
            else:
                seen = "What it saw went to the full extractor (page_fetched = 1)."
            info.update({"state": "inspected", "final_url": stored.get("final_url", ""),
                         "final_domain": stored.get("final_domain", ""), "redirects": stored.get("n_redirects", 0),
                         "page_facts": facts, "features": {k: row[k] for k in PAGE_COLUMNS},
                         "note": f"The lab opened this page earlier inside an isolated virtual machine. {seen}"})
        else:
            info.update({"state": "unreachable",
                         "note": "The lab tried this link but could not load a page "
                                 f"({stored.get('fail_reason', 'unreachable')}). Scored from its spelling ({forest.split(' (')[0]})."})
        return info

    def _score_fusion(self, text: str, frame: pd.DataFrame, row: dict, model_id: str, explain: bool) -> dict:
        """The earlier design: the two frozen extractors, then the gated fusion head (or the CNN alone)."""
        forest_started = time.perf_counter()
        outputs = {"url_only": extract_vector(self.extractors["url_only"], frame)}
        if row[PAGE_FETCHED]:
            outputs["full"] = extract_vector(self.extractors["full"], frame)
        used = "full" if row[PAGE_FETCHED] else "url_only"
        aux = torch.tensor(aux_from_inputs(frame, outputs[used], RECIPE).to_numpy(dtype="float32"))
        link_forest_ms = (time.perf_counter() - forest_started) * 1000

        prep_started = time.perf_counter()
        every, tokens = self._tokens(text)
        ids = torch.tensor([[self.word_to_idx[t] for t in tokens]], dtype=torch.long)
        text_prep_ms = (time.perf_counter() - prep_started) * 1000
        started = time.perf_counter()
        with self.lock:
            with torch.no_grad():
                if model_id == CNN_ID:
                    prob = torch.softmax(self.cnn(ids), dim=1)[0, SMISHING].item()
                else:
                    prob = self.model.predict_proba(ids, aux)[0, SMISHING].item()
            inference_ms = (time.perf_counter() - started) * 1000
            predicted = SMISHING if prob > 0.5 else 0
            explain_started = time.perf_counter()
            explanation = (explain_message(self.model, ids, aux, tokens, predicted, max_reasons=3,
                                           raw_text=text, url_names=self.columns)
                           if model_id == FUSION_ID and explain else
                           {"reasons": [], "spans": [], "cnn_evidence": None, "url_evidence": None})
            explanation_ms = (time.perf_counter() - explain_started) * 1000
        return {"prob": prob, "explanation": explanation, "every": every, "tokens": tokens, "word_to_idx": self.word_to_idx,
                "link_forest_ms": link_forest_ms, "text_prep_ms": text_prep_ms, "inference_ms": inference_ms,
                "explanation_ms": explanation_ms, "outputs": outputs, "used": used, "forest": "url_only extractor",
                "page_used": True, "source": "fusion_model"}

    def _score_film(self, text: str, normalised: str | None, stored: dict | None, explain: bool) -> dict:
        """The FiLM model: the Link RF's P and u (and whether a page was seen) go to the generator; a message with no link bypasses it."""
        film = self.film
        forest_started = time.perf_counter()
        p = u = 0.0
        page_used = False
        link_inputs = None
        if normalised:
            features, page_used = film.link_features(normalised, stored)
            p, u = film.link_numbers(features)
            link_inputs = film.link_inputs(features, page_used, stored)
        link = film.link_tensor(p, u, 0 if page_used else 1, int(normalised is not None))
        link_forest_ms = (time.perf_counter() - forest_started) * 1000

        prep_started = time.perf_counter()
        every, tokens = film.tokens(text)
        ids = film.token_ids(text)
        text_prep_ms = (time.perf_counter() - prep_started) * 1000
        started = time.perf_counter()
        with self.lock:
            prob = film.p_smishing(ids, link)
            inference_ms = (time.perf_counter() - started) * 1000
            predicted = SMISHING if prob > 0.5 else 0
            explain_started = time.perf_counter()
            explanation = (film.explain(ids, link, tokens, predicted, text) if explain else
                           {"reasons": [], "spans": [], "cnn_evidence": None, "url_evidence": None,
                            "link_effect": film.link_effect(ids, link)})
            explanation_ms = (time.perf_counter() - explain_started) * 1000
        outputs = {"link_rf": pd.DataFrame([{"P": p, "u": u}])} if normalised else {}
        return {"prob": prob, "explanation": explanation, "every": every, "tokens": tokens, "word_to_idx": film.word_to_idx,
                "link_forest_ms": link_forest_ms, "text_prep_ms": text_prep_ms, "inference_ms": inference_ms,
                "explanation_ms": explanation_ms, "outputs": outputs, "used": "link_rf", "forest": "Link RF, page hidden",
                "page_used": page_used, "source": "film_model", "link_inputs": link_inputs}

    # ------------------------------------------------------------------ main entry
    def predict(self, message: str, model_id: str | None = None, explain: bool = True) -> dict:
        model_id = model_id or self.model_ids[0]
        if model_id not in self.model_ids:
            raise ValueError("unknown_model")
        call_started = time.perf_counter()
        text = message[:MAX_TEXT_CHARS]
        links = extract_links(text)
        normalised = normalize_link(links[0]["raw"]) if links else None
        stored = self.store.get(normalised) if normalised else None
        row = self._row(normalised, stored)
        frame = pd.DataFrame([row])

        scored = (self._score_film(text, normalised, stored, explain) if model_id == FILM_ID
                  else self._score_fusion(text, frame, row, model_id, explain))
        prob, explanation = scored["prob"], scored["explanation"]
        predicted = SMISHING if prob > 0.5 else 0
        label = "smishing" if predicted == SMISHING else "legitimate"
        confidence = prob if predicted == SMISHING else 1.0 - prob
        link_feats = {k: row[k] for k in LINK_TEXT_FEATURES}
        link_feats["has_url"] = row["has_url"]
        all_urls = [normalize_link(l["raw"]) for l in links]
        inference_ms = scored["inference_ms"]

        return {
            "label": label,
            "confidence_percent": percent_floor(confidence),
            "detected_url": normalised,
            "smishing_probability": percent_floor(prob),
            "legitimate_probability": percent_floor(1.0 - prob),
            "risk_level": risk_level(prob),
            "model_id": model_id,
            "source": scored["source"],
            "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "urls": all_urls,
            "safe_browsing": None,
            "text_reasons": explanation["reasons"],
            "text_reason_spans": explanation["spans"],
            "cnn_evidence": explanation["cnn_evidence"],
            "url_evidence": explanation["url_evidence"],
            "link_effect": explanation.get("link_effect"),
            "link_inputs": scored.get("link_inputs"),               # the 37 numbers the Link RF read (hybrid model only)
            "url_reasons": url_reasons(link_feats, normalised or "") if normalised else [],
            "cleaned_text": " ".join(scored["tokens"]).replace("<PAD>", "").strip(),
            "url_features": link_feats,
            "global_key_indicators": self._key_indicators() if model_id == FUSION_ID else [],
            "jain_route": None,
            "model_info": self.film.info if model_id == FILM_ID else None,
            "inference_ms": round(inference_ms, 2),
            # Where this call's time went (ms), for the frontend's Efficiency card. The page inspection is NOT in here: it happens
            # earlier, in the isolated lab (GET /inspect/<job>), and takes seconds. model_ms is inference_ms.
            "timings": {"link_forest_ms": round(scored["link_forest_ms"], 2), "text_prep_ms": round(scored["text_prep_ms"], 2),
                        "model_ms": round(inference_ms, 2), "explanation_ms": round(scored["explanation_ms"], 2),
                        "total_ms": round((time.perf_counter() - call_started) * 1000, 2), "has_link": normalised is not None},
            "scope": assess_scope(text, scored["every"], scored["word_to_idx"], label),
            "inspection": self._inspection(normalised, stored, row, scored["used"], scored["outputs"],
                                           scored["forest"], scored["page_used"]),
        }

    def evidence_png(self, link_id: str) -> bytes | None:
        path = self.store.shot_path(link_id)
        return path.read_bytes() if path else None
