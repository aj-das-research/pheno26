"""Faithful, dataset-agnostic implementation of the MESHAgents 3-stage protocol (MICCAI 2025).

Maps 1:1 onto the paper (arXiv:2507.03460v2, §2):

  Tools T          -> `Tools`            : statistical significance tests, effect-size, distribution.
  Dynamic Memory   -> `AgentMemory`      : M_i with embedding retrieval R_i = argmax_h Sim(Emb(X),Emb(h)).
  Stage I          -> `stage1_valuation` : V_i = {phi_i(p_k, R_i)} = significance + clinical-relevance(LLM)
                                           + population-distribution + stability.
  Stage II         -> `stage2_factors`   : E_i = {(p_n,c_m,psi_i)} local + coordinator E_G = h({E_i});
                                           identifies confounders (factor associated with phenotype AND outcome).
  Stage III        -> `stage3_consensus` : sequential discussion O_i^t = g_i(V_i, H_{t-1}, M_i),
                                           coordinator-guided, <=10 rounds, until convergence.
  Aggregation f_AP -> `f_AP`             : A = f_AP({O^t},{R_i},TR), weighting opinions by statistical
                                           confidence (p<0.05) and on-topic relevance (>0.3).
  Auto-PheWAS      -> `dependency`,`coverage` : Q(P) and C(P) exactly as defined in §3.

Per the project decision, Stage III opinions are produced by an LLM (paper-exact). The model client is
swappable for the HPP TRE: set MESH_LLM_BASE_URL / MESH_LLM_MODEL to a local open-weight endpoint.
MESH_DRY_RUN=1 substitutes a statistical opinion stub *for development/testing only* (no API calls).
"""
import os
import re
import json
import logging
import hashlib
from collections import defaultdict
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("MESH_DRY_RUN", "0") == "1"          # testing-only: skip LLM, use statistical stub
MAX_ROUNDS = int(os.getenv("MESH_MAX_ROUNDS", "10"))     # paper: typically converges within 10 rounds
CONVERGE_JACCARD = float(os.getenv("MESH_CONVERGE_JACCARD", "0.9"))
SIG_THRESHOLD = 0.05                                     # f_AP statistical confidence (paper: <0.05)
RELEVANCE_THRESHOLD = 0.3                                # f_AP on-topic relevance (paper: >0.3)


# ============================================================ model client (LLM) + embeddings
class ModelClient:
    """Chat client; OpenAI by default, or any OpenAI-compatible endpoint (set MESH_LLM_BASE_URL)."""

    def __init__(self):
        from config import GPT_MODEL
        import llm_provider as P                         # provider switch (OpenAI <-> Ollama)
        self.model = P.chat_model(default=GPT_MODEL)
        self.client = P.chat_client()

    def chat_json(self, system: str, user: str, retries: int = 2) -> Dict[str, Any]:
        """Return a parsed JSON object from the model, with a strict-JSON request + robust fallback."""
        last = ""
        for attempt in range(retries + 1):
            try:
                kwargs = dict(model=self.model,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
                try:
                    resp = self.client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs)
                except Exception:
                    resp = self.client.chat.completions.create(**kwargs)   # endpoint may not support json mode
                last = resp.choices[0].message.content or ""
                return _parse_json(last)
            except Exception as e:
                logger.warning(f"chat_json attempt {attempt} failed: {e}")
        logger.error(f"chat_json: could not parse model output; got: {last[:200]!r}")
        return {}


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)       # extract first {...} block
        if m:
            return json.loads(m.group(0))
        raise


class Embedder:
    """Emb(.) for dynamic memory. OpenAI embeddings when available, deterministic hashing fallback else."""

    def __init__(self, dim: int = 512):
        import llm_provider as P                         # provider switch (OpenAI <-> Ollama)
        self.dim = dim
        self.model = P.embed_model()
        self._client = None
        if not DRY_RUN:
            try:
                self._client = P.embed_client()
            except Exception:
                self._client = None

    def embed(self, text: str) -> np.ndarray:
        if self._client is not None:
            try:
                v = self._client.embeddings.create(model=self.model, input=text).data[0].embedding
                return np.asarray(v, dtype=float)
            except Exception as e:
                logger.debug(f"embedding API failed, using hashing fallback: {e}")
        v = np.zeros(self.dim)
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:                              # API vs fallback dim mismatch -> no match
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ============================================================ Tools  T = {T_1..T_J}
class Tools:
    """Statistical tools the agents apply to the data (paper: significance, effect size, distribution)."""

    @staticmethod
    def association(x: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Phenotype<->factor/outcome association. Pearson r (point-biserial when y is binary)."""
        d = pd.concat([pd.to_numeric(x, errors="coerce"),
                       pd.to_numeric(y, errors="coerce")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        d.columns = ["x", "y"]
        if len(d) < 10 or d["x"].std() == 0 or d["y"].std() == 0:
            return {"effect": 0.0, "abs_effect": 0.0, "p": 1.0, "n": int(len(d))}
        r, p = stats.pearsonr(d["x"], d["y"])
        return {"effect": float(r), "abs_effect": float(abs(r)), "p": float(p), "n": int(len(d))}

    @staticmethod
    def distribution(x: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(x, errors="coerce").dropna()
        if len(s) < 8 or s.std() == 0:
            return {"skew": 0.0, "kurtosis": 0.0, "normal_p": 0.0}
        try:
            _, normal_p = stats.normaltest(s)
        except Exception:
            normal_p = 0.0
        return {"skew": float(stats.skew(s)), "kurtosis": float(stats.kurtosis(s)),
                "normal_p": float(normal_p)}

    @staticmethod
    def stability(x: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(x, errors="coerce").dropna()
        if len(s) < 8 or s.std() == 0:
            return {"stability": 0.0, "cv": 0.0, "outlier_ratio": 1.0}
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
        cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0.0
        return {"stability": float(1.0 - out), "cv": float(cv), "outlier_ratio": float(out)}


# ============================================================ Dynamic Memory  M_i
class AgentMemory:
    """Long-term memory bank M_i with similarity-based retrieval R_i = argmax_h Sim(Emb(X), Emb(h))."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.cases: List[Dict[str, Any]] = []

    def store(self, key_text: str, payload: Dict[str, Any]) -> None:
        self.cases.append({"emb": self.embedder.embed(key_text), "text": key_text, "payload": payload})

    def retrieve(self, query_text: str) -> Optional[Dict[str, Any]]:
        if not self.cases:
            return None
        q = self.embedder.embed(query_text)
        sims = [_cosine(q, c["emb"]) for c in self.cases]
        i = int(np.argmax(sims))
        return {"similarity": sims[i], "text": self.cases[i]["text"], "payload": self.cases[i]["payload"]}


# ============================================================ region/measure parsing (body-comp)
def parse_region_measure(col: str) -> (Optional[str], Optional[str]):
    regions = sorted(["android", "gynoid", "arm_left", "arm_right", "arms", "leg_left", "leg_right",
                      "legs", "trunk", "total"], key=len, reverse=True)
    if col.startswith("body_comp_"):
        rest = col[len("body_comp_"):]
        for r in regions:
            if rest.startswith(r + "_"):
                return r, rest[len(r) + 1:]
        return None, rest
    if col.startswith("total_scan_"):
        return "whole_body_adipose", col[len("total_scan_"):]
    return None, None


# ============================================================ Stage I  V_i = {phi_i(p_k, R_i)}
def stage1_valuation(region: str, phenotypes: List[str], data: pd.DataFrame, tools: Tools,
                     memory: AgentMemory, model: Optional[ModelClient],
                     use_memory: bool = True) -> Dict[str, Dict[str, Any]]:
    """phi_i incorporates significance, clinical relevance (LLM), population distribution, and stability."""
    retrieved = memory.retrieve(f"{region}: " + " ".join(phenotypes)) if use_memory else None
    relevance = _clinical_relevance(region, phenotypes, model)     # LLM judgment in [0,1] per phenotype
    V: Dict[str, Dict[str, Any]] = {}
    for p in phenotypes:
        x = data[p]
        dist = tools.distribution(x)
        stab = tools.stability(x)
        signal = 0.0 if pd.to_numeric(x, errors="coerce").std() in (0.0, np.nan) else 1.0
        rel = float(relevance.get(p, 0.5))
        skew_pen = 1.0 / (1.0 + abs(dist["skew"]))                  # well-behaved distribution -> ~1
        val = 0.30 * signal + 0.30 * rel + 0.25 * stab["stability"] + 0.15 * skew_pen
        V[p] = {"valuation": float(val), "relevance": rel, "stability": stab["stability"],
                "skew": dist["skew"], "signal": signal}
    if use_memory:
        memory.store(f"{region}: " + " ".join(phenotypes),
                     {"stage": "I", "valuations": {p: V[p]["valuation"] for p in V}})
    if retrieved:
        logger.debug(f"[{region}] memory retrieval sim={retrieved['similarity']:.3f}")
    return V


def _clinical_relevance(region: str, phenotypes: List[str], model: Optional[ModelClient]) -> Dict[str, float]:
    """Per-phenotype clinical-relevance score in [0,1] from the specialist agent (one LLM call/region)."""
    if DRY_RUN or model is None:
        return {p: 0.5 for p in phenotypes}
    system = (f"You are a specialist clinician-scientist for the '{region}' body-composition region. "
              f"Rate each phenotype's clinical relevance to cardiometabolic disease risk in [0,1].")
    user = ("Return ONLY JSON: {\"relevance\": {\"<phenotype>\": <float 0..1>, ...}} for these phenotypes:\n"
            + "\n".join(phenotypes))
    out = model.chat_json(system, user)
    rel = out.get("relevance", {}) if isinstance(out, dict) else {}
    return {p: float(rel.get(p, 0.5)) for p in phenotypes}


# ============================================================ Stage II  E_i, coordinator E_G, confounders
def stage2_factors(structures: Dict[str, List[str]], data: pd.DataFrame, factors: List[str],
                   label: Optional[str], tools: Tools) -> Dict[str, Any]:
    """Associate every phenotype with every non-imaging factor (local E_i); coordinator aggregates to
    E_G; identify confounders = factors associated with BOTH phenotypes AND the disease outcome."""
    factors = [f for f in factors if f in data.columns]
    from utils import apply_multiple_testing_correction
    E: Dict[str, Dict[str, float]] = {}
    per_factor_strength: Dict[str, List[float]] = defaultdict(list)
    per_factor_pvals: Dict[str, List[float]] = defaultdict(list)
    for region, phenos in structures.items():
        for p in phenos:
            if p not in data.columns:
                continue
            for f in factors:
                a = tools.association(data[p], data[f])
                E[f"{p}__{f}"] = a
                per_factor_strength[f].append(a["abs_effect"])
                per_factor_pvals[f].append(a["p"])
    factor_global = {f: float(np.nanmean(v)) if v else 0.0 for f, v in per_factor_strength.items()}
    factor_max = {f: float(np.nanmax(v)) if v else 0.0 for f, v in per_factor_strength.items()}
    # FDR-corrected share of phenotypes each factor is significantly associated with
    factor_pheno_sig_frac = {}
    for f in factors:
        pv = np.asarray(per_factor_pvals[f], dtype=float)
        pv = pv[np.isfinite(pv)]
        if len(pv) == 0:
            factor_pheno_sig_frac[f] = 0.0
            continue
        rejected, _ = apply_multiple_testing_correction(pv, method="fdr")
        factor_pheno_sig_frac[f] = float(np.mean(rejected))

    factor_disease = {}
    if label and label in data.columns:
        for f in factors:
            factor_disease[f] = tools.association(data[f], data[label])
    # Confounder (paper def: "influences both the independent [phenotype] and dependent [outcome] variable"):
    # (i) significantly associated with the OUTCOME (p<0.05) AND
    # (ii) significantly associated with the phenotype space (FDR-significant for > chance share, >5%).
    confounders = sorted(
        [f for f in factors
         if factor_disease.get(f, {}).get("p", 1.0) < SIG_THRESHOLD
         and factor_pheno_sig_frac.get(f, 0) > SIG_THRESHOLD],
        key=lambda f: factor_disease.get(f, {}).get("abs_effect", 0), reverse=True)
    return {"factor_global": factor_global, "factor_max": factor_max,
            "factor_pheno_sig_frac": factor_pheno_sig_frac, "factor_disease": factor_disease,
            "confounders": confounders, "n_factors": len(factors)}


# ============================================================ Stage III  sequential consensus
def stage3_consensus(structures: Dict[str, List[str]], V_by_region: Dict[str, Dict[str, Dict]],
                     assoc: pd.DataFrame, factor_info: Dict[str, Any], k: int,
                     memories: Dict[str, AgentMemory], model: Optional[ModelClient],
                     use_memory: bool = True, use_tools: bool = True,
                     single_round: bool = False) -> Dict[str, Any]:
    """O_i^t = g_i(V_i, H_{t-1}, M_i); coordinator aggregates via f_AP each round; stop on convergence.

    Ablation flags (for Table-1 reproduction):
      single_round=True  -> 'w/o Stage III' (one round, no iterative consensus)
      use_memory=False, use_tools=False -> 'w/o Memory & Tools'
    """
    sig = {r.phenotype: {"p": float(r.p_value), "eff": float(r.abs_beta),
                         "fdr": bool(r.significant)} for r in assoc.itertuples()}
    region_order = sorted(structures.keys())
    history: List[Dict[str, Any]] = []          # H: list of {round, region, opinion}
    prev_set: Optional[set] = None
    rounds_log: List[Dict[str, Any]] = []
    consensus: List[str] = []
    converged = False
    max_rounds = 1 if single_round else MAX_ROUNDS
    t = 0
    for t in range(1, max_rounds + 1):
        round_opinions: Dict[str, List[str]] = {}
        for region in region_order:
            domain = [p for p in structures[region] if p in V_by_region.get(region, {})]
            opinion = _agent_opinion(region, domain, V_by_region[region], sig, factor_info,
                                     history, memories[region], model, k, use_memory=use_memory)
            round_opinions[region] = opinion["recommended_phenotypes"]
            history.append({"round": t, "region": region, "opinion": opinion})
            if use_memory:
                memories[region].store(f"round{t} {region} consensus-step",
                                       {"stage": "III", "round": t, "opinion": opinion})
        consensus = f_AP(round_opinions, sig, k, use_tools=use_tools)
        cur = set(consensus)
        jac = _jaccard(cur, prev_set) if prev_set is not None else 0.0
        rounds_log.append({"round": t, "consensus": consensus, "jaccard_vs_prev": jac})
        logger.info(f"[consensus] round {t}: |set|={len(consensus)} jaccard_vs_prev={jac:.3f}")
        if prev_set is not None and jac >= CONVERGE_JACCARD:
            converged = True
            break
        prev_set = cur
    top_factor = max(factor_info["factor_global"], key=factor_info["factor_global"].get) \
        if factor_info["factor_global"] else None
    return {"discovered_phenotypes": consensus, "rounds_run": t, "converged": converged,
            "rounds_log": rounds_log, "history": history, "top_associative_factor": top_factor,
            "discovered_confounders": factor_info["confounders"]}


def _agent_opinion(region: str, domain: List[str], V: Dict[str, Dict], sig: Dict[str, Dict],
                   factor_info: Dict[str, Any], history: List[Dict], memory: AgentMemory,
                   model: Optional[ModelClient], k: int, use_memory: bool = True) -> Dict[str, Any]:
    """One specialist's opinion this round, conditioned on its valuations, the discussion history, memory."""
    # statistical evidence for this agent's domain (tool-derived; passed to the agent)
    evidence = []
    for p in domain:
        s = sig.get(p, {"p": 1.0, "eff": 0.0, "fdr": False})
        evidence.append({"phenotype": p, "valuation": round(V[p]["valuation"], 3),
                         "assoc_effect": round(s["eff"], 3), "p_value": s["p"], "fdr_sig": s["fdr"]})
    evidence.sort(key=lambda e: (e["fdr_sig"], e["assoc_effect"], e["valuation"]), reverse=True)

    if DRY_RUN or model is None:                         # testing-only statistical stub
        rec = [e["phenotype"] for e in evidence[:max(2, k // max(1, len(domain) // 3 or 1))]]
        return {"recommended_phenotypes": rec, "top_factor": None, "rationale": "stub"}

    retrieved = memory.retrieve(f"{region} opinion") if use_memory else None
    recent = [h for h in history[-len(set(h['region'] for h in history)) - 1:]] if history else []
    hist_txt = "; ".join(f"{h['region']}->{h['opinion'].get('recommended_phenotypes', [])[:3]}"
                         for h in recent) or "(none yet)"
    system = (f"You are the '{region}' specialist agent in a multi-disciplinary PheWAS panel. "
              f"Form an INDEPENDENT opinion, then refine it using the discussion so far. "
              f"Recommend the phenotypes from YOUR domain most worth carrying to consensus for disease "
              f"association, balancing statistical evidence (effect size, FDR significance) with clinical relevance.")
    user = (f"Domain statistical evidence (tool-derived):\n{json.dumps(evidence, indent=1)}\n\n"
            f"Discussion history (recent): {hist_txt}\n"
            f"Top cross-domain factor so far: {max(factor_info['factor_global'], key=factor_info['factor_global'].get) if factor_info['factor_global'] else 'n/a'}\n"
            f"Retrieved memory: {retrieved['text'] if retrieved else '(none)'}\n\n"
            f"Return ONLY JSON: {{\"recommended_phenotypes\": [<from your domain, ranked, up to {min(len(domain),5)}>], "
            f"\"top_factor\": \"<one factor>\", \"rationale\": \"<one sentence>\"}}")
    out = model.chat_json(system, user)
    rec = [p for p in out.get("recommended_phenotypes", []) if p in domain] if isinstance(out, dict) else []
    if not rec:                                          # safety: fall back to statistical ranking
        rec = [e["phenotype"] for e in evidence[:min(len(domain), 3)]]
    return {"recommended_phenotypes": rec, "top_factor": out.get("top_factor") if isinstance(out, dict) else None,
            "rationale": out.get("rationale", "") if isinstance(out, dict) else ""}


# ============================================================ f_AP aggregation
def f_AP(opinions: Dict[str, List[str]], sig: Dict[str, Dict], k: int,
         use_tools: bool = True) -> List[str]:
    """A = f_AP({O^t}, TR): weight each recommended phenotype by rank, statistical confidence (p<0.05),
    and on-topic relevance (|effect|>0.3). Return the top-k consensus phenotypes.
    use_tools=False ('w/o Memory & Tools' ablation) drops the tool-evidence weighting -> rank votes only."""
    votes: Dict[str, float] = defaultdict(float)
    for region, recs in opinions.items():
        for rank, p in enumerate(recs):
            if use_tools:
                s = sig.get(p, {"p": 1.0, "eff": 0.0})
                conf = 1.0 if s["p"] < SIG_THRESHOLD else 0.3
                rel = 1.0 if s["eff"] > RELEVANCE_THRESHOLD else 0.3
            else:
                conf = rel = 1.0
            votes[p] += (1.0 / (1 + rank)) * conf * rel
    ranked = sorted(votes, key=lambda p: votes[p], reverse=True)
    return ranked[:k]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ============================================================ Auto-PheWAS metrics  Q(P), C(P)
def dependency(P: List[str], data: pd.DataFrame,
               phenotype_universe: Optional[List[str]] = None) -> Optional[float]:
    """Auto-PheWAS Dependency (paper §3, exact):
        Q(P) = [1 - (2/(K(K-1))) * sum_{i<j} |corr(p_i, p_j)|] * (|P_valid| / |P|)
    where |P_valid| is the number of *valid* (real, non-hallucinated) phenotypes in P -- here, those
    present in the real phenotype universe (defaults to data.columns). Constant columns contribute no
    pairwise correlation (NaN) and are skipped via nanmean, but still count as valid (they are real).

    Note on direction: the paper labels this 'Dependency' with lower=better, yet the printed formula is an
    independence-x-validity score (higher for more-independent/valid sets). We implement the formula
    *verbatim*; interpret the sign per the paper's Table 1 convention.
    """
    if not P:
        return None
    universe = set(phenotype_universe) if phenotype_universe is not None else set(data.columns)
    valid = [p for p in P if p in universe]                 # validity = real phenotype (hallucination check)
    if len(valid) < 2:
        return None
    C = data[valid].apply(pd.to_numeric, errors="coerce").corr().abs().values
    iu = np.triu_indices(len(valid), k=1)
    mean_abs = float(np.nanmean(C[iu]))                     # constants -> NaN pairs, skipped
    return (1.0 - mean_abs) * (len(valid) / len(P))


def coverage(P: List[str], phenotype_universe: List[str],
             ws: float = 0.5, wf: float = 0.5) -> Optional[float]:
    """Auto-PheWAS Coverage (paper §3, exact):
        C(P) = ws * (|Omega_covered| / |Omega|) + wf * (|F_covered| / |F_total|)
    Omega = the set of anatomical structures, F = the structure-function combinations that *actually
    exist* in the full phenotype universe (NOT the Cartesian product structures x measures). Default
    weights ws=wf=0.5 (the paper specifies a weighted sum but not the exact weights).
    """
    if not P or not phenotype_universe:
        return None
    uni_struct, uni_combo = set(), set()
    for c in phenotype_universe:                            # build Omega and F_total from real data
        r, m = parse_region_measure(c)
        if r and r != "other":
            uni_struct.add(r)
            if m:
                uni_combo.add((r, m))
    if not uni_struct or not uni_combo:
        return None
    cov_struct, cov_combo = set(), set()
    for p in P:
        r, m = parse_region_measure(p)
        if r in uni_struct:
            cov_struct.add(r)
            if m and (r, m) in uni_combo:
                cov_combo.add((r, m))
    return ws * len(cov_struct) / len(uni_struct) + wf * len(cov_combo) / len(uni_combo)
