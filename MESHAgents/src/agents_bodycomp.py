"""Body-composition adaptation layer for MESHAgents.

Reuses the original framework (BaseOrganAgent, ChiefAgent, StatisticalTools, Memory) and replaces
only the cardiac-specific specialist/orchestration logic with dataset-agnostic versions:
  - RegionAgent: one specialist per body-composition region (android/gynoid/arms/legs/trunk/total/...)
  - BodyCompChiefAgent: builds region specialists from a `structures` dict and runs the same
    3-stage pipeline (per-region analysis -> cross-region -> phenotype discovery -> system-level -> GPT)

Faithful PheWAS reproduction (mirrors the upstream paper + Classification_methods.ipynb):
  - The agents' *statistical tool* for discovery is a confounder-adjusted association test of every
    phenotype against the disease label (logistic regression: label ~ z(phenotype) + z(confounders)),
    ranked by effect size among FDR-significant hits (reusing utils.apply_multiple_testing_correction).
  - The chief's *consensus* step distils a parsimonious `discovered_phenotypes` set (size ~ expert set),
    which the downstream classification eval compares against an expert-selected set (with confounders),
    exactly as in the paper.

Set env OFFLINE_LLM=1 to skip all OpenAI calls (statistical pipeline still runs and is logged).
"""
import os
import logging
from itertools import combinations
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from openai import OpenAI

from agents import BaseOrganAgent, ChiefAgent, StatisticalTools, Memory
from config import GPT_MODEL
from utils import apply_multiple_testing_correction
import mesh_core
import llm_provider as P                                  # provider switch (OpenAI <-> Ollama)

logger = logging.getLogger(__name__)

OFFLINE = os.getenv("OFFLINE_LLM", "0") == "1"
_STUB = "[OFFLINE_LLM] GPT analysis skipped; statistical results computed without the LLM."

# Known disease label columns produced by build_merged_data.py (used to auto-detect the label).
DISEASE_LABELS = ["diabetes", "hypertension", "abdominal_adiposity", "mafld", "osteoporosis"]

# Candidate non-imaging factors C (demographics + anthropometrics) for Stage II factor discovery.
FACTOR_CANDIDATES = ["age", "sex", "bmi", "weight", "height", "waist_circumference",
                     "hip_circumference", "waist_to_hip_ratio"]

# Upstream demographic confounders to adjust the phenotype->disease ASSOCIATION ranking for.
# (Anthropometric "confounders" found in Stage II are proxies of the adiposity signal -- adjusting the
#  ranking for them would over-adjust; instead they are reported and appended to the diagnosis classifier,
#  exactly as the paper appends Sex/Age/Weight/Height to the feature set.)
DEMOGRAPHIC_CONFOUNDERS = ["age", "sex"]

# Size of the consensus "discovered" set (paper matches it to the expert set size, ~13).
DISCOVER_K = int(os.getenv("DISCOVER_K", "13"))

# Clinically standard compact body-composition / cardiometabolic phenotype set (the "expert" baseline,
# analogue of the cardiac expert_selected list in Classification_methods.ipynb). Filtered to present cols.
EXPERT_BODYCOMP = [
    "total_scan_vat_mass", "total_scan_vat_volume", "total_scan_sat_mass",
    "body_comp_total_fat_mass", "body_comp_android_fat_mass", "body_comp_gynoid_fat_mass",
    "body_comp_trunk_fat_mass", "body_comp_total_lean_mass", "body_comp_total_fat_free_mass",
    "body_comp_android_region_percent_fat", "body_comp_total_region_percent_fat",
    "body_comp_legs_fat_mass", "body_comp_arms_fat_mass", "body_comp_total_bone_mass",
]


def _gpt(client: OpenAI, system: str, user: str, model: str = None) -> str:
    if OFFLINE:
        return _STUB
    try:
        resp = client.chat.completions.create(
            model=model or P.chat_model(default=GPT_MODEL),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return f"[GPT_ERROR] {e}"


def build_structures(columns: List[str]) -> Dict[str, List[str]]:
    """Group body_comp_<region>_* columns into region specialists (+ whole-body adipose)."""
    regions = sorted(["android", "gynoid", "arm_left", "arm_right", "arms",
                      "leg_left", "leg_right", "legs", "trunk", "total"],
                     key=len, reverse=True)
    groups: Dict[str, List[str]] = {}
    for c in columns:
        if c.startswith("body_comp_"):
            rest = c[len("body_comp_"):]
            reg = next((r for r in regions if rest.startswith(r)), "other")
            groups.setdefault(reg, []).append(c)
        elif c.startswith("total_scan_"):
            groups.setdefault("whole_body_adipose", []).append(c)
    return groups


def detect_label(columns) -> Optional[str]:
    """Return the disease label column present in the data, if any."""
    return next((d for d in DISEASE_LABELS if d in columns), None)


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if not sd or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def associate_phenotypes(data: pd.DataFrame, label_col: str, phenotypes: List[str],
                         confounders: List[str]) -> pd.DataFrame:
    """Confounder-adjusted PheWAS: associate every phenotype with the binary disease label.

    For each phenotype p, fit  label ~ z(p) + z(confounders)  (logistic regression) and record the
    standardised phenotype coefficient (effect size) and its Wald p-value. FDR-correct across all
    phenotypes with the codebase's own apply_multiple_testing_correction (Benjamini-Hochberg).

    Returns a DataFrame ranked by FDR-significance then |effect size|.
    """
    y = pd.to_numeric(data[label_col], errors="coerce")
    conf = [c for c in confounders if c in data.columns]
    Z = pd.DataFrame({c: _zscore(data[c]) for c in conf}, index=data.index)
    # drop confounders that are entirely non-numeric / all-NaN so they don't void every row
    conf = [c for c in conf if Z[c].notna().any()]
    Z = Z[conf]

    rows = []
    for p in phenotypes:
        if p not in data.columns:
            continue
        x = _zscore(data[p]).rename("pheno")
        d = pd.concat([y.rename("y"), x, Z], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if d["y"].nunique() < 2 or len(d) < 30 or d["pheno"].std() == 0:
            continue
        X = sm.add_constant(d[["pheno"] + conf], has_constant="add")
        try:
            res = sm.Logit(d["y"], X).fit(disp=0, maxiter=200)
            beta = float(res.params.get("pheno", np.nan))
            pval = float(res.pvalues.get("pheno", np.nan))
        except Exception as e:
            logger.debug(f"assoc failed for {p}: {e}")
            continue
        if not np.isfinite(pval) or not np.isfinite(beta):
            continue
        rows.append((p, beta, abs(beta), pval, int(len(d))))

    df = pd.DataFrame(rows, columns=["phenotype", "beta", "abs_beta", "p_value", "n"])
    if df.empty:
        df["fdr_p"] = []
        df["significant"] = []
        return df
    rejected, fdr_p = apply_multiple_testing_correction(df["p_value"].values, method="fdr")
    df["fdr_p"] = fdr_p
    df["significant"] = rejected
    return df.sort_values(["significant", "abs_beta"], ascending=[False, False]).reset_index(drop=True)


def discover_set(assoc: pd.DataFrame, k: int = DISCOVER_K) -> List[str]:
    """Consensus distillation: the top-k phenotypes by association, preferring FDR-significant ones."""
    if assoc is None or assoc.empty:
        return []
    sig = assoc[assoc["significant"]]
    pool = sig if len(sig) >= k else assoc
    return pool["phenotype"].head(k).tolist()


class RegionAgent(BaseOrganAgent):
    """Generic specialist for one body-composition region."""

    def __init__(self, name: str, phenotypes: List[str], api_key: str):
        super().__init__(name, phenotypes, api_key)
        # override the base-class OpenAI client/model with the provider-aware ones (Ollama/OpenAI)
        self.client = P.chat_client()
        self.model = P.chat_model(default=self.model)
        self.system_prompt = (
            f"You are a specialist analyst for the '{name}' body-composition region. "
            f"Interpret its structural/compositional phenotypes (fat, lean, bone, tissue %)."
        )

    def get_gpt_analysis(self, data_description: str, temperature: float = 0) -> str:
        if OFFLINE:
            return _STUB
        return _gpt(self.client, self.system_prompt, data_description, model=self.model)

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        cols = [c for c in self.phenotypes if c in data.columns]
        sub = data[cols].apply(pd.to_numeric, errors="coerce").dropna()
        results: Dict[str, Any] = {"name": self.name, "n_phenotypes": len(cols),
                                   "n_samples": int(len(sub))}
        try:
            results["basic_stats"] = self._calculate_basic_stats(sub)
        except Exception as e:
            logger.warning(f"[{self.name}] basic stats failed: {e}")
            results["basic_stats"] = {}
        means = ", ".join(f"{c}={sub[c].mean():.1f}" for c in cols[:6]) if len(sub) else "n/a"
        desc = f"Region {self.name}: {len(cols)} phenotypes, n={len(sub)}. Example means: {means}"
        results["gpt_analysis"] = self.get_gpt_analysis(desc)
        return results


class BodyCompChiefAgent(ChiefAgent):
    """Chief agent that coordinates body-composition region specialists."""

    def __init__(self, api_key: str, structures: Dict[str, List[str]],
                 clinical_factors: List[str] = None):
        # intentionally do NOT call super().__init__ (it builds cardiac agents)
        self.agents = {name: RegionAgent(name, cols, api_key)
                       for name, cols in structures.items()}
        self.memory = Memory()
        self.tools = StatisticalTools()
        self.latest_results = None
        self.api_key = api_key
        self.client = P.chat_client()                    # provider-aware (Ollama/OpenAI)
        self.model = P.chat_model(default=GPT_MODEL)
        self.clinical_factors = clinical_factors or []
        self.system_prompt = (
            "You are a senior whole-body phenotype expert leading a team of body-composition "
            "region specialists. Integrate per-region findings, identify cross-region patterns, "
            "and surface the most informative phenotypes for downstream association/diagnosis."
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run_analysis(self, data: pd.DataFrame, label: Optional[str] = None,
                           ablation: Optional[str] = None) -> Dict[str, Any]:
        """Faithful MESHAgents 3-stage protocol (arXiv:2507.03460v2 §2) over body-comp region agents.

        Stage I  (per-region phenotype valuation V_i)  -> Stage II (factor discovery E_i/E_G + confounders)
        -> confounder-adjusted PheWAS association (tool evidence TR) -> Stage III (sequential LLM consensus
        O_i^t = g_i(V_i,H_{t-1},M_i), <=10 rounds) -> f_AP -> discovered set, plus Auto-PheWAS metrics Q(P),C(P).
        The upstream unsupervised salience score is retained under `phenotype_importance` for continuity.
        """
        try:
            structures = {name: [c for c in a.phenotypes if c in data.columns]
                          for name, a in self.agents.items()}
            structures = {k: v for k, v in structures.items() if v}
            all_ph = [c for v in structures.values() for c in v]
            label = label or detect_label(data.columns)
            factors = [f for f in FACTOR_CANDIDATES if f in data.columns]

            results: Dict[str, Any] = {}
            logger.info("Running region-specific analyses (upstream stats)...")
            results["organ_specific"] = await self._run_parallel_analyses(data)
            results["cross_organ"] = await self._analyze_cross_organ_relationships(data)
            results["phenotype_importance"] = self._discover_key_phenotypes(data)  # upstream unsupervised

            # ablation flags (Table-1 reproduction): None/"full", "no_stage3", "no_mem_tools"
            use_memory = ablation != "no_mem_tools"
            use_tools = ablation != "no_mem_tools"
            single_round = ablation == "no_stage3"
            if ablation:
                logger.info(f"ABLATION='{ablation}' (use_memory={use_memory}, use_tools={use_tools}, "
                            f"single_round={single_round})")

            tools = mesh_core.Tools()
            embedder = mesh_core.Embedder()
            model = None if mesh_core.DRY_RUN else mesh_core.ModelClient()
            memories = {r: mesh_core.AgentMemory(embedder) for r in structures}

            # Stage I — phenotype valuation V_i per specialist
            logger.info("Stage I: phenotype valuation (V_i) per region specialist...")
            V_by_region = {r: mesh_core.stage1_valuation(r, ph, data, tools, memories[r], model,
                                                         use_memory=use_memory)
                           for r, ph in structures.items()}

            # Stage II — associative factor discovery + confounder identification (E_i, E_G)
            logger.info(f"Stage II: associative factor discovery over factors {factors}...")
            factor_info = mesh_core.stage2_factors(structures, data, factors, label, tools)
            logger.info(f"Stage II: discovered confounders = {factor_info['confounders']}")

            if label and label in data.columns:
                # tool evidence TR: phenotype<->disease association adjusted for upstream demographics
                # (age, sex). Anthropometric confounders are proxies -> not partialled out here.
                adj = [c for c in DEMOGRAPHIC_CONFOUNDERS if c in data.columns] or \
                    [c for c in self.clinical_factors if c in data.columns]
                assoc = associate_phenotypes(data, label, all_ph, adj)

                # Stage III — sequential multi-agent consensus (paper-exact, LLM opinions) + f_AP
                logger.info(f"Stage III: sequential consensus (<= {mesh_core.MAX_ROUNDS} rounds)...")
                consensus = mesh_core.stage3_consensus(structures, V_by_region, assoc, factor_info,
                                                       DISCOVER_K, memories, model,
                                                       use_memory=use_memory, use_tools=use_tools,
                                                       single_round=single_round)
                discovered = consensus["discovered_phenotypes"]

                # Auto-PheWAS metrics on the discovered set (Omega/F_total derived from real phenotypes)
                q = mesh_core.dependency(discovered, data, phenotype_universe=all_ph)
                cov = mesh_core.coverage(discovered, all_ph)

                results["phenotype_association"] = {
                    "label": label,
                    "association_adjusted_for": adj,
                    "confounders_used": factor_info["confounders"] or [c for c in self.clinical_factors if c in data.columns],
                    "discovered_confounders": factor_info["confounders"],
                    "factor_global": factor_info["factor_global"],
                    "n_phenotypes_tested": int(len(assoc)),
                    "n_significant_fdr": int(assoc["significant"].sum()) if not assoc.empty else 0,
                    "table": assoc.to_dict("records"),
                }
                results["consensus"] = {k: v for k, v in consensus.items()
                                        if k not in ("rounds_log", "history")}
                results["consensus_rounds"] = consensus["rounds_log"]
                results["consensus_history"] = consensus["history"]   # per-agent O_i^t trail (transparency)
                results["auto_phewas_metrics"] = {"dependency_Q": q, "coverage_C": cov}
                results["discovered_phenotypes"] = discovered
                logger.info(f"Discovered consensus set ({len(discovered)}): {discovered}")
                logger.info(f"Auto-PheWAS: dependency Q={q:.4f} (lower=better), coverage C={cov:.4f}")
            else:
                logger.warning("No disease label found; skipping Stage III consensus / discovery.")
                results["discovered_phenotypes"] = []

            results["stage1_valuations"] = {r: {p: round(V_by_region[r][p]["valuation"], 4) for p in V_by_region[r]}
                                            for r in V_by_region}
            results["system_level"] = self._system_level_analysis(data)
            results["gpt_analysis"] = self._get_gpt_analysis(data, results)

            self.latest_results = results
            self.memory.store_analysis("full_analysis", results)
            return results
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise

    async def _analyze_cross_organ_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        all_ph = [c for a in self.agents.values() for c in a.phenotypes if c in data.columns]
        pheno = data[all_ph].apply(pd.to_numeric, errors="coerce").dropna()
        corr = pheno.corr()
        names = list(self.agents.keys())
        pairs = list(combinations(names, 2))[:8]
        pair_res = {}
        for a, b in pairs:
            ca = [c for c in self.agents[a].phenotypes if c in data.columns]
            cb = [c for c in self.agents[b].phenotypes if c in data.columns]
            block = data[ca + cb].apply(pd.to_numeric, errors="coerce").dropna()
            if len(block) > 2 and ca and cb:
                cross = block[ca].corrwith(block[cb].mean(axis=1)).abs().mean()
                pair_res[f"{a}__{b}"] = {"mean_abs_cross_corr": float(cross)}
        return {"n_phenotypes": len(all_ph),
                "correlations": corr.round(3).to_dict(),
                "region_pairs": pair_res}

    def _system_level_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        region_summary = {}
        for name, agent in self.agents.items():
            cols = [c for c in agent.phenotypes if c in data.columns]
            sub = data[cols].apply(pd.to_numeric, errors="coerce")
            region_summary[name] = {
                "n_phenotypes": len(cols),
                "mean_of_means": float(np.nanmean(sub.mean().values)) if cols else None,
            }
        return {"region_summary": region_summary}

    def _get_gpt_analysis(self, data: pd.DataFrame, results: Dict[str, Any]) -> str:
        regions = list(self.agents.keys())
        discovered = results.get("discovered_phenotypes", [])
        assoc = results.get("phenotype_association", {})
        cons = results.get("consensus", {})
        metrics = results.get("auto_phewas_metrics", {})
        summary = (f"Whole-body body-composition PheWAS across regions {regions}. "
                   f"Label={assoc.get('label')}, discovered confounders={assoc.get('discovered_confounders')}. "
                   f"{assoc.get('n_significant_fdr', 0)} phenotypes FDR-significant. "
                   f"Consensus reached in {cons.get('rounds_run')} rounds (converged={cons.get('converged')}); "
                   f"top associative factor={cons.get('top_associative_factor')}. "
                   f"Auto-PheWAS dependency Q={metrics.get('dependency_Q')}, coverage C={metrics.get('coverage_C')}. "
                   f"Discovered set: {discovered}.")
        return _gpt(self.client, self.system_prompt,
                    f"Summarise these whole-body body-composition PheWAS findings:\n{summary}",
                    model=self.model)
