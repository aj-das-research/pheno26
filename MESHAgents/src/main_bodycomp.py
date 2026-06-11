"""Body-composition entry point for MESHAgents (mirrors main.py).

Run from the MESHAgents/ directory:
    python src/main_bodycomp.py
Set OFFLINE_LLM=1 to run the statistical pipeline without any OpenAI calls.
"""
import os
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import llm_provider as P          # provider switch — MUST precede `import config` (ollama dummy-key shim)
from config import OPENAI_API_KEY, DATA_PATH, RESULTS_PATH, LOG_PATH, GPT_MODEL
from agents_bodycomp import BodyCompChiefAgent, build_structures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Path(LOG_PATH) / "analysis_bodycomp.log"),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CLINICAL_FACTORS = ["age", "sex", "bmi", "weight", "waist_circumference"]


def json_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def save_results(results: Any, filepath: Path) -> None:
    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath)
    else:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4, default=json_serializer)
    logger.info(f"Results saved to {filepath}")


async def main():
    for path in [RESULTS_PATH, LOG_PATH]:
        Path(path).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    data = pd.read_csv(DATA_PATH)
    logger.info(f"Data loaded: {data.shape}")

    # structures.json if present (from build_merged_data.py), else derive from columns
    struct_path = Path(DATA_PATH).parent / "structures.json"
    if struct_path.exists():
        structures = json.loads(struct_path.read_text())
        structures = {k: [c for c in v if c in data.columns] for k, v in structures.items()}
    else:
        structures = build_structures(list(data.columns))
    structures = {k: v for k, v in structures.items() if v}
    logger.info(f"Region specialists ({len(structures)}): "
                f"{ {k: len(v) for k, v in structures.items()} }")

    offline = os.getenv("OFFLINE_LLM", "0") == "1"
    dry_run = os.getenv("MESH_DRY_RUN", "0") == "1"
    if not offline and not dry_run:
        P.check()                                        # fail fast if Ollama is down / model not pulled (no-op for OpenAI)
    logger.info(f"Initializing BodyCompChiefAgent "
                f"(provider={P.provider()}, model={P.chat_model(default=GPT_MODEL)}, OFFLINE_LLM={offline})")
    chief = BodyCompChiefAgent(OPENAI_API_KEY, structures, CLINICAL_FACTORS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Running analysis pipeline...")
    results = await chief.run_analysis(data)

    tasks = [save_results(results, Path(RESULTS_PATH) / f"analysis_results_{timestamp}.json")]

    phenotype_scores = pd.DataFrame(
        results["phenotype_importance"]["top_phenotypes"], columns=["Phenotype", "Score"]
    )
    tasks.append(save_results(phenotype_scores,
                              Path(RESULTS_PATH) / f"phenotype_scores_{timestamp}.csv"))

    for region, region_results in results["organ_specific"].items():
        tasks.append(save_results(region_results,
                                  Path(RESULTS_PATH) / f"{region}_analysis_{timestamp}.json"))

    if "gpt_analysis" in results:
        tasks.append(save_results({"interpretation": results["gpt_analysis"]},
                                  Path(RESULTS_PATH) / f"gpt_analysis_{timestamp}.json"))

    # PheWAS association + consensus discovered set + metrics (paper-faithful discovery artifacts)
    if "phenotype_association" in results:
        tasks.append(save_results(
            {"discovered_phenotypes": results.get("discovered_phenotypes", []),
             "discovered_confounders": results["phenotype_association"].get("discovered_confounders", []),
             "consensus": results.get("consensus", {}),
             "auto_phewas_metrics": results.get("auto_phewas_metrics", {}),
             "label": results["phenotype_association"]["label"],
             "confounders_used": results["phenotype_association"]["confounders_used"],
             "n_significant_fdr": results["phenotype_association"]["n_significant_fdr"],
             "n_phenotypes_tested": results["phenotype_association"]["n_phenotypes_tested"]},
            Path(RESULTS_PATH) / f"discovered_phenotypes_{timestamp}.json"))
        tasks.append(save_results(
            {"rounds": results.get("consensus_rounds", []),
             "agent_opinions": results.get("consensus_history", [])},
            Path(RESULTS_PATH) / f"consensus_transcript_{timestamp}.json"))
        assoc_df = pd.DataFrame(results["phenotype_association"]["table"])
        if not assoc_df.empty:
            tasks.append(save_results(assoc_df,
                                      Path(RESULTS_PATH) / f"phenotype_association_{timestamp}.csv"))

    await asyncio.gather(*tasks)

    logger.info("\nMain Findings:")
    logger.info(f"Analyzed phenotypes: {len(results['phenotype_importance']['scores'])}")
    logger.info("\nTop 15 phenotypes by (unsupervised) importance score:")
    logger.info(phenotype_scores.head(15).to_string())
    if results.get("discovered_phenotypes"):
        assoc = results["phenotype_association"]
        cons = results.get("consensus", {})
        metrics = results.get("auto_phewas_metrics", {})
        logger.info(f"\nPheWAS label = {assoc['label']}  |  confounders used = {assoc['confounders_used']}")
        logger.info(f"Stage II discovered confounders: {assoc.get('discovered_confounders')}")
        logger.info(f"FDR-significant phenotypes: {assoc['n_significant_fdr']}/{assoc['n_phenotypes_tested']}")
        logger.info(f"Stage III consensus: {cons.get('rounds_run')} rounds, converged={cons.get('converged')}, "
                    f"top factor={cons.get('top_associative_factor')}")
        logger.info(f"Auto-PheWAS metrics: dependency Q={metrics.get('dependency_Q')}, "
                    f"coverage C={metrics.get('coverage_C')}")
        logger.info(f"Consensus discovered set ({len(results['discovered_phenotypes'])}): "
                    f"{results['discovered_phenotypes']}")
    logger.info("Analysis pipeline completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
