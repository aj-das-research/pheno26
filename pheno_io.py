"""Unified data access for the HPP MESHAgents project.

Local development: reads synthetic parquet from SYNTH_DIR.
HPP VM:            set USE_SYNTHETIC = False -> uses the official pheno_utils.PhenoLoader.

The rest of the codebase only calls get_df(dataset, table); nothing else changes
between local and VM runs.
"""
import os
import pandas as pd

USE_SYNTHETIC = True                                    # <-- set False on the HPP VM
SYNTH_DIR = os.environ.get("SYNTH_DIR", "./synth_hpp")
KEYS = ["participant_id", "research_stage"]

_loaders = {}


def get_df(dataset, table):
    """Return a dataframe indexed by [participant_id, research_stage]."""
    if USE_SYNTHETIC:
        df = pd.read_parquet(os.path.join(SYNTH_DIR, f"{table}.parquet"))
        return df.set_index(KEYS)
    from pheno_utils import PhenoLoader                  # only imported on the VM
    if dataset not in _loaders:
        _loaders[dataset] = PhenoLoader(dataset)
    return _loaders[dataset].dfs[table]
