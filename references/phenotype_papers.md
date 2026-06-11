# Phenotype-data papers — baselines & SOTA for comparison

Curated list of papers that work directly on **phenotype data** (imaging/clinical-derived phenotypes,
PheWAS, phenotype discovery). Medical-QA / decision agents (MedAgents, MDAgents, etc.) are intentionally
excluded — they don't operate on phenotype data. Rows 1–6 are peer-reviewed in Nature-family journals or
A* venues; rows 7–8 are preprints (flagged).

| # | Paper | Venue (year) | Phenotype data / modality | Task | Link |
|---|---|---|---|---|---|
| 1 | MESHAgents | MICCAI 2025 (A*) | Cardiac & aortic imaging-derived phenotypes (UK Biobank) | Multi-agent PheWAS + disease diagnosis | [arXiv](https://arxiv.org/abs/2507.03460) · [Springer](https://link.springer.com/chapter/10.1007/978-3-032-04927-8_41) |
| 2 | DeepRare | Nature 2025 | HPO clinical phenotypes (+ genotype) | Agentic phenotype-driven rare-disease diagnosis | [Nature](https://www.nature.com/articles/s41586-025-10097-9) · [arXiv](https://arxiv.org/abs/2506.20430) |
| 3 | Bai et al. | Nature Medicine 2020 | Cardiac/aortic structure & function phenotypes (UKB) | Population phenome-wide association study | [Nature Medicine](https://www.nature.com/articles/s41591-020-1009-y) |
| 4 | UDIP (unsupervised deep representation learning) | Communications Biology 2024 | Brain MRI imaging-derived phenotypes (UKB) | Unsupervised phenotype discovery → GWAS | [Comms Biology](https://www.nature.com/articles/s42003-024-06096-7) |
| 5 | DL phenotyping of medical images | npj Digital Medicine 2023 | DL imaging-derived phenotypes | Phenotyping to boost gene-discovery power | [npj Digital Medicine](https://www.nature.com/articles/s41746-023-00903-x) |
| 6 | Unsupervised ECG disease profiling | npj Digital Medicine 2024 | ECG-derived phenotypes | Scalable PheWAS-style disease profiling (~1,600 Phecodes) | [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01418-9) |
| 7 | iGWAS (self-supervised deep phenotyping) | medRxiv / Comms Medicine (preprint) | Self-supervised image phenotypes | Image-based genome-wide association | [medRxiv](https://www.medrxiv.org/content/10.1101/2022.05.26.22275626) |
| 8 | PhenoGraph | bioRxiv 2025 (preprint) | Spatial-transcriptomics phenotypes | Multi-agent phenotype-driven discovery | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.06.658341v1) |

## How they serve as baselines for pheno26

- **Method baseline (the one we reproduce / extend):** #1 MESHAgents.
- **Top-journal SOTA reference (phenotype-driven agentic):** #2 DeepRare.
- **Expert / automated-pipeline gold standard for the discovered set:** #3 Bai et al. 2020 (the
  expert-selected phenotype baseline; analogue of our `EXPERT_BODYCOMP` set in `evaluate_diagnosis.py`).
- **Non-agent automatic phenotype-discovery baselines:** #4 UDIP, #5 DL phenotyping, #7 iGWAS.
- **Directly relevant to our future ECG dataset:** #6 unsupervised ECG disease profiling.
- **Closest multi-agent phenotype-discovery (different modality, preprint):** #8 PhenoGraph.

## BibTeX

```bibtex
@inproceedings{zhang2025meshagents,
  title     = {Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis},
  author    = {Zhang, Weitong and Qiao, Mengyun and Zang, Chengqi and Niederer, Steven and
               Matthews, Paul and Bai, Wenjia and Kainz, Bernhard},
  booktitle = {MICCAI},
  year      = {2025},
  note      = {arXiv:2507.03460}
}

@article{deeprare2025,
  title   = {DeepRare: an agentic system for rare disease diagnosis with traceable reasoning},
  journal = {Nature},
  year    = {2025},
  note    = {arXiv:2506.20430},
  url     = {https://www.nature.com/articles/s41586-025-10097-9}
}

@article{bai2020phewas,
  title   = {A population-based phenome-wide association study of cardiac and aortic structure and function},
  author  = {Bai, Wenjia and Suzuki, Hideaki and Huang, Jian and others},
  journal = {Nature Medicine},
  volume  = {26},
  pages   = {1654--1662},
  year    = {2020},
  url     = {https://www.nature.com/articles/s41591-020-1009-y}
}

@article{patel2024udip,
  title   = {Unsupervised deep representation learning enables phenotype discovery for genetic
             association studies of brain imaging},
  author  = {Patel, Khush and Xie, Ziqian and others},
  journal = {Communications Biology},
  year    = {2024},
  url     = {https://www.nature.com/articles/s42003-024-06096-7}
}

@article{dlphenotyping2023,
  title   = {Deep learning based phenotyping of medical images improves power for gene discovery of
             complex disease},
  journal = {npj Digital Medicine},
  year    = {2023},
  url     = {https://www.nature.com/articles/s41746-023-00903-x}
}

@article{ecgprofiling2024,
  title   = {Unsupervised deep learning of electrocardiograms enables scalable human disease profiling},
  journal = {npj Digital Medicine},
  year    = {2024},
  url     = {https://www.nature.com/articles/s41746-024-01418-9}
}

@article{igwas2022,
  title   = {iGWAS: image-based genome-wide association of self-supervised deep phenotyping of human
             medical images},
  journal = {medRxiv (preprint)},
  year    = {2022},
  url     = {https://www.medrxiv.org/content/10.1101/2022.05.26.22275626}
}

@article{phenograph2025,
  title   = {PhenoGraph: a multi-agent framework for phenotype-driven discovery in spatial transcriptomics},
  journal = {bioRxiv (preprint)},
  year    = {2025},
  url     = {https://www.biorxiv.org/content/10.1101/2025.06.06.658341v1}
}
```
