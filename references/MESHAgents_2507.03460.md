# MESHAgents — reference

**Title:** Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis
**Authors:** Weitong Zhang*, Mengyun Qiao*, Chengqi Zang, Steven Niederer, Paul Matthews, Wenjia Bai, Bernhard Kainz (*equal contribution)
**Venue:** MICCAI 2025
**arXiv:** 2507.03460
**Links:**
- PDF: https://arxiv.org/pdf/2507.03460
- Abstract page: https://arxiv.org/abs/2507.03460
- Springer chapter: https://link.springer.com/chapter/10.1007/978-3-032-04927-8_41
- Code: https://github.com/LumaLabAI/MESHAgents

## Summary

MESHAgents is a multi-agent LLM framework for phenome-wide association studies (PheWAS) on
cardiovascular imaging phenotypes. It orchestrates a team of domain-specialized agents — one per
anatomical structure (LV, RV, LA, RA, ascending aorta, descending aorta) — plus a coordinator, which
collaboratively surface confounders and phenotype–factor associations through a sequential discussion
and consensus protocol inspired by clinical multi-disciplinary team (MDT) panels. Agents carry memory
and statistical-reasoning tools.

Validated on UK Biobank cardiac MR imaging-derived phenotypes (training ~26,893 participants; test
~38,309), with nine disease outcomes for the diagnosis evaluation. The framework's automatically
discovered phenotypes match expert-selected ones on diagnosis performance (reported mean AUC
difference of about -0.004 +/- 0.010 for LDA) while improving recall for 6 of 9 disease types.

## Relevance to this project (pheno26)

This is the framework we are adapting from cardiac MRI to **whole-body HPP phenotypes**. The per-structure
specialist design maps onto our body-composition region groups (android / gynoid / arms / legs / trunk /
total) and, later, other HPP datasets. See `whole_body_multiagent_plan.md` and
`hpp_data_feasibility_findings.md` for the adaptation plan and data feasibility.

## Adding the PDF

The binary PDF is not included here (couldn't be auto-downloaded). To add it:

```bash
cd /Users/abhijit.das/Documents/GitHub/pheno26/references
curl -L -o MESHAgents_2507.03460.pdf https://arxiv.org/pdf/2507.03460
# or just open https://arxiv.org/pdf/2507.03460 in a browser and save it here
```
