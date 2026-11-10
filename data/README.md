# Data

All contract text in this directory comes from the **Contract Understanding Atticus Dataset
(CUAD) v1**, © The Atticus Project, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

* Project page: https://www.atticusprojectai.org/cuad
* Release used: https://zenodo.org/records/4595826 (`CUAD_v1.zip`)
* Paper: Hendrycks, Burns, Chen, Ball. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract
  Review.* NeurIPS 2021 Datasets and Benchmarks Track.

## Changes made (as required by CC BY 4.0)

Rebuild everything with `scripts/build_cuad_subset.py --cuad-dir CUAD_v1`.

| File | What it is |
|---|---|
| `cuad/train_clauses.jsonl` | Clause segments from ~90% of contracts (split by title hash). `label` comes from overlap with a CUAD expert span, mapped to this project's clause types (`label_source: cuad`), or from a heading rule for indemnification / confidentiality (`label_source: heading_rule`), which CUAD does not annotate. These are **weak labels**. |
| `cuad/heldout_candidates.jsonl` | Segments from the other ~10% of contracts. The gold sets were drawn from this pool. |
| `cuad/contracts/reference/`, `cuad/contracts/new/` | Unmodified full texts of 16 training-split and 8 held-out contracts, used by the drift demo. |
| `gold/gold.jsonl` | **Dev** set. 69 held-out clauses with a clause type and a playbook risk label (`compliant`, `fallback`, `non_compliant`, `needs_human`, or `null` for `other`) plus a one-line note. The project author labeled them before writing the playbook checks, then used them while writing those checks. |
| `gold/blind.jsonl` | **Blind** set. 24 further held-out clauses, labeled after the checks were frozen and not used for tuning. |

The risk labels are the project author's reading of each clause against the illustrative playbook
in `playbook/playbook.yaml`. They are not legal advice, and no lawyer has reviewed them.
`needs_human` marks clauses where the deciding term is redacted, cut off, or incorporated from
another document, so a reviewer must look at the source.
