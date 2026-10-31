# Southern Song Woodblock Books — Reproducible Numeric Data

Research: **Quantifying Visual Change in Southern Song Woodblock Books through Regular Script Morphology**

This is an **image-free, isolated publication branch**. It contains only the final numeric analysis input and the code needed to re-create the statistical results; source facsimiles, cropped glyph images, OCR caches, review-state JSON, editable manuscripts, and historic raw versions are not included.

## Run

From the repository root, using Python 3.10+:

    python -m pip install -r requirements-repro.txt
    python -m src.paper_repro.reproduce --data data/paper_repro/npj_v1

All expected output tables are regenerated under data/paper_repro/npj_v1/reference_results/. The script verifies SHA-256 of input files and independently checks 22,468 numeric observations against the 9,043 book-character representatives.

The full schema, comparisons, metric formulas, date sensitivity, limits, and old/new version distinction are documented in data/paper_repro/npj_v1/README.md.

## Reproducibility scope

- 38 editions, 317 observed character types, 22,468 confirmed instances, 9,043 book-character representatives.
- Three morphological features: aspect ratio, relative character face size, black-to-white **pixel ratio** (NOT literal ink mass).
- Liangzhe near-Ou 18-book time sequence, Fujian six-book series, regional comparisons using period-specific or fixed shared character pools as appropriate.
- Exact or Monte Carlo permutation tests; exploratory 12-test BH multiplicity correction; ambiguous dating and up-to-three-instance sensitivities.
- No visual images are necessary to reproduce the reported numeric statistics. End-to-end re-extraction from page scans is not possible with the public dataset and would require separate permitted access.

## Important research caution

The nominal date for 中庸辑略 (1195) is a historical WORKING CHOICE, not independently verified as a precise date. Catalogue and later research disagree; alternatives and an exclusion scenario are provided. This release maintains the dataset and method transparently, without optimizing dates or sample selection for small P values.

The image-related research archive is retained in the separate private development workspace. The original development Git repository already had historic commits containing some confirmed images and review data. This **orphan branch** has NO ancestral commits and carries none of those raw research assets in its own history. It does not automatically purge the older development branch or the remote's existing Git history.

Source-image rights are not transferred by publication of these scalar measurements. No permission to redistribute original scans is asserted.
