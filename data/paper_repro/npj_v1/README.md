# NPJ Heritage Science — numeric reproducibility dataset (v1)

This is the image-free numerical layer derived from 38 Southern Song editions and 22,468 reviewed glyph instances. It reproduces published numerical comparisons, not OCR/segmentation from copyrighted page scans. Local raw pages, cropped images, and review JSON remain PRIVATE.

## Minimal public inputs

| File | Description |
| --- | --- |
| instance_metrics.csv | 22,468 glyph instances, book/character/instance ID, volume/page, width/height, and three measured values |
| book_character_metrics.csv | 9,043 book-character means derived from instance measurements |
| books.csv | 38-edition metadata, nominal and uncertain dates, script/style category, region, control metrics |
| analysis_plan.json | Frozen near-Ou 18-book list, Fujian six, period groups, shared-character rule, threshold and trimming |
| exceptions_above3.csv | 70 book-character groups with 4–5 instances (72 beyond a strict cap of three) |
| manifest.json | Input SHA-256 checksums, release lineage; no raw images |
| reference_results/ | Numeric outputs reproduced independently from these public CSVs |

No scan, source PDF, grayscale glyph image, OCR output, review status/decision, absolute personal path or raw acquisition file belongs in GitHub.

## Reproduce without image files

From repository root (Python 3.10+ and NumPy):

    python -m src.paper_repro.reproduce --data data/paper_repro/npj_v1

The replay script independently reaggregates all 22,468 instance values to exactly 9,043 book-character means, verifies each of the three values to 1e-12, then produces:

- Liangzhe related near-Ou/flattened-Ou 18-book temporal results on 109 shared characters;
- Fujian six-book temporal results on 105 shared characters;
- same-period regional contrasts (middle period: 89 common characters, late: 99);
- between-period regional contrasts using a fixed 77-character pool across both periods;
- exact/Monte Carlo two-sided Spearman permutation P, bootstrap rho CI, Theil–Sen slopes, Benjamini–Hochberg adjustment across 12 temporal tests;
- sensitivity for disputed dating and the occasional 4–5 instance book-character group.

Results are written to the reference_results folder. Statistics are exploratory until historical metadata review and sampling limits are addressed.

## Re-freeze from restricted local images (private workspace only)

    python -m src.paper_repro.freeze --bundle data/releases/npj-22468-current-candidate/analysis --out data/paper_repro/npj_v1

This requires local copyrighted/restricted images, and cross-verifies every book-character mean against the existing image-processing code. It does not publish images.

## Indicator semantics

- Aspect ratio = glyph height / width.
- Face ratio = (glyph width + height) / (2 * Lb), a LINEAR relative scale, not an area fraction.
- Black-to-white ratio = B / W, with B = pixels below gray threshold 160 in the Lb square frame, W = Lb squared minus B. Legacy scripts call it ink_coverage; it is NOT black/(black+white), actual ink quantity or stroke weight.
- Lb = ceil(1.05 * nearest-rank P90 of glyph long sides in an edition). Clipping rate records how often width or height exceeds this Lb.
- Common character IDs are intersected within EACH specified comparison. Instance values are averaged by book-character before giving each character equal weight. Book-level means and population standard deviations use independent 5% tail trimming by metric and edition; retained IDs may therefore differ across books.
- Book-level dispersion is across DIFFERENT character identities, not repeated-glyph consistency.

## Important dating uncertainty

- Book IDs are immutable and may contain obsolete year hints. Book 04 has documented 1132–1133 printing dates even though its ID contains 1131.
- year_nominal reproduces an earlier chronological coding choice; it does not certify exact dates.
- 中庸辑略 has a catalogue range 1195–1224, whereas a 2025 scholarly reconsideration suggests approximately 1248–1251. year_min / year_max cover the OUTER BOUNDS of alternative views, NOT one continuously supported interval. Scripts include 1195, 1196, 1224, 1249, and excluded scenarios.
- Other unreviewed_working_year values are carried over from the project metadata and are not claimed independently verified.
- Dates must not be silently recoded to improve the statistical results.

## Repeat glyph counts

70 out of 9,043 book-character groups have four or five instances, rather than the previously described strict maximum of three. They enter the main means. A fully deterministic first-three-instance sensitivity is provided; no valid instances have been deleted.

## Historical version separation

- TDJ v1 = 9,710 instances / 9,108 book-character values (Git d4509e22). Historical entries and old S1 available locally; the complete historical image bytes were not saved together.
- TDJ v2 = 22,468 / 9,043 in the 2026-06-10 old analysis image snapshot, with 30 stale copies.
- NPJ v1 numeric = 22,468 / 9,043 built from the now synchronized confirmed images, including the 30 corrected files.

Historical copies are in the LOCAL, Git-ignored data/releases folder; see root DATASET_VERSIONS.md. Do not mix results between versions.

## Rights and reproducibility limits

These derived CSVs permit reproducing numerical statistics from accepted character measurements. They do NOT permit independently replaying image segmentation, human validation or pixel thresholding without protected facsimile images. Source access must be arranged under the original collection's rights. No blanket license for the scanned pages is asserted.
