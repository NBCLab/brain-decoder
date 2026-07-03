# Reviewer Analysis Plan

This document translates the commitments in `review_response.md` into a concrete analysis plan for `brain-decoder`.

## Overview

The reviewer response implies three kinds of work:

1. Reproduce and harden the existing decoding pipeline so the current HCP, IBC, and ROI analyses can be rerun reliably.
2. Add missing analyses that were explicitly promised in the response letter.
3. Package the outputs into figures, tables, and supplementary artifacts that map cleanly onto manuscript revisions.

## Reflection Rule

After each task in this plan is completed, write a short reflection before moving on to the next task. Each reflection should include:

1. What worked.
2. What did not work or remained ambiguous.
3. The next checks or follow-up analyses still needed to fully answer the reviewer's question tied to that task.

Treat these reflections as part of the deliverable for each task, not as optional notes.

## Current Repo Status

The repository already contains partial scaffolding for several decoding analyses:

- `jobs/decoding_hcp_nv.py` for HCP group-map decoding.
- `jobs/decoding_eval.py` for aggregate HCP evaluation.
- `jobs/decoding_ibc.py` for IBC decoding.
- `jobs/decoding_seeds.py` for ROI decoding.
- `braindec/predict.py` for task/concept/domain decoding.
- `braindec/cogatlas.py` for ontology construction and task/concept/domain mappings.

There are also important gaps that should be treated as prerequisites:

- `jobs/decoding_cnp.py` is referenced in `review_response.md` but does not exist in the repo.
- `jobs/decoding_ibc.py` and `jobs/decoding_seeds.py` appear stale relative to the current `image_to_labels_hierarchical` API.
- Multiple job scripts hard-code `project_dir` and `device`, which will block reproducible reruns.
- There is no script yet for per-term analysis, null/permutation baselines, embedding geometry, or SNR/sample-size sweeps.

## Recommended Execution Order

1. Stabilize the decoding jobs and evaluation interfaces.
2. Rerun and document the current HCP benchmark.
3. Add cross-dataset decoding for IBC and CNP.
4. Add term-level, ontology, and chance-normalized analyses.
5. Add embedding-geometry and SNR analyses.
6. Add the targeted emotion and striatum follow-up analyses.
7. Optionally run an NSD pilot if data are available and time permits.

## Task 0: Pipeline Hardening

### Goal

Make the current decoding and evaluation scripts runnable, configurable, and consistent with the current `braindec` APIs.

### Non-goals

- Changing model architecture.
- Re-training the CLIP model.
- Refactoring the whole repository.

### Concrete Implementation Plan

1. Update `jobs/decoding_hcp_nv.py`, `jobs/decoding_ibc.py`, `jobs/decoding_seeds.py`, and `jobs/decoding_eval.py` to accept CLI arguments for `project_dir`, `data_dir`, `results_dir`, `device`, `section`, `model_id`, and `reduced`.
2. Fix stale calls to `image_to_labels_hierarchical` so all jobs pass a `CognitiveAtlas` object rather than raw arrays.
3. Remove hard-coded `mps` device usage and default to a CLI/device helper.
4. Standardize output directory layout across HCP, IBC, CNP, ROI, and future analyses.
5. Verify that ground-truth files are loaded from the correct dataset-specific locations.
6. Add a small README section or module docstring describing expected inputs and outputs for each analysis job.

### Deliverables

- Runnable decoding jobs.
- Consistent prediction CSV outputs.
- A stable base for all downstream analyses.

## Task 1: HCP Benchmark Reproduction And Documentation

### Goal

Reproduce the core HCP group-map decoding results and explicitly document the task-level mapping used in the manuscript.

### Non-goals

- Expanding the benchmark beyond the currently selected representative task-domain contrasts.
- Fairness comparison with external IBMA models on HCP.

### Concrete Implementation Plan

1. Confirm the seven representative HCP contrasts used for task-level evaluation:
   - Emotion: Faces vs Shapes
   - Gambling: Reward vs Baseline
   - Language: Story vs Math
   - Motor: Average
   - Relational: Relational vs Match
   - Social: TOM vs Random
   - Working Memory: 2-Back vs 0-Back
2. Encode this mapping in a dedicated metadata file rather than leaving it implicit in filenames.
3. Rerun `jobs/decoding_hcp_nv.py` after pipeline hardening.
4. Update `jobs/decoding_eval.py` so it reads the explicit mapping file and computes task, concept, and domain metrics deterministically.
5. Export a supplementary table containing:
   - HCP image id
   - selected contrast
   - mapped task
   - mapped concepts
   - mapped domain
   - top-k decoded predictions

### Deliverables

- Reproducible HCP benchmark outputs.
- Supplementary mapping table for task-to-concept/domain interpretation.

## Task 2: Cross-Dataset Generalization On IBC

### Goal

Show that NiCLIP generalizes beyond HCP by decoding IBC statistical maps.

### Non-goals

- Full IBC methodological harmonization.
- Training on IBC.

### Concrete Implementation Plan

1. Repair and parameterize `jobs/decoding_ibc.py`.
2. Create an explicit IBC ground-truth mapping file from image names to task, concept, and domain labels.
3. Reuse the same evaluation interface as HCP so metrics are comparable.
4. Produce aggregate metrics:
   - task Recall@4
   - concept Recall@4
   - domain Recall@2
5. Break out results by task family where possible, especially emotion-related tasks if present.
6. Export per-image prediction tables for supplement use.

### Deliverables

- IBC prediction outputs.
- IBC summary metrics.
- Cross-dataset comparison against HCP.

## Task 3: Cross-Dataset Generalization On CNP

### Goal

Add the missing CNP benchmark promised in the response letter.

### Non-goals

- Broad psychiatric interpretation beyond task decoding.
- Subject-level clinical modeling.

### Concrete Implementation Plan

1. Add a new `jobs/decoding_cnp.py` modeled after the HCP and IBC jobs.
2. Define a CNP ground-truth mapping file for tasks such as BART, PAM encoding/retrieval, SCAP, Stop Signal, and Task Switching.
3. Ensure the CNP job writes outputs into a dataset-specific predictions directory with the same CSV schema used elsewhere.
4. Add or extend an evaluation script to score CNP task/concept/domain Recall@K.
5. Summarize which CNP tasks transfer well and which do not.

### Deliverables

- New CNP decoding job.
- CNP evaluation outputs.
- Cross-dataset generalization section inputs.

## Task 4: Fine-Grained Emotion Analysis

### Goal

Test whether NiCLIP can decode more specific emotional states than the coarse HCP emotion contrast.

### Non-goals

- Open-ended text generation of emotions.
- Claiming fine-grained emotion decoding if ontology coverage is weak.

### Concrete Implementation Plan

1. Identify emotion-specific contrasts in IBC and/or CNP that are more granular than `Faces vs Shapes`.
2. Audit the current Cognitive Atlas task and concept vocabulary for emotion-specific labels such as fear, anger, disgust, and happiness.
3. If needed, extend the vocabulary-generation path so emotion concepts and task definitions are included in the decoding vocabulary.
4. Generate vocabulary embeddings and priors for the expanded emotion-related vocabulary.
5. Decode the selected emotion maps and compare:
   - coarse emotion-domain predictions
   - emotion-concept predictions
   - fine-grained task predictions
6. Report whether failures are due to ontology coverage, training-data sparsity, or embedding mismatch.

### Deliverables

- Emotion-specific decoding results.
- A clear statement of current granularity limits.

## Task 5: Per-Term Decodability Analysis

### Goal

Quantify which individual tasks are actually decodable, rather than reporting only averaged benchmark scores.

### Non-goals

- Estimating calibrated probabilities.
- Claiming full coverage of the entire ontology if only a subset is evaluable.

### Concrete Implementation Plan

1. Add a new analysis script, for example `jobs/per_term_eval.py`.
2. Aggregate prediction outputs from HCP, IBC, and CNP into a single evaluation table.
3. For each task in the reduced ontology, compute:
   - number of evaluation maps tied to that task
   - Recall@K
   - rank statistics for the ground-truth term
4. Produce:
   - a per-task bar chart or heatmap
   - a table of best-decoded and worst-decoded tasks
5. Explicitly mark tasks with insufficient evaluation examples so they are not over-interpreted.

### Deliverables

- Per-task decodability table.
- Figure summarizing term-level decoding performance.

## Task 6: Null Baseline And Normalized Accuracy

### Goal

Contextualize raw task/concept/domain scores with chance-level and permutation-based baselines.

### Non-goals

- Formal probability calibration.
- Bayesian model comparison.

### Concrete Implementation Plan

1. Add a permutation procedure that shuffles ground-truth label assignments within each dataset.
2. Recompute Recall@K under the null distribution for tasks, concepts, and domains.
3. Estimate:
   - empirical chance baseline
   - p-value or percentile above null
   - normalized accuracy = observed / chance
4. Report how many terms are above chance in the per-term analysis.
5. Add a simple table comparing observed versus chance performance across label levels.

### Deliverables

- Permutation-based baseline outputs.
- Chance-normalized metrics.
- Count of above-chance decodable terms.

## Task 7: Factors Associated With Per-Term Performance

### Goal

Explain why some terms decode better than others.

### Non-goals

- Causal claims about ontology quality or article frequency.
- Full statistical modeling of all confounds.

### Concrete Implementation Plan

1. Build a term-level analysis table with one row per task.
2. Add candidate explanatory variables:
   - number of associated training articles
   - definition length
   - embedding norm or other simple text-feature proxies
   - map specificity measures derived from evaluation maps
3. Compute correlations and simple regression analyses between these variables and per-term Recall@K.
4. Report whether article frequency, definition quality/length, or map specificity appears most associated with performance.

### Deliverables

- Term-level feature table.
- Correlation/regression summary figure or table.

## Task 8: Reduced Versus Full Cognitive Atlas Comparison

### Goal

Substantiate the claim that the reduced/curated ontology improves decoding.

### Non-goals

- Rebuilding the ontology from scratch.
- Exhaustive ontology curation.

### Concrete Implementation Plan

1. Export summary statistics for the full and reduced ontology:
   - number of tasks
   - number of concepts
   - task-concept edges
   - concept-domain edges
2. Rerun a matched decoding/evaluation comparison under both ontologies.
3. Compare performance for each model/configuration where the ontology changes are the only intended difference.
4. Produce a supplementary table documenting what was retained, removed, or enriched in the reduced ontology.

### Deliverables

- Ontology comparison table.
- Performance comparison under full versus reduced CogAt.

## Task 9: Embedding Geometry Analysis

### Goal

Inspect the shared latent space to determine whether aligned task and image embeddings form meaningful structure.

### Non-goals

- Treating 2D projections as definitive evidence.
- Over-interpreting global geometry from UMAP/t-SNE alone.

### Concrete Implementation Plan

1. Add a new script, for example `jobs/embedding_geometry.py`.
2. Extract:
   - task text embeddings from the decoding vocabulary
   - HCP image embeddings from the trained CLIP image encoder
3. Produce 2D projections with UMAP and/or t-SNE.
4. Color points by:
   - cognitive domain for task embeddings
   - task family for image embeddings
5. Quantify auxiliary structure:
   - within-domain versus between-domain distances
   - distance from each image embedding to its matched task embedding
   - comparison of geometry across BrainGPT, Mistral, and Llama variants

### Deliverables

- Supplementary latent-space figure.
- Distance-based summary table.

## Task 10: SNR / Group-Size Sensitivity Analysis

### Goal

Test whether subject-level underperformance is mainly due to lower SNR by varying HCP group size.

### Non-goals

- Solving subject-level decoding in this revision.
- Training with synthetic noise augmentation unless needed as a later follow-up.

### Concrete Implementation Plan

1. Identify or generate HCP subject-level maps for the benchmark tasks.
2. Build group-average maps for subset sizes `N = {5, 10, 20, 50, 100, 200, 787}`.
3. For each group size:
   - sample multiple subsets if possible
   - decode each averaged map
   - compute mean and variance of Recall@K
4. Plot decoding performance versus group size.
5. Compare the trend against the qualitative covariate-shift explanation to determine whether noise alone explains the gap.

### Deliverables

- Group-size sensitivity figure.
- Quantitative statement about SNR versus text/image mismatch.

## Task 11: ROI Follow-Up And Striatum Hypothesis Example

### Goal

Use the existing ROI decoding setup to produce a concrete hypothesis-generation example centered on striatum-language associations.

### Non-goals

- Claiming novel neurobiological discovery solely from the decoder.
- Replacing literature validation with decoding outputs.

### Concrete Implementation Plan

1. Repair and rerun `jobs/decoding_seeds.py`.
2. Export top task, concept, and domain predictions for each ROI.
3. For striatum specifically, record:
   - top semantic/language tasks
   - top concepts
   - top domain probabilities
4. Package the striatum results into a concise table suitable for the manuscript and a separate table for supplement use.
5. Hand off the exact predicted labels and scores for downstream literature validation and meta-analytic discussion.

### Deliverables

- Updated ROI decoding outputs.
- Striatum-specific results table for the hypothesis-generation example.

## Task 12: Optional NSD Pilot

### Goal

Probe the boundary of model generalization on a natural-scene perception dataset.

### Non-goals

- Treating NSD as a core benchmark.
- Claiming scene-level semantic decoding if the ontology is mismatched.

### Concrete Implementation Plan

1. Only proceed if NSD maps are already available in a usable format.
2. Define a small pilot set of maps expected to load strongly on visual perception.
3. Run decoding with the existing perception-related ontology terms.
4. Report the pilot qualitatively and quantitatively as a boundary case, not a headline result.

### Deliverables

- Optional appendix-level NSD pilot result.

## Suggested New Files

- `jobs/decoding_cnp.py`
- `jobs/per_term_eval.py`
- `jobs/null_baseline.py`
- `jobs/embedding_geometry.py`
- `jobs/snr_sweep.py`
- `data/.../ground_truth_*.json` or `.csv` files for HCP, IBC, and CNP mappings
- `results/.../tables/` and `results/.../figures/` subdirectories for manuscript-ready outputs

## Minimal Milestone Cut

If time is constrained, the minimum credible set for the revision is:

1. Task 0: Pipeline hardening.
2. Task 1: HCP benchmark reproduction.
3. Task 2: IBC cross-dataset analysis.
4. Task 3: CNP cross-dataset analysis.
5. Task 5: Per-term decodability.
6. Task 6: Null/chance baseline analysis.
7. Task 8: Reduced versus full ontology comparison.
8. Task 10: SNR/group-size analysis.

The embedding geometry, striatum deep dive, emotion expansion, and NSD pilot are valuable but can be treated as secondary if deadlines are tight.
