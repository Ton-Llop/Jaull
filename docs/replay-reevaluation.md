# Replay and Re-evaluation

Jaull distinguishes offline re-evaluation from physical re-execution.

## Re-evaluation

Re-evaluation uses the inputs stored in an `ExperimentRecord` and runs Jaull's
current prediction and comparison code against them:

```text
frozen experiment input
        ↓
current prediction logic
        ↓
recorded observation
        ↓
current PredictionComparison
```

It does not detect the current host, query Hugging Face, download artifacts,
run llama.cpp, run PyTorch or mutate the original experiment record.

The purpose is to compare:

- what Jaull predicted when the experiment was recorded;
- what physically happened during that recorded run;
- what Jaull predicts now for the same frozen inputs;
- how the current comparison rules interpret the old observation.

Historical observations are immutable evidence. Prediction logic and evaluation
rules may evolve, but the original prediction, original comparison and recorded
observation remain separate from the current re-evaluation result.

## Exact replay

Exact replay is not implemented. It would require stronger provenance than the
current offline harness guarantees, including the historical Jaull code,
runtime build, driver stack and any external metadata used by the original
prediction.

Jaull should not call an approximate reconstruction an exact replay.

## Approximate reconstruction

Some records are useful but not fully reproducible. Examples:

- base-model enrichment depended on external metadata;
- the saved prediction input includes explicit reproducibility notes;
- older records have no `prediction_input` snapshot.

These records are reported as `approximate_only` or `not_reproducible` with
explicit reasons. Jaull must not fill missing metadata from the current host,
from Hugging Face or from mutable caches and present that as the original input.

Artifact identity certainty is reported separately from prediction
reproducibility. If the frozen model analysis, inference configuration,
hardware profile and artifact sizing metadata are present, Jaull can recompute
the prediction even when `ModelArtifact.sha256` is missing. In that case the
result remains prediction-reproducible but carries a warning that the physical
file identity is not cryptographically verified.

## Current limits

The first re-evaluation harness is prediction-only. It does not re-run a model
or benchmark and it does not calibrate Hardware Fit formulas.

Records created before `ExperimentPredictionInput` existed may still load, but
they cannot be re-evaluated with the current estimator unless the frozen model
analysis and inference configuration are present.
