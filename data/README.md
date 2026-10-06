# data

Everything the pipeline reads or writes, in one place. Nothing here is code, and
nothing here is part of the application's logic — `app/core/config.py` decides
where each of these directories lives, and every one of them can be pointed
elsewhere with an environment variable.

| Directory | Tracked? | What it is |
|---|---|---|
| `datasets/` | **yes** | The approved evaluation set. Curated and checksummed, not generated. |
| `manifests/` | **yes** | Provenance and checksums for the datasets. |
| `samples/` | no | Generated synthetic fixtures. Rebuildable. |
| `runs/` | no | One directory per analysis. The entire runtime output. |
| `uploads/` | no | Incoming videos, before they are moved into a run. |

The rule: **`datasets/` and `manifests/` are inputs, the rest are outputs.** A
dataset may only be added deliberately, with a manifest entry; an output may be
deleted at any time and nothing breaks.

## runs/ — one directory per analysis

```
data/runs/<analysis_id>/
  source.mp4        the uploaded video, moved here after validation
  tracking.json     what the single pass measured, before any inference
  result.json       the contract: deliveries, quarantine, calibration, summary
  clips/            delivery_NNN.mp4
  overlays/         delivery_NNN.mp4 with trajectory and markers drawn on it
```

`result.json` is the API contract (`docs/API_CONTRACT.md`). An analysis survives a
server restart as far as the filesystem is concerned: `/api/analysis/{id}/status`
reports `completed` from disk even when the job is not in memory.

Delivery IDs are assigned during segmentation and are **never renumbered**. A clip
filename means the same thing for the life of the run, so
`clips/delivery_007.mp4` in a result document is always the same physical clip.

Deleting a run directory deletes the analysis and nothing else.

The served URL prefix is `/data/runs/<analysis_id>`, mounted by `main.py`.

## datasets/source_clips/ — the approved 41-clip set

Delivery clips from one full stadium match, each with the upstream pipeline's shot
label and tracking outcome. Used as the regression set for segmentation and shot
classification.

```
data/datasets/source_clips/
  delivery_001.mp4 … delivery_041.mp4
  match_pipeline_results.json     upstream per-clip results, as received
```

`manifests/source_clips.json` carries the size and SHA-256 of every clip, the
upstream project/version it came from, and a per-clip summary of the upstream
label. Regenerate it after adding or replacing a clip.

Two things to know before using these as a measurement:

- **The labels are model output, not human annotation.** They are the upstream
  pipeline's predictions. Useful for regression and consistency checks; not a
  ground-truth accuracy figure.
- **They come from one match, one camera angle.** A 41-clip set from a single
  broadcast feed is enough to catch a regression and not enough to characterise
  performance. No accuracy claim in this repository rests on it alone.

The absolute paths inside `match_pipeline_results.json` record where the clips
originally came from and are not valid on this machine. Nothing reads them.

## samples/ — generated fixtures

`full_fixture.mp4` is a synthetic composite clip built by
`scripts/dataset/build_test_fixture.py`, which stitches source clips together so
a segmentation edge case can be reproduced deterministically. It is gitignored
because it is derived:

```bash
python scripts/dataset/build_test_fixture.py
```

## Environment overrides

| Variable | Default |
|---|---|
| `CRICKET_DATA_DIR` | `<project>/data` |
| `CRICKET_MODELS_DIR` | `<project>/backend/models` |
| `CRICKET_UPLOADS_DIR` | `<CRICKET_DATA_DIR>/uploads` |
| `CRICKET_RUNS_DIR` | `<CRICKET_DATA_DIR>/runs` |
| `CRICKET_SAMPLES_DIR` | `<CRICKET_DATA_DIR>/samples` |
| `CRICKET_DATASETS_DIR` | `<CRICKET_DATA_DIR>/datasets` |
| `CRICKET_MANIFESTS_DIR` | `<CRICKET_DATA_DIR>/manifests` |

A relative value is resolved against the project directory, so a service started
from any working directory finds the same data. `CRICKET_RUNS_DIR` is what a
container points at its mounted volume.

Note the vocabulary split: the API and the persisted documents say **analysis**
(the domain term); the directory says **runs** (generated output, safe to delete).
The two meet only in `app/core/config.py:analysis_root`.