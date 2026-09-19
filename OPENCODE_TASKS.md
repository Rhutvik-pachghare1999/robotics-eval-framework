# OPENCODE_TASKS — gated workflow

Run these **one step at a time** through OpenCode (primary: ASU Qwen3-Coder;
second-opinion reviewer: an OpenCode Zen / OpenRouter model). Each step has a
STOP gate. Do not let the agent proceed past a gate without your explicit OK.

> Paste this at the start of the OpenCode session:
> "Follow RESULT_CONTRACT.md. Do exactly one STEP from OPENCODE_TASKS.md, then
>  stop and report. Never invent results. Never flip an adapter's VERIFIED flag
>  without my confirmation of the source label semantics."

---

## STEP 0 — sanity (no gate)
- Run `python scripts/selfcheck.py` and `pytest tests/`. Both must pass.
- Confirm `python scripts/fetch_dataset.py --list` shows the 6 starter datasets.

## STEP 1 — pick ONE project to start
Recommended order (public-data-heavy first): **UAV → Edge AL → NeuroTraction → Safety → DriftBot**.
🚦 GATE: you choose the project. Agent does nothing else until you name it.

## STEP 2 — dataset provenance (per dataset, one at a time)
- Agent fills the registry YAML's real `url`, `license`, `citation` from the
  actual source (it may fetch the page to read it) — but marks anything it
  cannot confirm as `VERIFY`.
- You manually download (most are login/click-through) into `data/source/<id>/`.
- Run `python scripts/fetch_dataset.py <id>` to record `dataset_sha256`.
🚦 GATE: you confirm the license permits your use and the download is complete.

## STEP 3 — VERIFY label semantics (the critical human gate)
- Agent reads the source paper/record and writes a proposed `LABEL_SEMANTICS`
  into the adapter (what the target actually means; how/if it maps to the
  project target). It does NOT flip `VERIFIED`.
🚦 GATE: **you** read the agent's summary against the source and either correct
it or approve. Only then does the agent set `VERIFIED = True`.

## STEP 4 — implement the adapter
- Implement `load()` to emit standardized samples + a group id (recording/flight/run).
- Add a tiny test on a few real samples (shapes, label ranges, group ids present).
🚦 GATE: adapter test passes on real downloaded data.

## STEP 5 — baselines before neural nets
- Implement the simple baselines first (LogReg/RandomForest/linear), grouped split.
- Write an experiment module that emits a **result manifest** (never prints a
  hand-written number). `evidence_class: REAL-PUBLIC`.
🚦 GATE: `python scripts/validate_results.py <manifest>` passes locally on CPU
with a tiny subset.

## STEP 6 — propose the HPC sweep (do NOT submit)
- Agent writes/edits a `slurm/*.sbatch` + config for the real sweep and prints
  the exact `sbatch` command and the estimated GPU-hours.
🚦 GATE: **you** launch it on SOL. Start with a 4h `htc` smoke job, not the full sweep.

## STEP 7 — external / cross-domain validation
- Only after in-domain results exist: run the external test (e.g. train
  DronePropA → test TII) as a **separate** manifest. Never merge into the
  in-domain number.

## STEP 8 — write results into the project README
- Numbers come ONLY from validated manifests, each tagged with its evidence
  class, split, seed, git_sha, dataset_sha256. Second-opinion model reviews the
  claim/evidence table for any blurred category.
🚦 GATE: you approve the README diff before it is committed/pushed.

---

### Second-opinion loop (every code step)
primary implementer (Qwen3-Coder) → second reviewer (Zen/DeepSeek) reviews diff
for invented numbers / blurred evidence classes / random splits → you decide →
`pytest` → commit. Never merge on a single model's say-so.

### What "done" looks like per project
A claim/evidence table where every row is reproducible from the repo state:
`evidence_class | dataset | split | metric | seed | git_sha | dataset_sha256`.
