# RESULT CONTRACT — read before writing any code in this repo

This repo exists to produce **auditable** robotics evaluation results. Any agent
(OpenCode, Qwen3-Coder, a human) operating here agrees to the following. These
are enforced by CI (`scripts/selfcheck.py`, `scripts/validate_results.py`,
`tests/`), not just requested.

## 1. Never invent a result
- You may write code that *computes* a metric: `metrics = evaluate(model, test)`.
- You may **never** write a literal metric value (`accuracy = 0.987`) into a
  README, manifest, or doc unless it was produced by an executed experiment and
  lives in a validated result manifest.
- Result manifests are written **by the experiment script**, never by hand.
  `produced_by: manual` is rejected by the schema.

## 2. Every result declares its evidence class — no blurring
- `REAL-PUBLIC`  — metric computed on real, openly-released data (needs `dataset_sha256`).
- `SIMULATED`    — metric from a documented simulator/model (needs `dataset_sha256` of the generated set + the generator's git_sha).
- `CODE-VERIFIED`— math/software verification on our code (`dataset_sha256: n/a`).
- A metric that mixes classes is invalid. Report them as separate manifests.

## 3. No hardware claims
- ESP32/WiFi/micro-ROS latency, CPU load, packet loss, closed-loop vehicle
  improvement, on-device power — **out of scope**. This framework produces only
  the three classes above. Hardware numbers require real hardware measurement
  and live elsewhere, labelled `HARDWARE`.

## 4. Dataset label semantics must be verified before an adapter is implemented
- Every adapter starts `VERIFIED = False` and raises `AdapterNotVerified`.
- Before flipping it: read the source paper/record, write the real label meaning
  into `LABEL_SEMANTICS` (with citation), and decide either (a) a **documented**
  mathematical conversion, or (b) change the model to predict the source's actual
  target. **Never invent a conversion** to make the pipeline run.
- Terrain labels are not slip labels. `n_broken_propellers` is not a severity
  regression target unless the source says so.

## 5. Splits must respect grouping
- Use grouped/leave-one-out splits (by recording / flight / run / session).
- Random row or random window splits leak correlated neighbors — not allowed for
  headline metrics. The `split` field must name the real strategy.

## 6. Datasets are never committed
- Only registry YAML, download scripts, checksums, manifests, and adapters.

## 7. HPC is human-launched
- No script auto-submits `sbatch`. Propose the command; the human runs it.

## When unsure
Stop and ask the human. A missing result is fine. A fabricated or mislabelled
result is a failure of the whole framework.
