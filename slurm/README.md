# SLURM templates (ASU SOL)

These are **templates**, not auto-run jobs. Nothing in this repo submits SLURM
jobs for you — HPC time is real allocation, so a human always launches.

## One-time env setup (on a login node)
```bash
mamba create -n robeval python=3.11 -y
mamba activate robeval
pip install -e .
```

## Launch a job (you do this, deliberately)
```bash
sbatch slurm/experiment.sbatch experiments.uav.run_fault_clf configs/uav_dronepropa.yaml
```

## Rules
- Request the **smallest GPU** that fits (a30 / a100.20gb before a full a100/h100).
- `htc` partition (4h) for quick checks; `public`/`general` for real sweeps.
- Every job writes a result manifest and the job itself validates provenance.
- Never launch a million-scenario sweep as a first run — do a 4h `htc` smoke job first.
