# Relax-and-Gate

GPU macro placement for the Partcl/HRT Macro Placement Challenge.

The method uses a smooth, mass-conserving relaxation to propose global moves, then
accepts legal placements only when the exact benchmark objective improves.
Incremental scoring makes batches of discrete moves cheap to evaluate. It moves
both hard macros and soft cell clusters and uses no trained model at run time.

## Results

Mean proxy cost over all 17 IBM circuits; lower is better.

| Nominal search budget per circuit | Seeds | Mean proxy |
| --- | --- | --- |
| 900 seconds | 0 | 0.936626999225 |
| 300 seconds | 0, 1, 2 | 0.947672286085 |

All 68 saved runs pass the campaign's legality checks. The published challenge
winner's mean is 0.9507. These are comparisons with reported scores on different
hardware, not an official leaderboard result. Recorded elapsed times include
setup and final checks in addition to the nominal search budget.

The campaign ran on one NVIDIA L40S. The placer was frozen before the four test
circuits were scored. The split is in [SPLIT.json](SPLIT.json).

- [Paper](paper/manuscript.pdf)
- [Results page](https://numaro.tech/research/macro-placement-relax-and-gate-2026/)
- [Per-circuit scores and seed statistics](results-summary.json)
- [Verification request](https://github.com/partcleda/macro-place-challenge-2026/issues/124)
- [Challenge rules and leaderboard](https://github.com/partcleda/macro-place-challenge-2026)

## Verify the saved placements

The CPU verifier reloads the original benchmark netlists, recomputes proxy cost
from the saved coordinates, and checks overlaps, bounds, and fixed macros. It
imports no search code and needs no GPU or PyTorch.

Use Python 3.11 and clone the public challenge next to `verify.py`:

```bash
git clone https://github.com/unnir/relax-and-gate.git
cd relax-and-gate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-verify.txt

git clone https://github.com/partcleda/macro-place-challenge-2026.git challenge
git -C challenge checkout 996193c83eaad151e5ae3fb166cedbd83372d060
git -C challenge submodule update --init external/MacroPlacement

python verify.py artifacts/final900/results.json
python verify.py artifacts/seeds300/results.json
```

The pinned challenge uses evaluator commit
`45a721d01dfe56fc4800b95da52e9b4193b76d04`. The SHA-256 of
`CodeElements/Plc_client/plc_client_os.py` is
`49e5d0bb58fcb8e9ae4666bf1e11f8ba69baa49c2aa314cad82de8874237b29c`,
matching the campaign manifest. Benchmark data and third-party code remain in
the separate challenge checkout under their upstream licenses.

Coordinates are stored in each results file under `rows[].positions`, in hard
macro order followed by soft macro order. No separate coordinate download is
needed. `verify.py` supports this multi-seed schema. `verify-frozen.py` preserves
the earlier verifier, which expects the original `circuits` schema.

## Run a new placement

The original search environment was Python 3.11, PyTorch 2.4.1 with CUDA 12.4,
and an NVIDIA L40S. Install the matching CUDA build of PyTorch in addition to the
verification dependencies:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python mpx/extract.py
python mpx/validate.py ibm01
python run_full.py --circuits ibm01 --budget 300 --seeds 0 --tag repro
python verify.py artifacts/repro/results.json
```

`best_params.json` contains the parameters used for the saved runs. For a full
17-circuit sweep, use `--split all` instead of `--circuits ibm01`.

## Source and evidence

The algorithm files are preserved from the original method-freeze commit
`f264d1f8e0cd6efb82b72f260a87e1ab566f0a21` in `unnir/autoresearch`, under
`tasks/macro_placement`. Results, the later verifier, and campaign verdicts come
from snapshot `e80414af564a6f309189ecfb184201e638d0b652`.
This repository contains the released files rather than the private repository's
history. File provenance is recorded in [PROVENANCE.json](PROVENANCE.json).

- `solution.py`, `mpx/`: frozen placement algorithm and scorers.
- `artifacts/final900/`: all 17 placements at the 900-second budget.
- `artifacts/seeds300/`: all 51 placements at the 300-second budget.
- `paper/data/`: recorded verdicts and the paired-intervention table.
- `artifacts/ppa_sweep/`: six-placement OpenROAD study on ariane136.

The campaign reports standalone reference-evaluator verification. Release
preparation checked file hashes, Python syntax, score arithmetic, and aggregate
means; it did not rerun the GPU campaign or OpenROAD flows.

The physical study is limited: only the power correlation is significant at
`p < 0.05`, and timing does not bind. A lower proxy alone does not establish a
better chip. Raw intervention logs did not survive; the package retains the
recorded verdicts used in the paper.

Check the released file hashes with:

```bash
shasum -a 256 -c SHA256SUMS
```
