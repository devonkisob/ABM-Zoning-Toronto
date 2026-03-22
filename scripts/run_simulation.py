"""
run_simulation.py
-----------------
Runs all four scenarios and saves results to data/processed/.

Usage:
    python scripts/run_simulation.py           # N=100, T=40 (default)
    python scripts/run_simulation.py --fast    # N=10, T=40 (quick test)
    python scripts/run_simulation.py --full    # N=500, T=40 (final report)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import argparse
import time
from src.simulation import run_scenario, SCENARIOS

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="Quick test run (N=10)")
parser.add_argument("--full", action="store_true", help="Full run for final report (N=500)")
parser.add_argument("--scenario", type=str, default=None,
                    help="Run a single scenario e.g. --scenario S1")
args = parser.parse_args()

N = 10  if args.fast else (500 if args.full else 100)
T = 40  # 10 years × 4 quarters
scenarios = [args.scenario] if args.scenario else SCENARIOS

OUT_DIR = Path("data/processed")

# ── Run ────────────────────────────────────────────────────────────────────────
print(f"Running {len(scenarios)} scenario(s) | N={N} realisations | T={T} steps")
print(f"Estimated time: ~{len(scenarios) * N * T * 0.001 / 60:.1f} min\n")

total_start = time.time()

for scenario in scenarios:
    start = time.time()
    results = run_scenario(scenario, N=N, T=T, verbose=True)
    elapsed = time.time() - start

    out_path = OUT_DIR / f"results_{scenario.lower()}.npy"
    np.save(out_path, results)
    print(f"  Saved → {out_path} ({elapsed:.1f}s)\n")

total = time.time() - total_start
print(f"All done in {total/60:.1f} min")