"""
run_simulation.py
-----------------
Runs all four scenarios and saves results to data/processed/.

Usage:
    python scripts/run_simulation.py              # N=100, T=40, parallel
    python scripts/run_simulation.py --fast       # N=10, T=40 (quick test)
    python scripts/run_simulation.py --full       # N=500, T=40 (final report)
    python scripts/run_simulation.py --no-parallel  # sequential (debug)
    python scripts/run_simulation.py --scenario S1  # single scenario
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
import time
from multiprocessing import Pool, cpu_count
from src.simulation import run_scenario, SCENARIOS

OUT_DIR = Path("data/processed")


# ── Worker (must be top-level for multiprocessing on macOS) ───────────────────
def run_and_save(args_tuple):
    scenario, N, T = args_tuple
    results = run_scenario(scenario, N=N, T=T, verbose=True)
    out_path = OUT_DIR / f"results_{scenario.lower()}_final_calibration.npy"
    np.save(out_path, results)
    return f"Saved {scenario} → {out_path}"


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",        action="store_true", help="Quick test (N=10)")
    parser.add_argument("--full",        action="store_true", help="Final report (N=500)")
    parser.add_argument("--no-parallel", action="store_true", help="Run sequentially")
    parser.add_argument("--scenario",    type=str, default=None,
                        help="Single scenario e.g. --scenario S1")
    args = parser.parse_args()

    N = 10 if args.fast else (500 if args.full else 100)
    T = 40
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    print(f"Running {len(scenarios)} scenario(s) | N={N} | T={T} | "
          f"parallel={'no' if args.no_parallel or len(scenarios)==1 else 'yes'}")

    total_start = time.time()

    if args.no_parallel or len(scenarios) == 1:
        for scenario in scenarios:
            msg = run_and_save((scenario, N, T))
            print(msg)
    else:
        n_workers = min(len(scenarios), cpu_count())
        print(f"Using {n_workers} parallel workers ({cpu_count()} CPUs available)\n")
        with Pool(processes=n_workers) as pool:
            messages = pool.map(run_and_save, [(s, N, T) for s in scenarios])
        for msg in messages:
            print(msg)

    total = time.time() - total_start
    print(f"\nAll done in {total/60:.1f} min")