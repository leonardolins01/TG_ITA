"""Grava os CSV de `results/analysis/` e as tabelas de `latex/tables/` a partir
dos JSONL. Offline e determinístico.

Uso:
    python buildAnalysis.py            # todas as repetições completas
    python buildAnalysis.py --reps 2   # só as repetições 0 e 1
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import analysis, pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=None,
                        help="usar só as repetições 0..k-1 (padrão: todas as completas)")
    env = pipeline.Environment.load()
    reps = min(parser.parse_args().reps or analysis.completeReps(env),
               analysis.completeReps(env))
    print(f"lendo {reps} repetição(ões) completa(s) de {len(analysis.BENCHMARKS) * 15} "
          f"execuções em {env.path('results')}")
    print("(sem rede: os JSONL são lidos do disco, nada é chamado na API)\n")

    written = analysis.writeAll(env, reps=reps)

    for name, path in written.items():
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"  {name:11s} {rows:4d} linhas  ->  {path}")

    print(f"\npronto. {len(written)} arquivos em {written['summary'].parent}")

    import pandas as pd

    summary = pd.read_csv(written["summary"], encoding="utf-8")
    tables = analysis.writeLatexTables(env, summary)
    print("\ntabelas LaTeX:")
    for benchmark, path in tables.items():
        print(f"  {benchmark:11s}       ->  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
