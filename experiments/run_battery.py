"""Roda a bateria de `env.benchmark` fora do notebook, com retomada.

Uso:
    python run_battery.py            # retoma de onde parou
    python run_battery.py --restart  # ignora o que já existe e refaz tudo
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import datasets, llm, metrics, pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                        help="refaz as execuções do zero em vez de retomar")
    args = parser.parse_args()

    env = pipeline.Environment.load()
    check = llm.checkApi(env)
    if not check["ok"]:
        print(f"ABORTADO: a API não respondeu ({check['error']}).", flush=True)
        return 1

    runs = pipeline.battery(env)
    cost = pipeline.estimateBatteryCost(env, runs)
    print(f"benchmark={env.benchmark} amostra={env.sampleSize} seed={env.seed}", flush=True)
    print(f"{len(runs)} execuções | custo estimado US$ {cost['TOTAL']:.4f}", flush=True)
    print(f"uso acumulado na conta antes de começar: US$ {check['usage']}", flush=True)
    print("-" * 72, flush=True)

    items = datasets.loadBenchmark(env.benchmark, seed=env.seed,
                                   sampleSize=env.sampleSize)
    print(f"{len(items)} itens carregados", flush=True)

    started = time.time()
    summaries = []
    for n, run in enumerate(runs, 1):
        t0 = time.time()
        print(f"[{n:2d}/{len(runs)}] {run.fileId()} ...", flush=True)
        path = pipeline.runOne(env, run, items=items,
                               resume=not args.restart, progress=False)
        records = pipeline.loadRecords(path)
        summary = metrics.summarize(records, run)
        summaries.append(summary)
        row = summary.asRow()
        print(f"[{n:2d}/{len(runs)}] concluida em {time.time() - t0:7.1f}s | "
              f"n={row['n']} asr={row['asr']:.3f} utility={row['utility']:.3f} "
              f"overBlock={row['overBlock']:.3f} invalid={row['invalidOutput']:.3f} "
              f"marked={row['markedRate']:.3f}", flush=True)

    print("-" * 72, flush=True)
    print(f"BATERIA COMPLETA em {(time.time() - started) / 60:.1f} min", flush=True)
    after = llm.checkApi(env)
    if after["ok"] and after.get("usage") is not None:
        print(f"uso acumulado depois: US$ {after['usage']} "
              f"(gasto nesta bateria: US$ "
              f"{after['usage'] - check['usage']:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
