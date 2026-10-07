"""Refaz, item a item, os registros com erro operacional do provedor e grava no lugar.

Erro operacional é falha da infraestrutura, e não resposta do modelo: término
`error` (o provedor falhou no meio da geração) ou `content_filter` (moderação do
provedor sorteado) em qualquer estágio. O item é refeito inteiro, filtro e
componente, na mesma posição do arquivo; resposta vazia ou inválida de um modelo
que terminou normalmente não é tocada. Copia `results/runs/` antes.

Uso:
    python fix_provider_errors.py --dry-run   # lista os registros e não chama nada
    python fix_provider_errors.py             # cópia, refação e tempos do provedor
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import datasets, llm, pipeline  # noqa: E402
from rejudge_runs import backup, parseFileId, writeAtomic  # noqa: E402

OPERATIONAL = ("error", "content_filter")
ATTEMPTS = 5


def isOperational(rec: dict) -> bool:
    """Diz se algum estágio do registro terminou por falha do provedor."""
    return rec.get("filterFinish") in OPERATIONAL or rec.get("componentFinish") in OPERATIONAL


def runOf(path: Path) -> pipeline.Run:
    """A execução que gerou o arquivo, a partir do nome."""
    meta = parseFileId(path.stem)
    component = meta["component"] if "+" in path.stem.split("__")[0] else ""
    return pipeline.Run(meta["condition"], meta["filter"], meta["benchmark"],
                        componentModel=component, seed=meta["seed"], rep=meta["rep"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="lista os registros com erro operacional; não grava nada")
    args = parser.parse_args()

    env = pipeline.Environment.load()
    runsDir = env.path("results")
    pending = {}
    for path in sorted(runsDir.glob("*.jsonl")):
        if llm.getModel(parseFileId(path.stem)["filter"]).role == "retired":
            continue  # fora da bateria
        idx = [i for i, r in enumerate(pipeline.loadRecords(path)) if isOperational(r)]
        if idx:
            pending[path] = idx
    total = sum(len(v) for v in pending.values())
    print(f"{total} registro(s) com erro operacional em {len(pending)} arquivo(s)", flush=True)
    for path, idx in pending.items():
        print(f"   {len(idx):2d}  {path.name}", flush=True)
    if args.dry_run or not total:
        return 0

    check = llm.checkApi(env)
    if not check["ok"]:
        print(f"ABORTADO: a API não respondeu ({check['error']}).", flush=True)
        return 1
    backup(runsDir)

    client = llm.getClient(env)
    componentClient = pipeline.componentClientFor(env)
    items = {}
    remaining = 0
    for path, idx in pending.items():
        run = runOf(path)
        if run.benchmark not in items:
            items[run.benchmark] = {it["id"]: it for it in datasets.loadBenchmark(
                run.benchmark, seed=env.seed, sampleSize=pipeline.SAMPLE_SIZES[run.benchmark])}
        records = pipeline.loadRecords(path)
        for i in idx:
            old = records[i]
            for attempt in range(1, ATTEMPTS + 1):
                new = pipeline.runItem(client, run, items[run.benchmark][old["itemId"]],
                                       componentClient)
                if not isOperational(new):
                    break
            records[i] = new
            remaining += isOperational(new)
            print(f"   {path.stem} {old['itemId']}: "
                  f"{old['filterFinish'] or '-'}/{old['componentFinish'] or '-'} -> "
                  f"{new['filterFinish'] or '-'}/{new['componentFinish'] or '-'} "
                  f"({attempt} tentativa(s))", flush=True)
        writeAtomic(path, records)
        left = pipeline.fillTiming(env, path)
        if left:
            print(f"   {path.name}: {left} tempo(s) do provedor ainda faltando", flush=True)
    print(f"\npronto: {total} registro(s) refeito(s); {remaining} ainda com erro operacional",
          flush=True)
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
