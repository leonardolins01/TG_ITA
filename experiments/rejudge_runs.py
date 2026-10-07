"""Rejulga todos os registros com o juiz atual, offline e sem custo. Copia
`results/runs/` antes.

Uso:
    python rejudge_runs.py --dry-run   # quantos arquivos seriam rejulgados
    python rejudge_runs.py             # cópia e rejulgamento
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import datasets, metrics, pipeline  # noqa: E402


def parseFileId(stem: str) -> dict:
    """Separa o nome do JSONL em filtro, componente, condição, benchmark, seed e repetição (inverso de `Run.fileId`)."""
    parts = stem.split("__")
    models, condition, benchmark, seed = parts[:4]
    filterModel, _, component = models.partition("+")
    return {"filter": filterModel, "component": component or filterModel,
            "condition": condition, "benchmark": benchmark,
            "seed": int(seed.lstrip("s")),
            "rep": int(parts[4].lstrip("r")) if len(parts) > 4 else 0}


def writeAtomic(path: Path, records: list[dict]) -> None:
    """Regrava o JSONL por arquivo temporário + `os.replace`, preservando a ordem."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def backup(runsDir: Path) -> Path:
    """Copia `results/runs/` para `results/runs_backup_<data>/`, se a cópia do dia não existir."""
    target = runsDir.parent / f"runs_backup_{dt.date.today():%Y%m%d}"
    if not target.exists():
        shutil.copytree(runsDir, target)
        print(f"cópia de segurança: {target}", flush=True)
    else:
        print(f"cópia de segurança já existe: {target}", flush=True)
    return target


def rejudge(rec: dict, item: dict) -> tuple[bool, bool]:
    """Recalcula `attacked` e `accomplished` com o juiz atual; devolve o que mudou."""
    attacked = metrics.attackSucceeded(rec["response"], item.get("attackMarker", ""))
    accomplished = (not rec["blocked"]) and metrics.taskAccomplished(
        rec["response"], item.get("referenceAnswer", ""))
    changed = (attacked != rec["attacked"], accomplished != rec["accomplished"])
    rec["attacked"], rec["accomplished"] = attacked, accomplished
    rec["judgeVersion"] = metrics.JUDGE_VERSION
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra quantos arquivos seriam rejulgados; não grava nada")
    args = parser.parse_args()

    env = pipeline.Environment.load()
    runsDir = env.path("results")
    files = sorted(runsDir.glob("*.jsonl"))
    print(f"{len(files)} arquivos a rejulgar com o juiz v{metrics.JUDGE_VERSION}", flush=True)
    if args.dry_run:
        return 0

    itemsBy: dict[str, dict] = {}
    for bench, n in pipeline.SAMPLE_SIZES.items():
        itemsBy[bench] = {it["id"]: it for it in
                          datasets.loadBenchmark(bench, seed=env.seed, sampleSize=n)}
    backup(runsDir)
    for path in files:
        meta = parseFileId(path.stem)
        items = itemsBy[meta["benchmark"]]
        recs = pipeline.loadRecords(path)
        changedAtk = changedAcc = 0
        for rec in recs:
            a, b = rejudge(rec, items[rec["itemId"]])
            changedAtk += a
            changedAcc += b
        writeAtomic(path, recs)
        print(f"   {path.stem:62s} attacked mudou {changedAtk:3d} | "
              f"accomplished mudou {changedAcc:3d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
