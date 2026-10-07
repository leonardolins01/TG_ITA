"""Audita os JSONL de `results/runs/` (todas as repetições): JSON válido, sem `itemId`
repetido, mesmo conjunto e ordem de itens do loader, chaves presentes e execuções alinhadas.

Uso:
    python check_runs.py             # todos os benchmarks presentes no disco
    python check_runs.py bipia sep   # só os nomeados
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import analysis, datasets, pipeline

# Chaves que `metrics.summarize` consome.
REQUIRED = ("itemId", "label", "filterPred", "filterValid", "blocked",
            "markedCount", "response", "attacked", "accomplished")


def auditFile(path: Path, expectedIds: list[str] | None) -> list[str]:
    """Confere um JSONL e devolve a lista de problemas."""
    problems = []
    ids = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as err:
                # `pipeline.doneIds` pularia esta linha; `loadRecords` levanta
                problems.append(f"linha {n} não é JSON válido ({err})")
                continue
            ids.append(rec.get("itemId"))
            missing = [k for k in REQUIRED if k not in rec]
            if missing:
                problems.append(f"linha {n} sem as chaves {missing}")
            if rec.get("response") is None:
                problems.append(f"linha {n} com response nulo")

    seen, repeated = set(), []
    for i in ids:
        if i in seen:
            repeated.append(i)
        seen.add(i)
    if repeated:
        problems.append(f"{len(repeated)} itemId repetido(s): {repeated[:5]}")

    if expectedIds is not None:
        if len(ids) < len(expectedIds) and ids == expectedIds[:len(ids)]:
            return problems  # parcial e em ordem: a retomada completa
        if len(ids) != len(expectedIds):
            problems.append(f"INCOMPLETA: {len(ids)} de {len(expectedIds)} itens")
        faltando = set(expectedIds) - seen
        sobrando = seen - set(expectedIds)
        if faltando:
            problems.append(f"{len(faltando)} item(ns) do benchmark ausente(s)")
        if sobrando:
            problems.append(f"{len(sobrando)} itemId que não é do benchmark")
        if not faltando and not sobrando and ids != expectedIds:
            problems.append("ordem dos itens difere da ordem do loader")
    return problems


def auditAll(wanted: list[str] | None = None, env=None) -> int:
    """Audita todos os JSONL do disco (inclusive os fora da bateria); devolve o nº de problemas."""
    wanted = wanted or []
    env = env or pipeline.Environment.load()
    folder = env.path("results")

    byBenchmark = defaultdict(list)
    for path in sorted(folder.glob("*.jsonl")):
        parts = path.stem.split("__")      # modelo__condicao__benchmark__s<seed>[__r<rep>]
        if len(parts) in (4, 5):
            byBenchmark[parts[2]].append(path)

    if not byBenchmark:
        print(f"nenhum JSONL em {folder}")
        return 1

    total = 0
    for benchmark, paths in sorted(byBenchmark.items()):
        if wanted and benchmark not in wanted:
            continue
        print("=" * 78)
        print(f"{benchmark} — {len(paths)} arquivo(s) no disco "
              f"(a bateria tem {len(pipeline.battery(env))} por repetição)")
        print("=" * 78)

        # tamanho esperado vem dos próprios arquivos, não de `env.sampleSize`
        sizes = {sum(1 for line in path.open(encoding="utf-8") if line.strip())
                 for path in paths}
        sampleSize = max(sizes)
        if len(sizes) > 1:
            print(f"  execuções com tamanhos diferentes: {sorted(sizes)} "
                  f"— conferindo contra o maior ({sampleSize})")
        try:
            items = datasets.loadBenchmark(benchmark, seed=env.seed,
                                           sampleSize=sampleSize)
            expectedIds = [i["id"] for i in items]
            print(f"  benchmark recarregado: {len(expectedIds)} itens "
                  f"(seed {env.seed}, amostra {sampleSize})")
        except Exception as err:
            expectedIds = None
            print(f"  não deu para recarregar o benchmark: {err}")
            print("  (sem ele, não dá para conferir completude nem ordem)")

        orders = {}
        partial = 0
        for path in paths:
            problems = auditFile(path, expectedIds)
            total += len(problems)
            n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            isPartial = not problems and expectedIds is not None and n < len(expectedIds)
            partial += isPartial
            mark = "FALHA   " if problems else ("parcial " if isPartial else "ok      ")
            print(f"  {mark}{path.name:56s} {n:5d} itens")
            for p in problems:
                print(f"           - {p}")
            if not problems and not isPartial:
                orders[path.name] = [json.loads(line)["itemId"]
                                     for line in path.open(encoding="utf-8")
                                     if line.strip()]

        if partial:
            print(f"\n  {partial} execução(ões) parcial(is), em ordem; a retomada as completa.")
        seqs = list(orders.values())
        if len(seqs) > 1:
            if all(s == seqs[0] for s in seqs[1:]):
                print(f"\n  as {len(seqs)} execuções íntegras veem os mesmos itens, "
                      f"na mesma ordem — intervalos pareados válidos.")
            else:
                print("\n  ATENÇÃO: as execuções não estão alinhadas item a item; "
                      "os intervalos pareados de metrics não valem assim.")
                total += 1
        print()

    print("=" * 78)
    print("TUDO ÍNTEGRO" if not total else f"{total} problema(s) encontrados")
    print(f"repetições completas nos três benchmarks: {analysis.completeReps(env)} "
          f"de {pipeline.REPETITIONS}")
    return total


def main() -> int:
    return 0 if auditAll(sys.argv[1:]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
