"""Roda as repetições da bateria nos três benchmarks, com retomada, retentativa e
máquina acordada; se as 420 execuções ficarem íntegras, gera análise, tabelas e figuras.

Uso:
    python run_all.py                    # roda tudo o que falta (as dez repetições)
    python run_all.py --dry-run          # só mostra o plano e o custo
    python run_all.py --reps 2-4         # só essas repetições (de 0 a 9)
    python run_all.py --workers 1        # uma execução por vez (padrão: 4 simultâneas)
    python run_all.py --benchmark sep    # um benchmark só
    python run_all.py --smoke            # um item por execução pendente, e para

Interrompido, basta relançar o mesmo comando. Log em `results/battery_all.log`
(anexado) e PID em `results/battery_all.pid`; com uma repetição só (`--reps k`),
`battery_r<k>.log` e `battery_r<k>.pid`, para que instâncias paralelas
(`run_parallel.py`) não se misturem.
"""

import argparse
import ctypes
import dataclasses
import datetime as dt
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import analysis, datasets, llm, metrics, pipeline  # noqa: E402
import check_runs  # noqa: E402

BENCHMARKS = tuple(pipeline.SAMPLE_SIZES)
ALL_REPS = range(pipeline.REPETITIONS)
RETRY_WAIT = 300          # segundos entre tentativas de uma execução que falhou
RETRIES_PER_RUN = 12      # ~1 h de indisponibilidade antes de adiar a execução


class Log:
    """Escreve no console e no arquivo de log, com carimbo de tempo; seguro entre threads."""

    def __init__(self, path: Path):
        self.fh = path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, text: str = "") -> None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp}  {text}" if text else ""
        with self.lock:
            print(line, flush=True)
            self.fh.write(line + "\n")
            self.fh.flush()


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keepAwake(on: bool) -> None:
    """Pede (ou libera) que o Windows não suspenda a máquina; fora dele, nada."""
    if os.name != "nt":
        return
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def envFor(env, benchmark: str):
    """O ambiente com o benchmark e o tamanho de amostra dele."""
    return dataclasses.replace(env, benchmark=benchmark,
                               sampleSize=pipeline.SAMPLE_SIZES[benchmark])


def parseReps(text: str) -> list[int]:
    """`"3"` -> [3]; `"2-4"` -> [2, 3, 4]."""
    first, _, last = text.partition("-")
    reps = list(range(int(first), int(last or first) + 1))
    if not reps or reps[0] < 0 or reps[-1] >= pipeline.REPETITIONS:
        raise argparse.ArgumentTypeError(
            f"repetições vão de 0 a {pipeline.REPETITIONS - 1}: {text!r}")
    return reps


def plan(env, benchmarks, reps):
    """Separa as execuções em (pendentes, completas), repetição a repetição."""
    todo, done = [], []
    for rep in reps:
        for benchmark in benchmarks:
            envB = envFor(env, benchmark)
            for run in pipeline.battery(envB, rep):
                path = pipeline.runFile(envB, run)
                complete = (len(pipeline.doneIds(path)) >= envB.sampleSize
                            and pipeline.missingTiming(path) == 0)
                (done if complete else todo).append(run)
    return todo, done


def runWithRetries(env, run: pipeline.Run, items, log: Log, smoke: bool) -> bool:
    """Executa uma execução, insistindo até `RETRIES_PER_RUN` vezes; `smoke` roda um item."""
    name = run.fileId()
    for attempt in range(1, RETRIES_PER_RUN + 1):
        try:
            if smoke:
                path = pipeline.runFile(env, run)
                pending = [i for i in items if i["id"] not in pipeline.doneIds(path)]
                if not pending:
                    log(f"      {name}: nada pendente")
                    return True
                client = llm.getClient(env)
                rec = pipeline.runItem(client, run, pending[0],
                                       pipeline.componentClientFor(env))
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log(f"      {name} item {rec['itemId']}: filterPred={rec['filterPred']} "
                    f"blocked={rec['blocked']} attacked={rec['attacked']} "
                    f"accomplished={rec['accomplished']}")
                log(f"      filtro devolveu: {rec['filterRawOutput'][:160]!r}")
                log(f"      componente respondeu: {rec['response'][:160]!r}")
                return True
            pipeline.runOne(env, run, items=items, resume=True, progress=False)
            return True
        except Exception as err:
            log(f"      {name}: falha na tentativa {attempt}/{RETRIES_PER_RUN}: {err}")
            if attempt == RETRIES_PER_RUN:
                return False
            log(f"      {name}: esperando {RETRY_WAIT}s antes de tentar de novo")
            time.sleep(RETRY_WAIT)
    return False


def summarizeRun(env, run: pipeline.Run, log: Log) -> None:
    """Imprime a linha-resumo de uma execução concluída."""
    row = metrics.summarize(pipeline.records(env, run), run).asRow()
    log(f"      concluída {run.fileId()} | n={row['n']} asr={row['asr']:.3f} "
        f"utility={row['utility']:.3f} overBlock={row['overBlock']:.3f} "
        f"invalid={row['invalidOutput']:.3f} marked={row['markedRate']:.3f}")


def runTag(reps) -> str:
    """Sufixo do log e do PID: `r<k>` para uma repetição só, `all` para as demais."""
    reps = list(reps)
    return f"r{reps[0]}" if len(reps) == 1 else "all"


def postProcess(env, log: Log) -> None:
    """Audita o disco e, se íntegro, gera CSV, tabelas e figuras (uma instância por vez)."""
    lock = env.path("results").parent / "postprocess.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("outra instância está gerando a análise; esta não gera de novo")
        return
    os.close(fd)
    try:
        _postProcess(env, log)
    finally:
        lock.unlink(missing_ok=True)


def _postProcess(env, log: Log) -> None:
    log("auditando os JSONL (check_runs.auditAll)")
    problems = check_runs.auditAll(list(BENCHMARKS), env)
    if problems:
        log(f"{problems} problema(s) na auditoria; análise NÃO gerada. "
            f"Rode `python check_runs.py` e corrija antes de `buildAnalysis.py`.")
        return
    log("gerando results/analysis/*.csv")
    analysis.writeAll(env)
    log("gerando latex/tables/*.tex")
    analysis.writeLatexTables(env)
    log("gerando as figuras (makeFigures.main)")
    import makeFigures
    makeFigures.main()
    log("análise, tabelas e figuras geradas")


def runQueue(env, queue, itemsCache, log: Log, workers: int, smoke: bool) -> list:
    """Roda a fila com até `workers` execuções simultâneas; devolve as adiadas."""
    total = len(queue)

    def execute(n, run):
        log(f"[{n:3d}/{total}] começou  {run.fileId()}")
        envB = envFor(env, run.benchmark)
        t0 = time.time()
        if not runWithRetries(envB, run, itemsCache[run.benchmark], log, smoke):
            log(f"      ADIADA depois de {RETRIES_PER_RUN} tentativas: {run.fileId()}")
            return run
        if not smoke:
            log(f"      {run.fileId()}: {time.time() - t0:7.1f}s")
            summarizeRun(envB, run, log)
        return None

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(execute, n, run) for n, run in enumerate(queue, 1)]
        pending = set(futures)
        while pending:  # espera com prazo para que o Ctrl+C chegue no Windows
            _, pending = wait(pending, timeout=5)
        return [r for r in (f.result() for f in futures) if r is not None]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def runRepetitions(reps=range(pipeline.REPETITIONS), workers: int = 4,
                   dryRun: bool = False, smoke: bool = False,
                   benchmarks=BENCHMARKS) -> int:
    """Roda o que falta das repetições pedidas; relançar continua de onde parou."""
    env = pipeline.Environment.load()
    resultsDir = env.path("results").parent
    reps = list(reps)
    tag = runTag(reps)
    log = Log(resultsDir / f"battery_{tag}.log")

    log("=" * 72)
    log(f"run_all.py  reps={reps[0]}-{reps[-1]} benchmarks={list(benchmarks)} "
        f"workers={workers} dry_run={dryRun} smoke={smoke}")

    todo, done = plan(env, benchmarks, reps)
    if len(reps) == 1 and todo:
        # instâncias paralelas começam em pontos diferentes da lista, para não
        # concentrar todas as chamadas no mesmo modelo e provedor
        offset = (4 * reps[0]) % len(todo)
        todo = todo[offset:] + todo[:offset]
    log(f"{len(done)} execução(ões) já completa(s), {len(todo)} pendente(s)")
    for run in todo:
        log(f"   pendente  {run.fileId()}")

    cost = pipeline.estimateBatteryCost(env, todo)["TOTAL"]
    log(f"custo estimado: US$ {cost:.4f}")

    check = {"usage": None}
    if todo and not dryRun:
        check = llm.checkApi(env)
        if not check["ok"]:
            log(f"ABORTADO: a API não respondeu ({check['error']}).")
            return 1
        log(f"API ok ({check['keySource']}); uso acumulado US$ {check['usage']}")

    if dryRun or not todo:
        log("nada executado" + (" (dry-run)" if dryRun else ": tudo completo"))
        if not todo and not dryRun and tuple(benchmarks) == BENCHMARKS:
            if not plan(env, BENCHMARKS, ALL_REPS)[0]:
                postProcess(env, log)
        return 0

    itemsCache = {}
    for b in sorted({r.benchmark for r in todo}):
        itemsCache[b] = datasets.loadBenchmark(b, seed=env.seed,
                                               sampleSize=pipeline.SAMPLE_SIZES[b])
        log(f"{b}: {len(itemsCache[b])} itens carregados")

    pidFile = resultsDir / f"battery_{tag}.pid"
    pidFile.write_text(str(os.getpid()), encoding="utf-8")
    keepAwake(True)
    started = time.time()
    try:
        deferred = runQueue(env, todo, itemsCache, log, workers, smoke)
        if deferred:
            log(f"segunda passada: {len(deferred)} execução(ões) adiada(s)")
            runQueue(env, deferred, itemsCache, log, workers, smoke)
    except KeyboardInterrupt:
        log("interrompido; relance o mesmo comando para continuar de onde parou")
        keepAwake(False)
        pidFile.unlink(missing_ok=True)
        os._exit(130)  # as threads em andamento não param sozinhas
    finally:
        keepAwake(False)
        pidFile.unlink(missing_ok=True)

    log("-" * 72)
    log(f"fim em {(time.time() - started) / 60:.1f} min")
    if smoke:
        log("smoke: nada mais executado; confira as saídas acima e rode sem --smoke")
        return 0

    remaining, _ = plan(env, BENCHMARKS, ALL_REPS)
    if check.get("usage") is not None:
        after = llm.checkApi(env)
        if after["ok"] and after.get("usage") is not None:
            log(f"gasto nesta rodada: US$ {after['usage'] - check['usage']:.4f}")
    if remaining:
        log(f"{len(remaining)} execução(ões) ainda pendente(s); "
            f"repetições completas: {analysis.completeReps(env)} de {pipeline.REPETITIONS}")
        log("rode `python run_all.py` de novo para completar; a retomada continua "
            "de onde parou. Análise completa NÃO gerada.")
        return 1

    postProcess(env, log)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra o plano e o custo; não chama nada")
    parser.add_argument("--reps", type=parseReps,
                        default=list(range(pipeline.REPETITIONS)),
                        help=f"repetições a rodar, como 3 ou 2-4 "
                             f"(padrão: 0-{pipeline.REPETITIONS - 1})")
    parser.add_argument("--workers", type=int, default=4,
                        help="execuções simultâneas; 1 = estritamente sequencial")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS),
                        help="um benchmark só (padrão: os três)")
    parser.add_argument("--smoke", action="store_true",
                        help="um item por execução pendente, para conferir o caminho")
    args = parser.parse_args()
    return runRepetitions(reps=args.reps, workers=args.workers, dryRun=args.dry_run,
                          smoke=args.smoke,
                          benchmarks=(args.benchmark,) if args.benchmark else BENCHMARKS)


if __name__ == "__main__":
    raise SystemExit(main())
