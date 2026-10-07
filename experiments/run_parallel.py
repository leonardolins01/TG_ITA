"""Inicia uma instância de `run_all.py` por repetição, cada uma num processo próprio.

`startInstance` verifica o estado da repetição antes de iniciar: completa, não faz
nada; já com instância viva (PID em `results/battery_r<k>.pid`), não faz nada;
senão inicia `run_all.py --reps <k>`, que retoma do ponto em que parou.

Uso:
    python run_parallel.py --dry-run      # só o estado de cada repetição
    python run_parallel.py                # inicia o que falta, uma instância por repetição
    python run_parallel.py --workers 2    # execuções simultâneas dentro de cada instância
"""

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src import pipeline  # noqa: E402
import run_all  # noqa: E402


def isAlive(pid: int) -> bool:
    """Diz se `pid` é um processo Python vivo (um PID antigo pode ter sido reusado)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return False  # 259 = STILL_ACTIVE
        name = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(1024)
        kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size))
        return "python" in name.value.lower()
    finally:
        kernel.CloseHandle(handle)


def runningPid(env, rep: int) -> int | None:
    """PID da instância viva da repetição, se houver."""
    pidFile = env.path("results").parent / f"battery_{run_all.runTag([rep])}.pid"
    try:
        pid = int(pidFile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if isAlive(pid) else None


def status(env, rep: int) -> tuple[int, int]:
    """(execuções completas, total) da repetição."""
    todo, done = run_all.plan(env, run_all.BENCHMARKS, [rep])
    return len(done), len(done) + len(todo)


def startInstance(env, rep: int, workers: int, dryRun: bool = False) -> str:
    """Inicia a instância da repetição se ela não estiver completa nem já rodando."""
    pid = runningPid(env, rep)
    if pid:
        return f"já em execução (PID {pid}); nada a fazer"
    done, total = status(env, rep)
    if done == total:
        return f"completa ({done}/{total}); nada a fazer"
    if dryRun:
        return f"{done}/{total} completas; seria iniciada e retomaria de onde parou"
    resultsDir = env.path("results").parent
    tag = run_all.runTag([rep])
    flags = 0
    if os.name == "nt":  # sem console e fora do grupo do lançador, para sobreviver a ele
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)
    with (resultsDir / f"battery_{tag}.stdout.log").open("a", encoding="utf-8") as out, \
         (resultsDir / f"battery_{tag}.stderr.log").open("a", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(HERE / "run_all.py"), "--reps", str(rep),
             "--workers", str(workers)],
            cwd=HERE, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
            creationflags=flags, close_fds=True)
    return f"{done}/{total} completas; iniciada (PID {proc.pid}), retoma de onde parou"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra o estado de cada repetição; não inicia nada")
    parser.add_argument("--workers", type=int, default=4,
                        help="execuções simultâneas dentro de cada instância")
    args = parser.parse_args()

    env = pipeline.Environment.load()
    for rep in range(pipeline.REPETITIONS):
        print(f"r{rep}: {startInstance(env, rep, args.workers, args.dry_run)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
