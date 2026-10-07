"""Banco de provas: separação de canal em guardrails de entrada (ver `.claude/CLAUDE.md`)."""

from pathlib import Path

__version__ = "0.4.0"


def repoRoot() -> Path:
    """Raiz do repositório, dois níveis acima de `experiments/src/`."""
    return Path(__file__).resolve().parents[2]
