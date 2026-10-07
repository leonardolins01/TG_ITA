"""Teste 1: *prompt* simples, sem delimitação e sem marcação (baseline do 2×2)."""

from __future__ import annotations

from . import (
    COMPONENT_POLICY,
    COMPONENT_ROLE,
    FILTER_ROLE,
    OUTPUT_CONTRACT,
    buildSystem,
    buildUser,
)

NAME = "simple"
TITLE = "Prompt simples"
DESCRIPTION = "Filtro apenas com o objetivo de identificar ataques; sem delimitação, sem marcação."

USES_FILTER = True
DELIMITS = False
MARKS = False


def filterContent(item: dict) -> str:
    """Conteúdo como o filtro o vê: o texto cru."""
    return item.get("untrustedText", "")


def filterPrompt(item: dict) -> tuple[str, str]:
    """Par (system, user) do filtro."""
    return buildSystem(FILTER_ROLE, OUTPUT_CONTRACT), buildUser(item, filterContent(item))


def prepareContent(item: dict, spans: list[str]) -> tuple[str, list[str]]:
    """Repassa o texto íntegro ao componente principal."""
    return item.get("untrustedText", ""), []


def componentPrompt(item: dict, content: str) -> tuple[str, str]:
    """Par (system, user) do componente principal."""
    return buildSystem(COMPONENT_ROLE, COMPONENT_POLICY), buildUser(item, content)
