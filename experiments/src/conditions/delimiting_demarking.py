"""Teste 4: delimitação na entrada do filtro e marcação seletiva na saída."""

from __future__ import annotations

from .. import marking
from . import (
    COMPONENT_POLICY,
    COMPONENT_ROLE,
    FILTER_ROLE,
    OUTPUT_CONTRACT,
    buildSystem,
    buildUser,
)

NAME = "delimiting_demarking"
TITLE = "Delimitação + marcação"
DESCRIPTION = "Canal de dado em <data> na entrada do filtro E marcação seletiva na saída."

USES_FILTER = True
DELIMITS = True
MARKS = True


def filterContent(item: dict) -> str:
    """Conteúdo como o filtro o vê: dentro de `<data>` e escapado."""
    return marking.delimit(item.get("untrustedText", "")).content


def filterPrompt(item: dict) -> tuple[str, str]:
    """Par (system, user) do filtro, com a explicação da tag no system."""
    delimited = marking.delimit(item.get("untrustedText", ""))
    system = buildSystem(FILTER_ROLE, delimited.instruction, OUTPUT_CONTRACT)
    return system, buildUser(item, delimited.content)


def prepareContent(item: dict, spans: list[str]) -> tuple[str, list[str]]:
    """Marca no texto original os trechos que o filtro leu escapados."""
    marked = marking.markSelectively(item.get("untrustedText", ""), spans)
    return marked.content, marked.spans


def componentPrompt(item: dict, content: str) -> tuple[str, str]:
    """Par (system, user) do componente principal, com o aviso sobre a marca."""
    system = buildSystem(COMPONENT_ROLE, COMPONENT_POLICY, marking.cautionInstruction())
    return system, buildUser(item, content)
