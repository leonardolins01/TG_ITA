"""Teste 2: delimitação. O filtro recebe o conteúdo em `<data>...</data>`, escapado;
o componente principal recebe o texto íntegro, como em `simple`."""

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

NAME = "delimiting"
TITLE = "Delimitação"
DESCRIPTION = "Canal de dado do filtro em <data>...</data>; instrução fora da tag. Sem marcação."

USES_FILTER = True
DELIMITS = True
MARKS = False


def filterContent(item: dict) -> str:
    """Conteúdo como o filtro o vê: dentro de `<data>` e escapado."""
    return marking.delimit(item.get("untrustedText", "")).content


def filterPrompt(item: dict) -> tuple[str, str]:
    """Par (system, user) do filtro, com a explicação da tag no system."""
    delimited = marking.delimit(item.get("untrustedText", ""))
    system = buildSystem(FILTER_ROLE, delimited.instruction, OUTPUT_CONTRACT)
    return system, buildUser(item, delimited.content)


def prepareContent(item: dict, spans: list[str]) -> tuple[str, list[str]]:
    """Repassa o texto íntegro: a delimitação vale só na entrada do filtro."""
    return item.get("untrustedText", ""), []


def componentPrompt(item: dict, content: str) -> tuple[str, str]:
    """Par (system, user) do componente principal, igual a `simple`."""
    return buildSystem(COMPONENT_ROLE, COMPONENT_POLICY), buildUser(item, content)
