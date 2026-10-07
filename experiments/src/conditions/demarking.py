"""Teste 3: marcação seletiva. O filtro vê o texto cru; ao liberar, marca com `^`
os trechos que apontou e o componente principal recebe o aviso de cautela."""

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

NAME = "demarking"
TITLE = "Marcação seletiva"
DESCRIPTION = "Filtro marca com ^ os trechos suspeitos e avisa o componente principal do perigo."

USES_FILTER = True
DELIMITS = False
MARKS = True


def filterContent(item: dict) -> str:
    """Conteúdo como o filtro o vê: o texto cru, igual a `simple`."""
    return item.get("untrustedText", "")


def filterPrompt(item: dict) -> tuple[str, str]:
    """Par (system, user) do filtro, igual a `simple`."""
    return buildSystem(FILTER_ROLE, OUTPUT_CONTRACT), buildUser(item, filterContent(item))


def prepareContent(item: dict, spans: list[str]) -> tuple[str, list[str]]:
    """Marca os trechos apontados pelo filtro; devolve (texto, trechos marcados)."""
    marked = marking.markSelectively(item.get("untrustedText", ""), spans)
    return marked.content, marked.spans


def componentPrompt(item: dict, content: str) -> tuple[str, str]:
    """Par (system, user) do componente principal, com o aviso sobre a marca."""
    system = buildSystem(COMPONENT_ROLE, COMPONENT_POLICY, marking.cautionInstruction())
    return system, buildUser(item, content)
