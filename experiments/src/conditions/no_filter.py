"""Referência sem defesa: o conteúdo vai cru ao componente principal, sem filtro."""

from __future__ import annotations

from . import COMPONENT_POLICY, COMPONENT_ROLE, buildSystem, buildUser

NAME = "no_filter"
TITLE = "Sem filtro"
DESCRIPTION = "Conteúdo externo direto ao componente principal, sem defesa alguma."

USES_FILTER = False
DELIMITS = False
MARKS = False


def filterContent(item: dict) -> str:
    """Vazio: não há estágio de filtro."""
    return ""


def filterPrompt(item: dict) -> tuple[str, str]:
    """Não existe nesta condição; levanta para que uma regressão apareça."""
    raise RuntimeError(
        "no_filter não possui estágio de filtro; o pipeline deve verificar "
        "USES_FILTER antes de chamar esta função."
    )


def prepareContent(item: dict, spans: list[str]) -> tuple[str, list[str]]:
    """Entrega o conteúdo externo cru."""
    return item.get("untrustedText", ""), []


def componentPrompt(item: dict, content: str) -> tuple[str, str]:
    """Par (system, user) do componente principal, igual a `simple`."""
    return buildSystem(COMPONENT_ROLE, COMPONENT_POLICY), buildUser(item, content)
