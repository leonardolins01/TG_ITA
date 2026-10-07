"""Único ponto de contato com a API (OpenRouter, OpenAI-compatível) e registry dos modelos."""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


# --------------------------------------------------------------------------- #
# Registry de modelos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    """Um modelo: id no OpenRouter, papel (system | guard | retired) e preço por milhão."""

    key: str
    apiId: str
    role: str
    inputPrice: float
    outputPrice: float
    description: str = ""


MODELS: dict[str, Model] = {
    "llama-3.1-8b": Model(
        "llama-3.1-8b", "meta-llama/llama-3.1-8b-instruct", "system", 0.05, 0.08,
        "Meta, jul/2024, 8B densos. Backbone sobre o qual Meta SecAlign e "
        "DataFilter publicam resultados.",
    ),
    "gpt-oss-20b": Model(
        "gpt-oss-20b", "openai/gpt-oss-20b", "system", 0.03, 0.13,
        "OpenAI, ago/2025, MoE 21B com 3,6B ativos, Apache 2.0.",
    ),
    "gpt-oss-safeguard": Model(
        "gpt-oss-safeguard", "openai/gpt-oss-safeguard-20b", "guard", 0.075, 0.30,
        "Fine-tunado para classificação de segurança sobre o MESMO backbone do "
        "gpt-oss-20b. Roda as quatro condições como filtro, com o gpt-oss-20b "
        "como componente principal: é o que permite medir o que o fine-tuning "
        "acrescenta a cada configuração, em vez de citá-lo.",
    ),
    "llama-guard-4": Model(
        "llama-guard-4", "meta-llama/llama-guard-4-12b", "retired", 0.18, 0.18,
        "Fora da bateria desde set/2026: classificador de conteúdo nocivo "
        "(taxonomia S1–S14, sem categoria de injeção) e podado do Llama 4 "
        "Scout, não do Llama 3.1. Os três JSONL ficam preservados em "
        "results/runs/ e não entram na análise.",
    ),
}

# Filtro fine-tunado -> modelo base que faz o papel de componente principal.
FILTER_SYSTEMS: dict[str, str] = {
    "gpt-oss-safeguard": "gpt-oss-20b",
}


def getModel(key: str) -> Model:
    """Devolve o registro do modelo; chave errada falha antes de gastar."""
    if key not in MODELS:
        raise KeyError(f"modelo '{key}' desconhecido; esperado um de {sorted(MODELS)}")
    return MODELS[key]


def systemModels() -> list[str]:
    """Modelos que instanciam o sistema inteiro (filtro e componente principal)."""
    return [m.key for m in MODELS.values() if m.role == "system"]


def guardModels() -> list[str]:
    """Filtros fine-tunados que rodam as quatro condições."""
    return [m.key for m in MODELS.values() if m.role == "guard"]


# --------------------------------------------------------------------------- #
# Chamada
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    """Texto, consumo, latência de ponta a ponta (s), `finishReason` e a identificação da geração."""

    text: str
    inputTokens: int
    outputTokens: int
    latency: float
    model: str
    finishReason: str = ""
    genId: str = ""      # consulta os tempos medidos pelo provedor (`generationStats`)
    provider: str = ""   # provedor que o OpenRouter escolheu para esta chamada


class OpenRouter:
    """Cliente HTTP mínimo do endpoint de chat, com retentativa em 429 e 5xx."""

    def __init__(self, baseUrl: str = DEFAULT_BASE_URL, apiKey: str | None = None,
                 maxTokens: int = 512, temperature: float = 0.0,
                 retries: int = 15, timeout: int = 180):
        self.baseUrl = baseUrl.rstrip("/")
        self.apiKey = apiKey or os.environ.get(API_KEY_ENV_VAR, "")
        if not self.apiKey:
            raise RuntimeError(
                f"chave de API ausente. Preencha `apiKey` no experiments/env.yaml "
                f"(gitignorado) ou exporte {API_KEY_ENV_VAR}:\n"
                f'  Windows PowerShell:  $env:{API_KEY_ENV_VAR} = "sk-or-..."\n'
                f'  bash:                export {API_KEY_ENV_VAR}="sk-or-..."\n'
                f"Obtenha em https://openrouter.ai/keys."
            )
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()

    def _session(self):
        """Sessão HTTP da thread: reaproveita a conexão TLS em vez de abrir uma por chamada."""
        import requests

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.apiKey}"})
            self._local.session = session
        return session

    def generate(self, model: Model, system: str, user: str) -> Completion:
        """Faz uma chamada system+user; espera 1, 2, 4... até 60 s entre tentativas."""
        payload = {
            "model": model.apiId,
            "temperature": self.temperature,
            "max_tokens": self.maxTokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.apiKey}",
                   "Content-Type": "application/json"}

        t0 = time.perf_counter()
        lastError = None
        for attempt in range(self.retries):
            try:
                r = self._session().post(f"{self.baseUrl}/chat/completions",
                                  json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise _Transient(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {}) or {}
                choice = data["choices"][0]
                if choice.get("finish_reason") == "error":
                    # o provedor falhou no meio da geração: erro operacional, não resposta
                    raise _Transient(f"finish_reason error ({data.get('provider')})")
                return Completion(
                    text=choice["message"].get("content") or "",
                    inputTokens=int(usage.get("prompt_tokens", 0)),
                    outputTokens=int(usage.get("completion_tokens", 0)),
                    latency=time.perf_counter() - t0,
                    model=model.key,
                    finishReason=str(choice.get("finish_reason") or ""),
                    genId=str(data.get("id") or r.headers.get("X-Generation-Id") or ""),
                    provider=str(data.get("provider") or ""),
                )
            except Exception as e:
                lastError = e
                if not isinstance(e, _Transient):
                    self._local.session = None  # conexão possivelmente quebrada: abre outra
                if attempt == self.retries - 1:
                    break
                time.sleep(min(2 ** attempt, 60) + random.random())
        raise RuntimeError(
            f"falha ao chamar {model.apiId} após {self.retries} tentativas: {lastError}"
        )


    def generationStats(self, genId: str, attempts: int = 8) -> dict | None:
        """Tempos medidos pelo provedor (`latency` e `generation_time`, em ms) de uma geração."""
        headers = {"Authorization": f"Bearer {self.apiKey}"}
        for attempt in range(attempts):
            try:
                r = self._session().get(f"{self.baseUrl}/generation", params={"id": genId},
                                 headers=headers, timeout=30)
                if r.status_code == 200:
                    data = r.json().get("data") or {}
                    return {"latencyMs": data.get("latency"),
                            "generationMs": data.get("generation_time"),
                            "provider": data.get("provider_name") or ""}
            except Exception:
                self._local.session = None  # conexão possivelmente quebrada: abre outra
            time.sleep(min(2 ** attempt, 30))  # a estatística aparece segundos depois da geração
        return None


class _Transient(Exception):
    """Erro que vale a pena repetir (limite de taxa, indisponibilidade)."""


# --------------------------------------------------------------------------- #
# Fábrica, verificação da API e estimativa de custo
# --------------------------------------------------------------------------- #
_CLIENT_CACHE: dict[tuple, object] = {}


def resolveApiKey(env) -> tuple[str, str]:
    """Devolve (chave, origem); a variável de ambiente vence o `apiKey` do env.yaml."""
    fromEnvVar = os.environ.get(API_KEY_ENV_VAR, "")
    if fromEnvVar:
        return fromEnvVar, f"variável de ambiente {API_KEY_ENV_VAR}"
    fromFile = getattr(env, "apiKey", "") or ""
    return fromFile, "campo apiKey do env.yaml" if fromFile else "nenhuma"


def getClient(env) -> OpenRouter:
    """Devolve o cliente, cacheado por `(baseUrl, maxTokens)`."""
    apiKey = resolveApiKey(env)[0] or None
    cacheKey = (env.baseUrl, env.maxTokens)
    if cacheKey not in _CLIENT_CACHE:
        _CLIENT_CACHE[cacheKey] = OpenRouter(
            baseUrl=env.baseUrl, maxTokens=env.maxTokens,
            temperature=env.temperature, apiKey=apiKey,
        )
    return _CLIENT_CACHE[cacheKey]


def generate(client: OpenRouter, modelKey: str, system: str, user: str) -> Completion:
    """Gera uma resposta com o modelo indicado pela chave curta."""
    return client.generate(getModel(modelKey), system, user)


def checkApi(env, modelKey: str = "llama-3.1-8b") -> dict:
    """Valida a chave (`GET /key`) e faz uma geração de 5 tokens; nunca levanta."""
    out = {"ok": False, "keySource": None, "label": None,
           "usage": None, "limit": None, "model": modelKey, "latency": None,
           "sample": None, "error": None}

    apiKey, source = resolveApiKey(env)
    out["keySource"] = source
    if not apiKey:
        out["error"] = (
            f"nenhuma chave encontrada. Preencha `apiKey` no experiments/env.yaml "
            f"ou exporte {API_KEY_ENV_VAR}."
        )
        return out

    try:
        import requests
        r = requests.get(f"{env.baseUrl.rstrip('/')}/key",
                         headers={"Authorization": f"Bearer {apiKey}"}, timeout=30)
        if r.status_code in (401, 403):
            out["error"] = f"chave rejeitada pelo OpenRouter (HTTP {r.status_code})."
            return out
        r.raise_for_status()
        info = (r.json() or {}).get("data", {}) or {}
        out["label"] = info.get("label")
        out["usage"] = info.get("usage")
        out["limit"] = info.get("limit")
    except Exception as e:
        out["error"] = f"não foi possível validar a chave: {e}"
        return out

    try:
        probe = OpenRouter(baseUrl=env.baseUrl, apiKey=apiKey, maxTokens=5,
                           temperature=0.0, retries=2, timeout=60)
        c = probe.generate(getModel(modelKey),
                           "Responda com uma única palavra.", "Diga OK.")
        out.update(ok=True, latency=round(c.latency, 2),
                   sample=(c.text or "").strip()[:40])
    except Exception as e:
        out["error"] = f"a chave é válida, mas a geração falhou: {e}"
    return out


def estimateCost(nItems: int, modelKey: str, inputTokens: int = 1200,
                 outputTokens: int = 300, stages: int = 2) -> float:
    """Custo estimado em US$ de `nItems`; os tokens padrão são dos dois estágios."""
    m = getModel(modelKey)
    inCost = nItems * inputTokens / 1e6 * m.inputPrice
    outCost = nItems * outputTokens / 1e6 * m.outputPrice
    return (inCost + outCost) * (stages / 2)
