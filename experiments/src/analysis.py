"""Consolida as execuções (42 por repetição) nos CSV de `results/analysis/` e nas
tabelas LaTeX. Lê os JSONL do disco; nada aqui chama a API."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from . import conditions, llm, metrics, pipeline

BENCHMARKS = ("bipia", "sep", "notinject")
BENCH_TITLE = {"bipia": "BIPIA", "sep": "SEP", "notinject": "NotInject"}

CONDITION_ORDER = ("no_filter", "simple", "delimiting", "demarking",
                   "delimiting_demarking")

# Linhas das figuras; a do safeguard usa o `no_filter` do gpt-oss-20b (`runsByKey`).
ROWS = ("llama-3.1-8b", "gpt-oss-20b", "gpt-oss-safeguard")

# Contra no_filter: proteção total. Contra simple: efeito da separação de canal.
BASELINES = ("no_filter", "simple")

DELTA_METRICS = {
    "asr": "attacked",
    "utility": "accomplished",
}

# Taxas absolutas que ganham IC em summary.csv: coluna -> desfecho por item.
ABSOLUTE_METRICS = {
    "asr": "attacked",
    "utility": "accomplished",
    "overBlock": "blocked",
}

# Contrastes do 2×2 como pesos sobre as quatro condições (efeito = média por item).
EFFECTS = {
    "delimiting": {"simple": -0.5, "demarking": -0.5,
                   "delimiting": 0.5, "delimiting_demarking": 0.5},
    "marking": {"simple": -0.5, "delimiting": -0.5,
                "demarking": 0.5, "delimiting_demarking": 0.5},
    "interaction": {"simple": 1.0, "delimiting": -1.0,
                    "demarking": -1.0, "delimiting_demarking": 1.0},
}


def runsFor(env: pipeline.Environment, benchmark: str) -> list[pipeline.Run]:
    """As quatorze execuções de um benchmark, independentemente de `env.benchmark`."""
    return pipeline.battery(dataclasses.replace(env, benchmark=benchmark))


def rowKey(run: pipeline.Run) -> str:
    """Linha da figura: o filtro, se fine-tunado; senão, o componente principal."""
    return run.model if run.model in llm.FILTER_SYSTEMS else run.component


def runsByKey(env: pipeline.Environment,
              benchmark: str) -> dict[tuple[str, str], pipeline.Run]:
    """Execuções por (linha, condição), com o alias `no_filter` da linha do guard."""
    byKey = {(rowKey(r), r.condition): r for r in runsFor(env, benchmark)}
    for guard, base in llm.FILTER_SYSTEMS.items():
        if (base, "no_filter") in byKey and (guard, "no_filter") not in byKey:
            byKey[(guard, "no_filter")] = byKey[(base, "no_filter")]
    return byKey


def completeReps(env: pipeline.Environment) -> int:
    """Quantas repetições 0..k−1 estão completas nas execuções dos três benchmarks."""
    runs = [r for b in BENCHMARKS for r in runsFor(env, b)]
    k = 0
    while k < pipeline.REPETITIONS and all(
            len(pipeline.doneIds(pipeline.runFile(env, dataclasses.replace(r, rep=k))))
            >= pipeline.SAMPLE_SIZES[r.benchmark] for r in runs):
        k += 1
    return k


def repRecords(env: pipeline.Environment, run: pipeline.Run, reps: int) -> list[dict]:
    """Registros das repetições 0..reps−1 de uma execução, concatenados."""
    return [rec for k in range(reps)
            for rec in pipeline.records(env, dataclasses.replace(run, rep=k))]


_MEANS: dict[tuple[str, int], dict[str, dict[str, float]]] = {}


def itemOutcomes(env: pipeline.Environment, run: pipeline.Run,
                 reps: int) -> dict[str, dict[str, float]]:
    """Média por item de cada desfecho sobre as repetições (em cache por arquivo)."""
    key = (str(pipeline.runFile(env, run)), reps)
    if key not in _MEANS:
        recs = repRecords(env, run, reps)
        _MEANS[key] = {o: metrics.itemMeans(recs, o) for o in metrics.OUTCOME_LABEL}
    return _MEANS[key]


def _reps(env: pipeline.Environment, reps: int | None) -> int:
    reps = reps or completeReps(env)
    if not reps:
        raise RuntimeError("nenhuma repetição completa em results/runs/; "
                           "rode `python check_runs.py`")
    return reps


def buildSummary(env: pipeline.Environment, reps: int | None = None) -> pd.DataFrame:
    """Uma linha por execução e benchmark, com IC das taxas, mais as de alias (`alias=True`)."""
    reps = _reps(env, reps)
    rows = []
    for benchmark in BENCHMARKS:
        for (row_, condition), run in runsByKey(env, benchmark).items():
            recs = repRecords(env, run, reps)
            row = metrics.summarize(recs, run).asRow()
            row["n"] = len({r["itemId"] for r in recs})
            row["reps"] = reps
            for column, outcome in ABSOLUTE_METRICS.items():
                means = list(itemOutcomes(env, run, reps)[outcome].values())
                if column == "overBlock" and run.condition == "no_filter":
                    low = high = float("nan")  # sem filtro não há bloqueio a estimar
                else:
                    _, low, high, _ = metrics.itemCi(means, seed=env.seed)
                row[f"{column}Low"], row[f"{column}High"] = low, high
            row["filterModel"] = row["model"]
            row["model"] = row_
            row["condition"] = condition
            row["alias"] = rowKey(run) != row_
            rows.append(row)
    return pd.DataFrame(rows)


def buildDeltas(env: pipeline.Environment, reps: int | None = None) -> pd.DataFrame:
    """Δ de ASR e utilidade contra `no_filter` e `simple`, com IC BCa sobre os itens."""
    reps = _reps(env, reps)
    rows = []
    for benchmark in BENCHMARKS:
        byKey = runsByKey(env, benchmark)
        for model in ROWS:
            for baseline in BASELINES:
                reference = itemOutcomes(env, byKey[(model, baseline)], reps)
                for condition in CONDITION_ORDER:
                    if condition == baseline or (model, condition) not in byKey:
                        continue
                    current = itemOutcomes(env, byKey[(model, condition)], reps)
                    for name, outcome in DELTA_METRICS.items():
                        ref = reference[outcome]
                        ids = [i for i in current[outcome] if i in ref]
                        a = [current[outcome][i] for i in ids]
                        b = [ref[i] for i in ids]
                        point, low, high, method = metrics.itemCi(a, b, seed=env.seed)
                        rows.append({
                            "benchmark": benchmark,
                            "model": model,
                            "condition": condition,
                            "metric": name,
                            "baseline": baseline,
                            "method": method,
                            "point": point,
                            "ciLow": low,
                            "ciHigh": high,
                            "significant": bool(round(low, 12) > 0 or round(high, 12) < 0),
                        })
    return pd.DataFrame(rows)


def buildEffects(env: pipeline.Environment, reps: int | None = None) -> pd.DataFrame:
    """Efeitos principais e interação do 2×2, com IC BCa; sem intervalo se quase não há variação."""
    reps = _reps(env, reps)
    rows = []
    for benchmark in BENCHMARKS:
        byKey = runsByKey(env, benchmark)
        for model in ROWS:
            means = {c: itemOutcomes(env, byKey[(model, c)], reps)
                     for c in conditions.FOUR_CONDITIONS}
            for name, outcome in DELTA_METRICS.items():
                m = {c: means[c][outcome] for c in conditions.FOUR_CONDITIONS}
                ids = [i for i in m["simple"]
                       if all(i in m[c] for c in conditions.FOUR_CONDITIONS)]
                if not ids:
                    continue
                for effect, weights in EFFECTS.items():
                    # arredondado para que resíduo de ponto flutuante não pareça variação
                    values = [round(sum(w * m[c][i] for c, w in weights.items()), 12)
                              for i in ids]
                    point, low, high, method = metrics.itemCi(values, seed=env.seed)
                    rows.append({
                        "benchmark": benchmark,
                        "model": model,
                        "metric": name,
                        "effect": effect,
                        "method": method,
                        "n": len(ids),
                        "point": point,
                        "ciLow": low,
                        "ciHigh": high,
                        "significant": bool(round(low, 12) > 0 or round(high, 12) < 0),
                    })
    return pd.DataFrame(rows)


def buildTruncation(env: pipeline.Environment, reps: int | None = None) -> pd.DataFrame:
    """Respostas vazias e gerações cortadas no teto, entre os itens não bloqueados, somando as repetições."""
    reps = _reps(env, reps)
    rows = []
    for benchmark in BENCHMARKS:
        for (row_, condition), run in runsByKey(env, benchmark).items():
            records = [r for r in repRecords(env, run, reps) if not r["blocked"]]
            if not records:
                continue
            empty = sum(1 for r in records if not r["response"].strip())
            cut = sum(1 for r in records if r.get("componentFinish") == "length")
            rows.append({
                "benchmark": benchmark,
                "model": row_,
                "condition": condition,
                "answered": len(records),
                "empty": empty,
                "emptyRate": empty / len(records),
                "cut": cut,
                "cutRate": cut / len(records),
            })
    return pd.DataFrame(rows)


# Linguagem simples e onde a medida não se aplica; complementa COLUMN_LEGENDS.
GLOSSARY: list[dict[str, str]] = [
    {"metric": "asr", "title": "Taxa de sucesso do ataque",
     "plain": "De cada 100 tentativas de ataque, quantas o componente principal "
              "de fato executou. É a métrica primária do trabalho.",
     "direction": "menor é melhor",
     "appliesTo": "bipia, sep",
     "notAppliesWhy": "O NotInject é só-benigno (339 de 339): não há ataque, "
                      "logo não há denominador."},
    {"metric": "deltaAsr", "title": "Diferença de ASR",
     "plain": "Quanto a condição baixou o ASR em relação a uma referência. "
              "Contra no_filter mostra o que o guardrail inteiro protege; "
              "contra simple isola o efeito da separação de canal, que é a "
              "pergunta de pesquisa.",
     "direction": "mais negativo é melhor",
     "appliesTo": "bipia, sep",
     "notAppliesWhy": "Depende do ASR, que não existe no NotInject."},
    {"metric": "effect", "title": "Efeitos do desenho 2x2",
     "plain": "O efeito principal da delimitação é a média das duas condições "
              "com delimitação menos a das duas sem; o da marcação, idem para a "
              "marcação. Usam as quatro condições de uma vez, e por isso o "
              "intervalo é mais estreito que o da comparação de uma condição com "
              "o prompt simples. A interação diz se os dois fatores se somam: "
              "zero significa que o ganho da marcação não depende da delimitação.",
     "direction": "no ASR, negativo é melhor; na utilidade, negativo é custo",
     "appliesTo": "ASR em bipia e sep; utilidade nos três",
     "notAppliesWhy": "O NotInject não tem ataque, logo não há ASR a decompor."},
    {"metric": "utility", "title": "Utilidade",
     "plain": "Fração das tarefas legítimas que o sistema cumpriu. Item "
              "bloqueado conta como fracasso. É o contrapeso do ASR: uma defesa "
              "que zera o ataque quebrando a tarefa não serve.",
     "direction": "maior é melhor",
     "appliesTo": "os três, com sentidos diferentes",
     "notAppliesWhy": "Só o BIPIA publica resposta de referência. Em sep e "
                      "notinject a medida cai para \"respondeu sem ter sido "
                      "bloqueado\", não \"respondeu certo\"."},
    {"metric": "overBlock", "title": "Sobre-bloqueio",
     "plain": "Fração do conteúdo legítimo que o filtro bloqueou por engano. "
              "É o preço que usuários inocentes pagam pela defesa.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": ""},
    {"metric": "markedRate", "title": "Alcance da marcação",
     "plain": "Dos itens que o filtro liberou, em quantos ele chegou a apontar "
              "algum trecho para marcar. É o limite honesto do método: sem "
              "trecho apontado, a marcação seletiva não tem o que marcar e a "
              "condição degrada para o prompt simples.",
     "direction": "não há direção boa; é diagnóstico",
     "appliesTo": "os três",
     "notAppliesWhy": "É zero por construção nas condições que não marcam."},
    {"metric": "invalidOutput", "title": "Saída inválida do filtro",
     "plain": "Fração das respostas do filtro que não puderam ser "
              "interpretadas. Contam conservadoramente como não bloqueio: na "
              "dúvida, o item passa e é medido no desfecho.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": "É 0,000 em todas as execuções do llama; só o gpt-oss varia."},
    {"metric": "emptyOutput", "title": "Resposta vazia do componente",
     "plain": "Dos itens que o filtro liberou, em quantos o componente principal "
              "devolveu resposta vazia, quase sempre por esgotar o teto de tokens "
              "raciocinando. Conta como saída inválida: não é bloqueio nem "
              "sobre-bloqueio, não executa o ataque e não cumpre a tarefa.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": "É praticamente zero no llama, que não raciocina antes de responder."},
    {"metric": "meanTokens", "title": "Custo em tokens",
     "plain": "Média de tokens por item, somando o filtro e o componente "
              "principal. Delimitar e marcar acrescentam texto, então este é o "
              "preço direto da defesa.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": ""},
    {"metric": "latP50", "title": "Tempo de geração",
     "plain": "Tempo que o provedor levou gerando as respostas de um item, "
              "somando filtro e componente: a mediana, e o percentil 95 como "
              "cauda. É medido pelo próprio provedor, então não inclui fila, rede "
              "nem esperas por limite de taxa. O guardrail acrescenta uma chamada "
              "antes da resposta.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": "O OpenRouter pode atender um mesmo modelo por provedores "
                      "diferentes, com hardware diferente."},
    {"metric": "emptyRate", "title": "Resposta vazia",
     "plain": "Fração das respostas que saíram em branco. No gpt-oss, que é "
              "modelo de raciocínio, isso é geração cortada no teto de tokens, "
              "não recusa — o modelo gastou o orçamento pensando.",
     "direction": "menor é melhor",
     "appliesTo": "os três",
     "notAppliesWhy": "É zero no llama, que não raciocina antes de responder."},
    {"metric": "tprAt1Fpr", "title": "Detecção a 1% de falsos positivos",
     "plain": "Quanto o filtro detecta se calibrado para errar em no máximo 1% "
              "do conteúdo legítimo. Mede a classificação do filtro, não o "
              "desfecho do sistema.",
     "direction": "maior é melhor",
     "appliesTo": "só gpt-oss e safeguard em bipia e sep",
     "notAppliesWhy": "Exige que o filtro emita risk_score; o llama não emite, "
                      "e o NotInject não tem ataque."},
    {"metric": "f1", "title": "F1 do filtro",
     "plain": "Qualidade da decisão binária do filtro, misturando precisão e "
              "revocação. Mede o acerto do filtro isolado, e não o que sai do "
              "sistema.",
     "direction": "maior é melhor",
     "appliesTo": "bipia, sep",
     "notAppliesWhy": "Precisa dos dois rótulos; o NotInject só tem benignos."},
]


def buildGlossary() -> pd.DataFrame:
    """O glossário em linguagem simples, como DataFrame."""
    return pd.DataFrame(GLOSSARY)


def writeAll(env: pipeline.Environment, outDir: Path | None = None,
             reps: int | None = None) -> dict[str, Path]:
    """Calcula e grava os cinco CSV (por padrão, sobre todas as repetições completas)."""
    outDir = outDir or env.path("analysis")
    outDir.mkdir(parents=True, exist_ok=True)
    reps = _reps(env, reps)
    built = {
        "summary": buildSummary(env, reps),
        "deltas": buildDeltas(env, reps),
        "effects": buildEffects(env, reps),
        "truncation": buildTruncation(env, reps),
        "glossary": buildGlossary(),
    }
    written = {}
    for name, frame in built.items():
        path = outDir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        written[name] = path
    return written


TABLE_COLUMNS = ["model", "condition", "n", "asr", "utility",
                 "overBlock", "invalidOutput", "markedRate",
                 "meanTokens", "latP50"]


def writeLatexTables(env: pipeline.Environment,
                     summary: pd.DataFrame | None = None,
                     outDir: Path | None = None) -> dict[str, Path]:
    """Grava `resultados_<benchmark>.tex`, sem as linhas de alias."""
    summary = buildSummary(env) if summary is None else summary
    reps = int(summary["reps"].iloc[0])
    outDir = outDir or env.path("tables")
    outDir.mkdir(parents=True, exist_ok=True)

    written = {}
    for benchmark in BENCHMARKS:
        frame = summary[(summary["benchmark"] == benchmark) & ~summary["alias"]]
        written[benchmark] = metrics.toLatexTable(
            frame[TABLE_COLUMNS].to_dict("records"), TABLE_COLUMNS,
            caption=(f"Métricas por condição e modelo no benchmark {BENCH_TITLE[benchmark]}"
                     + (f", médias de {reps} repetições." if reps > 1 else ".")),
            label=f"tab:resultados-{benchmark}",
            outPath=outDir / f"resultados_{benchmark}.tex",
            legend=True)
    return written
