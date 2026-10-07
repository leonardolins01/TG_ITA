"""Desenha as figuras (grade 3×3: linhas × benchmarks) só a partir de
`results/analysis/*.csv`; nunca abre `results/runs/`.

Uso:
    python makeFigures.py                  # le results/analysis, grava results/figures
    python makeFigures.py <entrada> <saida>
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sem janela: roda em terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis import BENCH_TITLE, BENCHMARKS, CONDITION_ORDER as ORDER, ROWS as MODELS  # noqa: E402

SHORT = {
    "no_filter": "sem\nfiltro",
    "simple": "prompt\nsimples",
    "delimiting": "delim.",
    "demarking": "marc.",
    "delimiting_demarking": "delim.+\nmarc.",
}

# cinza nas referências; azul, mais escuro quanto mais intervenção
COLORS = {
    "no_filter": "#b0b0b0",
    "simple": "#6e6e6e",
    "delimiting": "#8fbcd9",
    "demarking": "#3d7ea6",
    "delimiting_demarking": "#14425f",
}

# cada efeito do 2×2 com a cor da condição que ele isola
EFFECT_ORDER = ["delimiting", "marking", "interaction"]
EFFECT_LABEL = {"delimiting": "efeito da\ndelimitação",
                "marking": "efeito da\nmarcação",
                "interaction": "interação"}
EFFECT_COLOR = {"delimiting": "#8fbcd9", "marking": "#3d7ea6",
                "interaction": "#14425f"}

MODEL_TITLE = {"llama-3.1-8b": "llama-3.1-8b",
               "gpt-oss-20b": "gpt-oss-20b",
               "gpt-oss-safeguard": "gpt-oss-safeguard (filtro)\n+ gpt-oss-20b"}


def _grid(title: str, subtitle: str, unit: str, bare: bool = False):
    """Figura 3×3 vazia com títulos de coluna; `bare` omite título e subtítulo (variante do LaTeX)."""
    nRows = len(MODELS)
    fig, axes = plt.subplots(nRows, 3, figsize=(12.6, 3.15 * nRows + (0.1 if bare else 0.8)))
    if not bare:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
        fig.text(0.5, 0.958, subtitle, ha="center", fontsize=9.5, color="#555555")
    fig.rectTop = 0.985 if bare else 0.945
    fig.rowAxes = [axes[row][0] for row in range(nRows)]
    for col, benchmark in enumerate(BENCHMARKS):
        axes[0][col].set_title(BENCH_TITLE[benchmark], fontsize=11.5, pad=9)
    for row, model in enumerate(MODELS):
        axes[row][0].set_ylabel(unit, fontsize=9, color="#444444")
    return fig, axes


def _finish(fig):
    """Fecha o layout e escreve os rótulos de linha na altura de cada linha."""
    fig.tight_layout(rect=(0.032, 0, 1, fig.rectTop))
    for ax, model in zip(fig.rowAxes, MODELS):
        box = ax.get_position()
        fig.text(0.013, (box.y0 + box.y1) / 2, MODEL_TITLE[model], rotation=90,
                 va="center", ha="center", fontsize=10.5, fontweight="bold")
    return fig


def _alignRow(axes, low: float, high: float) -> None:
    """Mesma escala de y nos painéis vivos de uma linha."""
    for ax in axes:
        ax.set_ylim(low, high)


def _blank(ax, reason: str) -> None:
    """Marca o painel como não aplicável, com o motivo escrito dentro."""
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=9.5,
            color="#8a8a8a", style="italic", transform=ax.transAxes,
            linespacing=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _dress(ax, conditions: list[str], labels: dict | None = None) -> None:
    """Eixo x e estilo comuns a todos os painéis."""
    labels = labels or SHORT
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([labels[c] for c in conditions], fontsize=7.6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _valueLabel(fmt: str, value: float) -> str:
    """O valor formatado; abaixo da última casa, `<0.001` em vez de um zero sobre barra visível."""
    label = fmt.format(value)
    if value > 0 and float(label) == 0:
        digits = int(re.search(r"\.(\d+)f", fmt).group(1))
        label = "<" + fmt.format(10 ** -digits)
    return label


def plotAbsolute(summary: pd.DataFrame, column: str, title: str, subtitle: str,
                 unit: str, blankOn=(), blankWhy: str = "",
                 fmt: str = "{:.3f}", bare: bool = False):
    """Uma barra por condição, com o valor anotado e, se houver `<coluna>Low/High`, a haste do IC."""
    fig, axes = _grid(title, subtitle, unit, bare)
    withCi = f"{column}Low" in summary.columns
    for row, model in enumerate(MODELS):
        live, top = [], 0.0
        for col, benchmark in enumerate(BENCHMARKS):
            ax = axes[row][col]
            if benchmark in blankOn:
                _blank(ax, blankWhy)
                continue
            values, lows, highs = [], [], []
            for condition in ORDER:
                hit = summary[(summary.benchmark == benchmark)
                              & (summary.model == model)
                              & (summary.condition == condition)]
                value = float(hit[column].iloc[0]) if len(hit) else np.nan
                values.append(value)
                lows.append(float(hit[f"{column}Low"].iloc[0]) if withCi and len(hit) else value)
                highs.append(float(hit[f"{column}High"].iloc[0]) if withCi and len(hit) else value)
            bars = ax.bar(range(len(ORDER)), values, width=0.72,
                          color=[COLORS[c] for c in ORDER],
                          edgecolor="white", linewidth=0.9)
            highs = [v if np.isnan(h) else h for v, h in zip(values, highs)]
            if withCi:
                for x, (lo, hi) in enumerate(zip(lows, highs)):
                    if np.isnan(lo):
                        continue
                    ax.vlines(x, lo, hi, color="#2b2b2b", linewidth=1.1, zorder=3)
                    ax.hlines([lo, hi], x - 0.08, x + 0.08, color="#2b2b2b",
                              linewidth=1.0, zorder=3)
            top = max(top, max(v for v in highs if not np.isnan(v)))
            for bar, value, high in zip(bars, values, highs):
                if not np.isnan(value):
                    ax.annotate(_valueLabel(fmt, value),
                                (bar.get_x() + bar.get_width() / 2, high),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7.1, color="#333333")
            _dress(ax, ORDER)
            live.append(ax)
        _alignRow(live, 0, (top or 1.0) * 1.20)
    return _finish(fig)


def _pointsWithCi(ax, xs, points, lows, highs, colors) -> None:
    """Cada estimativa como ponto com a haste do intervalo."""
    for x, pt, lo, hi, color in zip(xs, points, lows, highs, colors):
        ax.vlines(x, lo, hi, color="#2b2b2b", linewidth=1.3, zorder=2)
        ax.hlines([lo, hi], x - 0.09, x + 0.09, color="#2b2b2b", linewidth=1.1, zorder=2)
        ax.scatter([x], [pt], s=62, color=color, edgecolor="#1b1b1b",
                   linewidth=0.8, zorder=3)


def plotDelta(deltas: pd.DataFrame, metric: str, baseline: str, title: str,
              subtitle: str, unit: str, blankOn=(), blankWhy: str = "",
              bare: bool = False):
    """Diferenças contra `baseline`, ponto com IC 95%, linha do zero."""
    fig, axes = _grid(title, subtitle, unit, bare)
    conditions = [c for c in ORDER if c != baseline]
    for row, model in enumerate(MODELS):
        live, span, panels = [], 0.0, []
        for col, benchmark in enumerate(BENCHMARKS):
            ax = axes[row][col]
            if benchmark in blankOn:
                _blank(ax, blankWhy)
                continue
            points, lows, highs = [], [], []
            for condition in conditions:
                hit = deltas[(deltas.benchmark == benchmark)
                             & (deltas.model == model)
                             & (deltas.condition == condition)
                             & (deltas.metric == metric)
                             & (deltas.baseline == baseline)]
                if not len(hit):  # célula ausente: sem ponto, não um zero
                    points.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                    continue
                r = hit.iloc[0]
                points.append(float(r.point))
                lows.append(float(r.ciLow))
                highs.append(float(r.ciHigh))
            span = max([span] + [abs(v) for v in lows + highs if not np.isnan(v)])
            panels.append((ax, points, lows, highs))
            live.append(ax)
        span = max(span, 0.02) * 1.32
        for ax, points, lows, highs in panels:
            ax.axhline(0, color="#333333", linewidth=0.9)
            idx = [i for i, v in enumerate(points) if not np.isnan(v)]
            _pointsWithCi(ax, idx, [points[i] for i in idx], [lows[i] for i in idx],
                          [highs[i] for i in idx], [COLORS[conditions[i]] for i in idx])
            ax.set_xlim(-0.6, len(conditions) - 0.4)
            _dress(ax, conditions)
        _alignRow(live, -span, span)
    return _finish(fig)


def plotEffects(effects: pd.DataFrame, metric: str, title: str, subtitle: str,
                unit: str, blankOn=(), blankWhy: str = "", bare: bool = False):
    """Efeitos principais e interação do 2×2, ponto com IC 95%."""
    fig, axes = _grid(title, subtitle, unit, bare)
    for row, model in enumerate(MODELS):
        live, span, panels = [], 0.0, []
        for col, benchmark in enumerate(BENCHMARKS):
            ax = axes[row][col]
            hit = effects[(effects.benchmark == benchmark) & (effects.model == model)
                          & (effects.metric == metric)]
            if benchmark in blankOn or not len(hit):
                _blank(ax, blankWhy)
                continue
            vals = {r.effect: r for r in hit.itertuples()}
            points = [float(vals[e].point) for e in EFFECT_ORDER]
            lows = [float(vals[e].ciLow) for e in EFFECT_ORDER]
            highs = [float(vals[e].ciHigh) for e in EFFECT_ORDER]
            span = max([span] + [abs(v) for v in lows + highs])
            panels.append((ax, points, lows, highs))
            live.append(ax)
        span = max(span, 0.02) * 1.32
        for ax, points, lows, highs in panels:
            ax.axhline(0, color="#333333", linewidth=0.9)
            _pointsWithCi(ax, range(len(EFFECT_ORDER)), points, lows, highs,
                          [EFFECT_COLOR[e] for e in EFFECT_ORDER])
            ax.set_xlim(-0.6, len(EFFECT_ORDER) - 0.4)
            _dress(ax, EFFECT_ORDER, EFFECT_LABEL)
        _alignRow(live, -span, span)
    return _finish(fig)


def plotLatency(summary: pd.DataFrame, bare: bool = False):
    """Tempo de geração mediano (barra) e percentil 95 (traço), medido pelo provedor."""
    fig, axes = _grid("Tempo de geração por item",
                      "medido pelo provedor, sem fila nem rede; barra = mediana, "
                      "traço = percentil 95; em segundos, somando filtro e componente",
                      "segundos", bare)
    for row, model in enumerate(MODELS):
        live, top = [], 0.0
        for col, benchmark in enumerate(BENCHMARKS):
            ax = axes[row][col]
            p50, p95 = [], []
            for condition in ORDER:
                hit = summary[(summary.benchmark == benchmark)
                              & (summary.model == model)
                              & (summary.condition == condition)]
                p50.append(float(hit.latP50.iloc[0]) if len(hit) else np.nan)
                p95.append(float(hit.latP95.iloc[0]) if len(hit) else np.nan)
            top = max(top, np.nanmax(p95))
            ax.bar(range(len(ORDER)), p50, width=0.72,
                   color=[COLORS[c] for c in ORDER],
                   edgecolor="white", linewidth=0.9)
            ax.scatter(range(len(ORDER)), p95, marker="_", s=330,
                       color="#111111", linewidths=1.6, zorder=5)
            _dress(ax, ORDER)
            live.append(ax)
        _alignRow(live, 0, top * 1.15)
    return _finish(fig)


NOT_IN_NOTINJECT = ("não se aplica\n\nO NotInject é só-benigno:\n"
                    "não há ataque, logo não há\ndenominador para o ASR")


def main() -> int:
    inDir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    outDir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if inDir is None:
        from src import pipeline
        env = pipeline.Environment.load()
        inDir, outDir = env.path("analysis"), env.path("figures")
    outDir = outDir or inDir.parent / "figures"
    outDir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(inDir / "summary.csv")
    deltas = pd.read_csv(inDir / "deltas.csv")
    effects = pd.read_csv(inDir / "effects.csv")
    print(f"lendo {inDir}\n(nenhum registro por item é aberto)\n")

    written = _emit(_build(summary, deltas, effects, bare=False), outDir,
                    (".pdf", ".png"))
    print(f"{written} figuras (pdf + png) em {outDir}")

    # variante sem título para o LaTeX, na mesma execução
    texDir = inDir.parent.parent / "latex" / "figs"
    if texDir.parent.is_dir():
        texDir.mkdir(parents=True, exist_ok=True)
        written = _emit(_build(summary, deltas, effects, bare=True), texDir,
                        (".pdf",))
        print(f"{written} figuras (pdf sem título, para o LaTeX) em {texDir}")
    return 0


def _emit(figures: dict, outDir, suffixes) -> int:
    """Grava as figuras nos formatos pedidos e as fecha."""
    for name, fig in figures.items():
        for suffix in suffixes:
            fig.savefig(outDir / f"{name}{suffix}", bbox_inches="tight",
                        **({"dpi": 150} if suffix == ".png" else {}))
        plt.close(fig)
    return len(figures)


def _build(summary, deltas, effects, bare: bool) -> dict:
    """Desenha as doze figuras e as devolve por nome."""
    return {
        "asr": plotAbsolute(
            summary, "asr", "Taxa de sucesso do ataque (ASR)",
            "de cada 100 ataques, quantos o componente principal executou — "
            "menor é melhor", "ASR",
            blankOn=("notinject",), blankWhy=NOT_IN_NOTINJECT, bare=bare),
        "delta_asr_vs_simple": plotDelta(
            deltas, "asr", "simple",
            "O efeito da separação de canal",
            "ΔASR contra o prompt simples, IC 95% BCa sobre os itens",
            "ΔASR",
            blankOn=("notinject",), blankWhy=NOT_IN_NOTINJECT, bare=bare),
        "delta_utility_vs_simple": plotDelta(
            deltas, "utility", "simple",
            "O que a separação de canal custa em utilidade",
            "Δutilidade contra o prompt simples, IC 95% BCa sobre os itens",
            "Δutilidade", bare=bare),
        "over_block": plotAbsolute(
            summary, "overBlock", "Sobre-bloqueio",
            "fração do conteúdo legítimo que o filtro bloqueou por engano — "
            "menor é melhor", "sobre-bloqueio", bare=bare),
        "utility": plotAbsolute(
            summary, "utility", "Utilidade",
            "fração das tarefas legítimas cumpridas; item bloqueado conta como "
            "fracasso — maior é melhor", "utilidade", bare=bare),
        "marked_rate": plotAbsolute(
            summary, "markedRate", "Alcance da marcação seletiva",
            "dos itens liberados, em quantos o filtro apontou algum trecho para "
            "marcar — sem trecho apontado, não há o que marcar", "alcance", bare=bare),
        "effects_asr": plotEffects(
            effects, "asr", "Efeitos do desenho 2x2 sobre o ASR",
            "efeito principal = média das duas condições com o fator menos a das "
            "duas sem; IC 95% BCa — negativo é melhor", "efeito no ASR",
            blankOn=("notinject",), blankWhy=NOT_IN_NOTINJECT, bare=bare),
        "effects_utility": plotEffects(
            effects, "utility", "Efeitos do desenho 2x2 sobre a utilidade",
            "efeito principal = média das duas condições com o fator menos a das "
            "duas sem; IC 95% BCa — negativo é custo", "efeito na utilidade",
            bare=bare),
        "invalid_output": plotAbsolute(
            summary, "invalidOutput", "Saída inválida do filtro",
            "fração das saídas que não puderam ser interpretadas; contam "
            "conservadoramente como não bloqueio", "saída inválida", bare=bare),
        "empty_output": plotAbsolute(
            summary, "emptyOutput", "Resposta vazia do componente principal",
            "fração dos itens liberados sem resposta; conta como saída inválida, "
            "não como bloqueio nem sobre-bloqueio", "resposta vazia", bare=bare),
        "cost_tokens": plotAbsolute(
            summary, "meanTokens", "Custo em tokens por item",
            "média somando o filtro e o componente principal; delimitar e marcar "
            "acrescentam texto", "tokens", fmt="{:.0f}", bare=bare),
        "cost_latency": plotLatency(summary, bare=bare),
    }


if __name__ == "__main__":
    raise SystemExit(main())
