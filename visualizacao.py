"""
Módulo de visualização — gera gráficos comparativos entre os algoritmos.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from simulacao import executar_simulacao, MetricasSistema
from typing import List


# ──────────────────────────────────────────────
# Paleta e estilo
# ──────────────────────────────────────────────

CORES = {
    "FCFS":                      "#2563EB",
    "SJF":                       "#16A34A",
    "Round Robin (Q=4)":         "#D97706",
    "Prioridade (Preemptivo)":   "#DC2626",
}

CORES_LISTA = list(CORES.values())

plt.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor":   "#F8FAFC",
})


def _cor(algoritmo: str) -> str:
    for k, v in CORES.items():
        if k in algoritmo:
            return v
    return "#6B7280"


def _nomes_curtos(resultados: List[MetricasSistema]) -> List[str]:
    mapa = {
        "FCFS": "FCFS",
        "SJF":  "SJF",
        "Round Robin": "RR",
        "Prioridade":  "Prio",
    }
    out = []
    for m in resultados:
        for k, v in mapa.items():
            if k in m.algoritmo:
                out.append(v)
                break
        else:
            out.append(m.algoritmo[:6])
    return out


# ──────────────────────────────────────────────
# Gráfico 1 — Métricas de CPU
# ──────────────────────────────────────────────

def grafico_cpu(resultados: List[MetricasSistema], ax_espera, ax_retorno, ax_throughput):
    nomes  = _nomes_curtos(resultados)
    cores  = [_cor(m.algoritmo) for m in resultados]
    x      = np.arange(len(resultados))
    width  = 0.55

    # Tempo médio de espera
    vals = [m.tempo_medio_espera for m in resultados]
    bars = ax_espera.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_espera.set_title("Tempo Médio de Espera (CPU)", fontweight="bold", fontsize=11)
    ax_espera.set_ylabel("Unidades de tempo")
    ax_espera.set_xticks(x); ax_espera.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_espera.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Tempo médio de retorno
    vals = [m.tempo_medio_retorno for m in resultados]
    bars = ax_retorno.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_retorno.set_title("Tempo Médio de Retorno", fontweight="bold", fontsize=11)
    ax_retorno.set_ylabel("Unidades de tempo")
    ax_retorno.set_xticks(x); ax_retorno.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_retorno.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Throughput
    vals = [m.throughput for m in resultados]
    bars = ax_throughput.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_throughput.set_title("Throughput (proc / unidade de tempo)", fontweight="bold", fontsize=11)
    ax_throughput.set_ylabel("proc / t")
    ax_throughput.set_xticks(x); ax_throughput.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_throughput.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                           f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")


# ──────────────────────────────────────────────
# Gráfico 2 — Métricas de I/O
# ──────────────────────────────────────────────

def grafico_io(resultados, ax_latencia, ax_throughput_io, ax_cache):
    nomes = _nomes_curtos(resultados)
    cores = [_cor(m.algoritmo) for m in resultados]
    x     = np.arange(len(resultados))
    width = 0.55

    # Latência de I/O
    vals = [m.latencia_media_io for m in resultados]
    bars = ax_latencia.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_latencia.set_title("Latência Média de I/O", fontweight="bold", fontsize=11)
    ax_latencia.set_ylabel("Unidades de tempo")
    ax_latencia.set_xticks(x); ax_latencia.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_latencia.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Throughput de I/O (KB/t)
    vals = [m.throughput_io_kb for m in resultados]
    bars = ax_throughput_io.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_throughput_io.set_title("Throughput de I/O (KB / unidade de tempo)", fontweight="bold", fontsize=11)
    ax_throughput_io.set_ylabel("KB / t")
    ax_throughput_io.set_xticks(x); ax_throughput_io.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_throughput_io.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                              f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Taxa de cache hit
    vals = [m.taxa_cache_hit * 100 for m in resultados]
    bars = ax_cache.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_cache.set_title("Taxa de Cache Hit (%)", fontweight="bold", fontsize=11)
    ax_cache.set_ylabel("%")
    ax_cache.set_ylim(0, 110)
    ax_cache.set_xticks(x); ax_cache.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_cache.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")


# ──────────────────────────────────────────────
# Gráfico 3 — Radar (spider chart)
# ──────────────────────────────────────────────

def grafico_radar(resultados: List[MetricasSistema], ax):
    categorias = ["Throughput\nCPU", "Throughput\nI/O", "Cache\nHit", "Baixa\nEspera", "Baixa\nLatência IO"]
    N = len(categorias)
    angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angulos += angulos[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax_labels = ax.set_xticklabels(categorias, size=9)

    def normalizar(valores):
        vmin, vmax = min(valores), max(valores)
        if vmax == vmin:
            return [0.5] * len(valores)
        return [(v - vmin) / (vmax - vmin) for v in valores]

    throughput_cpu = normalizar([m.throughput        for m in resultados])
    throughput_io  = normalizar([m.throughput_io_kb  for m in resultados])
    cache_hit      = normalizar([m.taxa_cache_hit     for m in resultados])
    inv_espera     = normalizar([1 / max(m.tempo_medio_espera, 0.1) for m in resultados])
    inv_latencia   = normalizar([1 / max(m.latencia_media_io, 0.1)  for m in resultados])

    for i, m in enumerate(resultados):
        valores = [throughput_cpu[i], throughput_io[i], cache_hit[i],
                   inv_espera[i], inv_latencia[i]]
        valores += valores[:1]
        cor = _cor(m.algoritmo)
        ax.plot(angulos, valores, "o-", linewidth=2, color=cor, markersize=4)
        ax.fill(angulos, valores, alpha=0.12, color=cor)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7)
    ax.set_title("Comparativo Normalizado\n(maior = melhor)", fontweight="bold", fontsize=11, pad=20)

    legendas = [mpatches.Patch(color=_cor(m.algoritmo), label=m.algoritmo) for m in resultados]
    ax.legend(handles=legendas, loc="upper right", bbox_to_anchor=(1.55, 1.15),
              fontsize=8, framealpha=0.8)


# ──────────────────────────────────────────────
# Gráfico 4 — Linha temporal de starvation
# ──────────────────────────────────────────────

def grafico_starvation(resultados, ax):
    nomes  = _nomes_curtos(resultados)
    esperas = [m.tempo_medio_espera for m in resultados]
    limiar  = 50  # mesmo da simulação

    cores_barras = []
    for m in resultados:
        cores_barras.append("#DC2626" if m.starvation_detectado else _cor(m.algoritmo))

    bars = ax.barh(nomes, esperas, color=cores_barras, edgecolor="white", linewidth=1.5, height=0.5)
    ax.axvline(x=limiar, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Limiar starvation ({limiar})")
    ax.set_title("Tempo Médio de Espera & Starvation", fontweight="bold", fontsize=11)
    ax.set_xlabel("Unidades de tempo")

    for bar, m, v in zip(bars, resultados, esperas):
        label = f"{v:.1f}" + (" ⚠ STARVATION" if m.starvation_detectado else "")
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, fontweight="bold",
                color="#DC2626" if m.starvation_detectado else "#1F2937")

    ax.legend(fontsize=9)


# ──────────────────────────────────────────────
# Figura principal
# ──────────────────────────────────────────────

def gerar_figura(resultados: List[MetricasSistema], caminho: str = "resultado.png"):
    fig = plt.figure(figsize=(18, 14), facecolor="#F8FAFC")
    fig.suptitle(
        "Impacto da Política de Escalonamento de CPU\nsobre o Comportamento do Sistema de Arquivos",
        fontsize=16, fontweight="bold", y=0.98, color="#1F2937"
    )

    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.42,
                          left=0.07, right=0.97, top=0.91, bottom=0.07)

    # Linha 1 — métricas de CPU
    ax_esp  = fig.add_subplot(gs[0, 0])
    ax_ret  = fig.add_subplot(gs[0, 1])
    ax_thr  = fig.add_subplot(gs[0, 2])
    grafico_cpu(resultados, ax_esp, ax_ret, ax_thr)

    # Radar ocupa coluna 3, linhas 0-1
    ax_rad = fig.add_subplot(gs[0:2, 3], polar=True)
    grafico_radar(resultados, ax_rad)

    # Linha 2 — métricas de I/O
    ax_lat  = fig.add_subplot(gs[1, 0])
    ax_tio  = fig.add_subplot(gs[1, 1])
    ax_cac  = fig.add_subplot(gs[1, 2])
    grafico_io(resultados, ax_lat, ax_tio, ax_cac)

    # Linha 3 — starvation (largura total)
    ax_sta = fig.add_subplot(gs[2, :])
    grafico_starvation(resultados, ax_sta)

    # Rodapé
    fig.text(0.5, 0.01,
             "IFCE – Campus Maracanaú  |  Sistemas Operacionais 2026.1  |  "
             "Simulação: Escalonamento de CPU × Sistema de Arquivos",
             ha="center", fontsize=8, color="#6B7280")

    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"  Figura salva em: {caminho}")
    return fig


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Executando simulação...")
    resultados = executar_simulacao(n_processos=8, quantum_rr=4, seed=42)

    from simulacao import imprimir_tabela
    imprimir_tabela(resultados)

    print("Gerando gráficos...")
    gerar_figura(resultados, "resultado.png")
    print("Concluído.")