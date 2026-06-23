"""
Módulo de visualização — gera gráficos comparativos entre os algoritmos.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from simulacao import executar_simulacao, MetricasSistema  # usa simulacao.py do mesmo diretório
from typing import List


# ──────────────────────────────────────────────
# Paleta e estilo
# ──────────────────────────────────────────────

CORES = {
    "FCFS":                      "#2563EB",
    "SRTF":                      "#0891B2",
    "SJF":                       "#16A34A",
    "Round Robin":               "#D97706",
    "Prioridade":                "#DC2626",
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
    # BUG CORRIGIDO: o nome completo do escalonador é
    # "SRTF (SJF Preemptivo)", que contém a substring "SJF". Com "SJF"
    # testado ANTES de "SRTF" no dict (ordem de inserção), toda chamada
    # para SRTF caía no `"SJF" in algoritmo` primeiro e retornava a cor
    # do SJF — por isso SRTF aparecia pintado igual ao SJF em todos os
    # gráficos. Corrigido reordenando CORES para testar "SRTF" antes de
    # "SJF" (chave mais específica primeiro).
    for k, v in CORES.items():
        if k in algoritmo:
            return v
    return "#6B7280"


def _nomes_curtos(resultados: List[MetricasSistema]) -> List[str]:
    mapa = {
        "FCFS":        "FCFS",
        "SRTF":        "SRTF",
        "SJF":         "SJF",
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

    # Throughput global (processos / makespan) — ver nota técnica: esta
    # métrica é pouco discriminativa quando o makespan é dominado pela carga
    # de trabalho fixa, não pela política. Por isso some-se a eficiência de
    # retorno (1/tempo_médio_retorno) como segunda linha no rótulo de cada
    # barra: ela reage de forma mais visível às diferenças entre algoritmos.
    vals = [m.throughput for m in resultados]
    vals_efic = [m.eficiencia_retorno for m in resultados]
    bars = ax_throughput.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_throughput.set_title("Throughput CPU (barra) vs\nEficiência de Retorno 1/T.Ret (itálico)",
                              fontweight="bold", fontsize=10)
    ax_throughput.set_ylabel("proc / t")
    ax_throughput.set_xticks(x); ax_throughput.set_xticklabels(nomes)
    ax_throughput.set_ylim(0, max(vals) * 1.35 if vals else 1)
    for bar, v, ve in zip(bars, vals, vals_efic):
        ax_throughput.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                           f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_throughput.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.10,
                           f"({ve:.4f})", ha="center", va="bottom", fontsize=7.5, style="italic",
                           color="#374151")


# ──────────────────────────────────────────────
# Gráfico 2 — Métricas de I/O
# ──────────────────────────────────────────────

def grafico_io(resultados, ax_latencia, ax_throughput_io, ax_cache):
    nomes = _nomes_curtos(resultados)
    cores = [_cor(m.algoritmo) for m in resultados]
    x     = np.arange(len(resultados))
    width = 0.55

    # Latência de I/O — decomposta em espera de fila (disco ocupado) + serviço
    # (seek + rotação + transferência), para deixar visível POR QUE algoritmos
    # com menor espera de CPU podem ter maior latência de I/O: eles geram ondas
    # de requisições mais simultâneas, que saturam o único braço de disco.
    vals_fila    = [m.espera_media_fila_io for m in resultados]
    vals_servico = [m.servico_medio_io for m in resultados]
    bars_fila = ax_latencia.bar(x, vals_fila, width, color=cores, edgecolor="white",
                                 linewidth=1.5, alpha=0.55, hatch="//", label="Espera na fila")
    bars_serv = ax_latencia.bar(x, vals_servico, width, bottom=vals_fila, color=cores,
                                 edgecolor="white", linewidth=1.5, label="Serviço (seek+rot+transf.)")
    ax_latencia.set_title("Latência Média de I/O\n(fila + serviço)", fontweight="bold", fontsize=11)
    ax_latencia.set_ylabel("Unidades de tempo")
    ax_latencia.set_xticks(x); ax_latencia.set_xticklabels(nomes)
    ax_latencia.legend(fontsize=6.5, loc="upper left")
    for i, (vf, vs) in enumerate(zip(vals_fila, vals_servico)):
        total = vf + vs
        ax_latencia.text(x[i], total + 0.1, f"{total:.1f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")

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
# Gráfico 4 — Tamanho da fila de E/S ao longo do tempo
# ──────────────────────────────────────────────

def grafico_fila_io(resultados: List[MetricasSistema], ax):
    """Plota o tamanho da fila de E/S a cada tick para cada algoritmo.

    Este gráfico é central para o tema do trabalho: mostra como a política
    de escalonamento de CPU molda o padrão de chegada de requisições ao
    subsistema de arquivos. Round Robin com quantum pequeno tende a gerar
    picos mais frequentes; FCFS preserva rajadas concentradas.
    """
    for m in resultados:
        if not m.historico_fila_io:
            continue
        # suaviza com média móvel de janela 5 para legibilidade
        serie = np.array(m.historico_fila_io, dtype=float)
        janela = min(5, len(serie))
        if janela > 1:
            serie = np.convolve(serie, np.ones(janela) / janela, mode="same")
        ticks = np.arange(len(serie))
        cor = _cor(m.algoritmo)
        ax.plot(ticks, serie, linewidth=1.8, color=cor, label=m.algoritmo, alpha=0.85)

    ax.set_title("Fila de E/S ao Longo do Tempo\n(média móvel janela=5)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Tick de simulação")
    ax.set_ylabel("Requisições na fila")
    ax.legend(fontsize=8, loc="upper right")


# ──────────────────────────────────────────────
# Gráfico 5 — Fragmentação, trocas de contexto e utilização de CPU
# ──────────────────────────────────────────────

def grafico_overhead(resultados: List[MetricasSistema], ax_frag, ax_ctx, ax_util):
    """Três métricas de overhead que complementam o quadro geral."""
    nomes  = _nomes_curtos(resultados)
    cores  = [_cor(m.algoritmo) for m in resultados]
    x      = np.arange(len(resultados))
    width  = 0.55

    # Fragmentação (extents por arquivo)
    vals = [m.media_extents_por_arquivo for m in resultados]
    bars = ax_frag.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_frag.set_title("Fragmentação\n(extents médios / arquivo)", fontweight="bold", fontsize=10)
    ax_frag.set_ylabel("extents")
    ax_frag.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1, label="sem fragmentação (1.0)")
    ax_frag.set_xticks(x); ax_frag.set_xticklabels(nomes)
    ax_frag.legend(fontsize=7)
    for bar, v in zip(bars, vals):
        ax_frag.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Trocas de contexto
    vals = [m.trocas_contexto for m in resultados]
    bars = ax_ctx.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_ctx.set_title("Trocas de Contexto", fontweight="bold", fontsize=10)
    ax_ctx.set_ylabel("quantidade")
    ax_ctx.set_xticks(x); ax_ctx.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_ctx.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Utilização da CPU
    vals = [m.utilizacao_cpu * 100 for m in resultados]
    bars = ax_util.bar(x, vals, width, color=cores, edgecolor="white", linewidth=1.5)
    ax_util.set_title("Utilização da CPU (%)", fontweight="bold", fontsize=10)
    ax_util.set_ylabel("%")
    ax_util.set_ylim(0, 110)
    ax_util.set_xticks(x); ax_util.set_xticklabels(nomes)
    for bar, v in zip(bars, vals):
        ax_util.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")


def grafico_starvation(resultados, ax):
    """
    BUG CORRIGIDO — cor da barra desalinhada do que ela representa:
    Na versão anterior, a barra ia de 0 até a MÉDIA e era pintada de
    vermelho inteira quando havia starvation. Como o critério de
    starvation é a MÁXIMA individual (losango), não a média, isso
    produzia barras vermelhas que terminavam BEM ANTES do limiar
    (ex.: SRTF com média 25.8 pintada vermelha, losango da máxima a
    123.0 — bem mais à direita, fora da barra) — visualmente parece
    "a barra errada foi marcada", embora a flag em si estivesse correta.

    Correção: a barra agora desenha o intervalo COMPLETO de espera de
    cada algoritmo, da média até a máxima individual (não só até a
    média). O trecho até a média usa a cor do algoritmo; o trecho de
    média até máxima é desenhado em hachura. Só a fração da barra que
    está de fato ACIMA do limiar de starvation é pintada em vermelho —
    assim a área vermelha e o losango da máxima sempre coincidem, e a
    barra nunca "termina" antes do ponto que motivou a marcação.
    """
    nomes  = _nomes_curtos(resultados)
    esperas = [m.tempo_medio_espera for m in resultados]
    esperas_max = [m.tempo_max_espera for m in resultados]

    limiar = resultados[0].limiar_starvation if resultados else 100

    y = np.arange(len(resultados))
    altura = 0.5

    for yi, m, v, vmax in zip(y, resultados, esperas, esperas_max):
        cor_base = _cor(m.algoritmo)
        # 1) trecho 0 -> média: cor do algoritmo (ou vermelho se a própria
        #    média já superou o limiar)
        cor_media = "#DC2626" if v > limiar else cor_base
        ax.barh(yi, v, height=altura, color=cor_media, edgecolor="white", linewidth=1.5, zorder=2)
        # 2) trecho média -> máxima: hachurado, mesma lógica de cor por
        #    segmento (abaixo do limiar = cor do algoritmo, acima = vermelho)
        if vmax > v:
            if v >= limiar:
                ax.barh(yi, vmax - v, left=v, height=altura, color="#DC2626",
                        edgecolor="white", linewidth=1.0, hatch="//", alpha=0.55, zorder=2)
            elif vmax > limiar:
                # o segmento cruza o limiar: parte cor base, parte vermelha
                ax.barh(yi, limiar - v, left=v, height=altura, color=cor_base,
                        edgecolor="white", linewidth=1.0, hatch="//", alpha=0.55, zorder=2)
                ax.barh(yi, vmax - limiar, left=limiar, height=altura, color="#DC2626",
                        edgecolor="white", linewidth=1.0, hatch="//", alpha=0.55, zorder=2)
            else:
                ax.barh(yi, vmax - v, left=v, height=altura, color=cor_base,
                        edgecolor="white", linewidth=1.0, hatch="//", alpha=0.55, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.invert_yaxis()  # mantém a mesma ordem de cima->baixo da versão anterior (barh inverte por padrão)

    # Marcador da espera MÁXIMA individual — é ela, não a média, que decide
    # starvation. Agora ele cai sempre na ponta da barra (fim do hachurado).
    ax.scatter(esperas_max, y, marker="D", s=55, color="#1F2937", zorder=5,
               label="Espera MÁXIMA individual (decide starvation)")
    ax.axvline(x=limiar, color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"Limiar starvation ({limiar} ticks)")
    ax.set_title("Tempo Médio de Espera & Starvation\n"
                  "(barra sólida = média | hachura = intervalo até a máxima | vermelho = trecho acima do limiar)",
                  fontweight="bold", fontsize=11)
    ax.set_xlabel("Unidades de tempo")

    # Reserva margem suficiente no eixo X para o texto do rótulo não ser
    # cortado/sobreposto na borda direita (ver nota de correção anterior).
    maior_valor = max(esperas_max + esperas + [limiar])
    margem = maior_valor * 0.55
    ax.set_xlim(0, maior_valor + margem)

    for yi, m, v, vmax in zip(y, resultados, esperas, esperas_max):
        label = f"méd {v:.1f} / máx {vmax:.1f}" + (" ⚠ STARVATION" if m.starvation_detectado else "")
        ax.text(max(v, vmax) + maior_valor * 0.015, yi,
                label, va="center", fontsize=9, fontweight="bold",
                color="#DC2626" if m.starvation_detectado else "#1F2937")

    ax.legend(fontsize=8, loc="lower right")


# ──────────────────────────────────────────────
# Figura principal
# ──────────────────────────────────────────────

def gerar_figura(resultados: List[MetricasSistema], caminho: str = "resultado.png"):
    fig = plt.figure(figsize=(20, 22), facecolor="#F8FAFC")
    fig.suptitle(
        "Impacto da Política de Escalonamento de CPU\nsobre o Comportamento do Sistema de Arquivos",
        fontsize=16, fontweight="bold", y=0.99, color="#1F2937"
    )

    gs = fig.add_gridspec(5, 4, hspace=0.65, wspace=0.42,
                          left=0.07, right=0.97, top=0.95, bottom=0.04)

    # ── Linha 1: métricas de CPU ──────────────────────────────────────────
    ax_esp  = fig.add_subplot(gs[0, 0])
    ax_ret  = fig.add_subplot(gs[0, 1])
    ax_thr  = fig.add_subplot(gs[0, 2])
    grafico_cpu(resultados, ax_esp, ax_ret, ax_thr)

    # Radar ocupa coluna 3, linhas 0-1
    ax_rad = fig.add_subplot(gs[0:2, 3], polar=True)
    grafico_radar(resultados, ax_rad)

    # ── Linha 2: métricas de I/O ──────────────────────────────────────────
    ax_lat  = fig.add_subplot(gs[1, 0])
    ax_tio  = fig.add_subplot(gs[1, 1])
    ax_cac  = fig.add_subplot(gs[1, 2])
    grafico_io(resultados, ax_lat, ax_tio, ax_cac)

    # ── Linha 3: fila de E/S ao longo do tempo (largura total) ───────────
    ax_fila = fig.add_subplot(gs[2, :])
    grafico_fila_io(resultados, ax_fila)

    # ── Linha 4: overhead — fragmentação, trocas de contexto, utilização ──
    ax_frag = fig.add_subplot(gs[3, 0])
    ax_ctx  = fig.add_subplot(gs[3, 1])
    ax_util = fig.add_subplot(gs[3, 2])
    grafico_overhead(resultados, ax_frag, ax_ctx, ax_util)
    # coluna 3 vazia nessa linha (radar já ocupou acima)

    # ── Linha 5: starvation (largura total) ───────────────────────────────
    ax_sta = fig.add_subplot(gs[4, :])
    grafico_starvation(resultados, ax_sta)

    # ── Rodapé ────────────────────────────────────────────────────────────
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