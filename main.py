"""
main.py — Ponto de entrada da simulação.

Executa três cenários e gera os gráficos correspondentes.

Uso:
    python3 main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulacao import executar_simulacao, imprimir_tabela
from visualizacao import gerar_figura


# ──────────────────────────────────────────────
# Casos de Teste
# ──────────────────────────────────────────────

CENARIOS = [
    {
        "nome":        "Cenário 1 — Carga Padrão (8 processos, Q=4)",
        "n_processos": 8,
        "quantum":     4,
        "seed":        42,
        "arquivo":     "cenario1.png",
    },
    {
        "nome":        "Cenário 2 — Alta Concorrência (16 processos, Q=4)",
        "n_processos": 16,
        "quantum":     4,
        "seed":        7,
        "arquivo":     "cenario2.png",
    },
    {
        "nome":        "Cenário 3 — Quantum Pequeno (8 processos, Q=2)",
        "n_processos": 8,
        "quantum":     2,
        "seed":        99,
        "arquivo":     "cenario3.png",
    },
    {
        # Carga com muitos processos e quantum pequeno: maximiza concorrência e
        # preempção — o cenário que mais estresa o subsistema de E/S e revela
        # melhor as diferenças entre políticas no sistema de arquivos.
        "nome":        "Cenário 4 — Alta Concorrência + Quantum Pequeno (16 processos, Q=2)",
        "n_processos": 16,
        "quantum":     2,
        "seed":        77,
        "arquivo":     "cenario4.png",
    },
]


def separador(titulo: str):
    print("\n" + "─" * 70)
    print(f"  {titulo}")
    print("─" * 70)


def executar_cenario(cfg: dict):
    separador(cfg["nome"])
    resultados = executar_simulacao(
        n_processos=cfg["n_processos"],
        quantum_rr=cfg["quantum"],
        seed=cfg["seed"],
    )
    imprimir_tabela(resultados)

    # exibe alguns eventos de I/O
    print("\n  [Amostra de eventos de I/O — FCFS]")
    for evt in resultados[0].log_eventos[:5]:
        print(f"    {evt}")
    if len(resultados[0].log_eventos) > 5:
        print(f"    ... (+{len(resultados[0].log_eventos) - 5} eventos)")

    # gera figura
    print(f"\n  Gerando figura '{cfg['arquivo']}'...")
    gerar_figura(resultados, cfg["arquivo"])
    return resultados


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Simulação: Escalonamento de CPU × Sistema de Arquivos          ║")
    print("║  IFCE Campus Maracanaú — Sistemas Operacionais 2026.1           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    todos_resultados = []
    for cfg in CENARIOS:
        res = executar_cenario(cfg)
        todos_resultados.append((cfg["nome"], res))

    # ──────────────────────────────────────────
    # Resumo comparativo entre cenários
    # ──────────────────────────────────────────
    separador("RESUMO — Melhor algoritmo por métrica em cada cenário")
    metricas_keys = [
        ("tempo_medio_espera",       "Menor espera",           min),
        ("tempo_medio_retorno",      "Menor retorno",          min),
        ("throughput",               "Maior throughput CPU",   max),
        ("utilizacao_cpu",           "Maior utilização CPU",   max),
        ("trocas_contexto",          "Menos trocas contexto",  min),
        ("latencia_media_io",        "Menor latência I/O",     min),
        ("throughput_io_kb",         "Maior throughput I/O",   max),
        ("taxa_cache_hit",           "Maior cache hit",        max),
        ("media_extents_por_arquivo","Menor fragmentação",     min),
    ]

    for nome_cenario, resultados in todos_resultados:
        print(f"\n  {nome_cenario}")
        for attr, label, fn in metricas_keys:
            melhor = fn(resultados, key=lambda m: getattr(m, attr))
            print(f"    {label:<26}: {melhor.algoritmo} "
                  f"({getattr(melhor, attr):.4f})")

    print("\n✓ Simulação concluída. Gráficos salvos nos arquivos PNG.")


if __name__ == "__main__":
    main()