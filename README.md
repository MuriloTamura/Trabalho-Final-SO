# Simulação de Impacto da Política de Escalonamento de CPU sobre o Comportamento do Sistema de Arquivos

Este repositório contém o projeto final desenvolvido para a disciplina de **Sistemas Operacionais**. A aplicação utiliza a linguagem **Python** para simular, analisar e visualizar como diferentes algoritmos de escalonamento de processos impactam diretamente o subsistema de arquivos e o desempenho geral do computador.

---

## 📌 Descrição do Projeto

O objetivo deste trabalho é simular o comportamento de um Sistema Operacional sob cenários de alta concorrência. O grande diferencial do simulador é analisar além da CPU: ele mapeia como as decisões de escalonamento (fatias de tempo, preempção e prioridades) afetam métricas críticas do sistema de arquivos, tais como:

* **Latência e Throughput de E/S (I/O):** Gargalos gerados pelo travamento de filas de recursos periféricos.
* **Taxa de Acerto de Cache (Cache Hit %):** O impacto direto das trocas de contexto na invalidação do cache de arquivos.
* **Fragmentação de Arquivos:** Como o tempo de CPU disponível dita o ritmo de escrita em disco (`extents` médios por arquivo).
* **Starvation:** Análise rigorosa do tempo máximo de espera sob estresse crônico.

### Algoritmos Simulados e Comparados:
* **FCFS** (First-Come, First-Served) — Não preemptivo.
* **SJF** (Shortest Job First) — Não preemptivo.
* **SRTF** (Shortest Remaining Time First) — Preemptivo.
* **Round Robin (Alternância Circular)** — Preemptivo com ajuste dinâmico de *Quantum*.
* **Prioridade Dinâmica com Aging (Envelhecimento)** — Preemptivo.

---

## 📂 Estrutura de Arquivos

O repositório está organizado de forma modular para separar a lógica de simulação da camada de apresentação visual:

* `main.py`: O ponto de entrada da aplicação. Coordena a inicialização dos cenários, gerencia a linha do tempo da simulação e interliga os módulos core e de renderização.
* `simulacao.py`: Contém o **núcleo (kernel) da simulação**. Implementa as estruturas de dados das filas (Prontos, Bloqueados/IO), a máquina de estados dos processos, as políticas de escalonamento e os contadores estatísticos.
* `visualizacao.py`: Módulo responsável por consolidar as métricas ao final da execução e gerar os relatórios visuais. Renderiza os gráficos estatísticos comparativos (radar normativo, comportamento de filas e diagramas de starvation).
* `.gitignore`: Configuração para evitar o versionamento de ambientes virtuais (`.venv`), caches do Python (`__pycache__`) e os arquivos de imagem gerados pelos testes.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3 (100%)
* **Análise de Dados e Gráficos:** `matplotlib` / `seaborn` (utilizados para a estruturação dos painéis visuais).

 ---

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python 3.10+ e o gerenciador de pacotes `pip` instalados em seu ambiente Linux/Ubuntu ou equivalente.

### 1. Clonar o Repositório
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio