"""
Simulação: Impacto da Política de Escalonamento de CPU
       sobre o Comportamento do Sistema de Arquivos

IFCE - Campus Maracanaú | Sistemas Operacionais | 2026.1
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ──────────────────────────────────────────────
# Modelos de dados
# ──────────────────────────────────────────────

class OperacaoArquivo(Enum):
    LEITURA  = "Leitura"
    ESCRITA  = "Escrita"
    ABERTURA = "Abertura"


@dataclass
class RequisicaoArquivo:
    """Representa uma operação de I/O sobre o sistema de arquivos."""
    processo_id: int
    operacao: OperacaoArquivo
    bloco_disco: int          # bloco lógico acessado (0–499)
    tamanho_kb: float         # tamanho da transferência em KB
    tempo_chegada: int        # instante em que a requisição é emitida

    # preenchidos durante simulação
    tempo_inicio_io: int  = 0
    tempo_fim_io: int     = 0

    @property
    def latencia_io(self) -> int:
        return self.tempo_fim_io - self.tempo_inicio_io


@dataclass
class Processo:
    """Processo com rajadas de CPU e requisições de E/S intercaladas."""
    pid: int
    nome: str
    burst_total: int           # tempo total de CPU (unidades)
    prioridade: int            # menor = maior prioridade
    tempo_chegada: int         # instante de entrada na fila de prontos
    requisicoes: List[RequisicaoArquivo] = field(default_factory=list)

    # métricas preenchidas pelo escalonador
    tempo_inicio: int     = -1
    tempo_conclusao: int  = 0
    tempo_espera: int     = 0
    tempo_retorno: int    = 0
    burst_restante: int   = 0

    def __post_init__(self):
        self.burst_restante = self.burst_total


@dataclass
class MetricasSistema:
    """Agrega os resultados de uma execução completa."""
    algoritmo: str
    tempo_medio_espera: float        = 0.0
    tempo_medio_retorno: float       = 0.0
    throughput: float                = 0.0   # processos / unidade de tempo
    latencia_media_io: float         = 0.0
    total_io_operacoes: int          = 0
    throughput_io_kb: float          = 0.0   # KB / unidade de tempo
    taxa_cache_hit: float            = 0.0
    tempo_total_simulacao: int       = 0
    starvation_detectado: bool       = False
    processos_com_starvation: List[int] = field(default_factory=list)
    log_eventos: List[str]           = field(default_factory=list)


# ──────────────────────────────────────────────
# Cache de páginas de arquivo (LRU simples)
# ──────────────────────────────────────────────

class CacheArquivos:
    """Cache LRU com capacidade fixa de blocos."""

    def __init__(self, capacidade: int = 16):
        self.capacidade = capacidade
        self._cache: list = []
        self.hits = 0
        self.misses = 0

    def acessar(self, bloco: int) -> bool:
        """Retorna True se bloco está em cache (hit), False caso contrário."""
        if bloco in self._cache:
            self._cache.remove(bloco)
            self._cache.append(bloco)
            self.hits += 1
            return True
        # miss: carrega bloco
        if len(self._cache) >= self.capacidade:
            self._cache.pop(0)
        self._cache.append(bloco)
        self.misses += 1
        return False

    @property
    def taxa_hit(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ──────────────────────────────────────────────
# Sistema de Arquivos Simulado
# ──────────────────────────────────────────────

class SistemaArquivos:
    """
    Modela o subsistema de I/O.

    Custo de acesso ao disco depende da distância entre a cabeça de leitura
    e o bloco alvo (seek time simplificado). Operações de escrita têm custo
    adicional de commit.
    """

    SEEK_BASE         = 2    # tempo mínimo de seek
    SEEK_POR_CILINDRO = 1    # tempo extra por cilindro de distância
    LATENCIA_ROTACAO  = 3    # latência rotacional fixa
    TAXA_TRANSFER     = 0.5  # unidades de tempo por KB

    def __init__(self, cache: CacheArquivos):
        self.cache = cache
        self.cabeca_disco: int = 0   # posição atual da cabeça
        self.total_kb_transferidos: float = 0.0

    def processar(self, req: RequisicaoArquivo, tempo_atual: int) -> int:
        """
        Simula o processamento de uma requisição e retorna o custo de tempo.
        Atualiza req.tempo_inicio_io e req.tempo_fim_io.
        """
        req.tempo_inicio_io = tempo_atual

        # verifica cache
        cache_hit = self.cache.acessar(req.bloco_disco)

        if cache_hit:
            custo = 1  # acesso rápido à memória
        else:
            distancia = abs(req.bloco_disco - self.cabeca_disco)
            seek = self.SEEK_BASE + distancia * self.SEEK_POR_CILINDRO
            latencia_rot = self.LATENCIA_ROTACAO
            transferencia = max(1, int(req.tamanho_kb * self.TAXA_TRANSFER))
            custo = seek + latencia_rot + transferencia
            self.cabeca_disco = req.bloco_disco

        if req.operacao == OperacaoArquivo.ESCRITA:
            custo += 2  # commit / sync

        self.total_kb_transferidos += req.tamanho_kb
        req.tempo_fim_io = tempo_atual + custo
        return custo


# ──────────────────────────────────────────────
# Algoritmos de Escalonamento
# ──────────────────────────────────────────────

class Escalonador:
    """Classe base para todos os algoritmos."""

    LIMIAR_STARVATION = 50  # espera máxima aceitável

    def __init__(self, nome: str, quantum: int = 4):
        self.nome = nome
        self.quantum = quantum

    def executar(self, processos: List[Processo]) -> MetricasSistema:
        raise NotImplementedError


class EscalonadorFCFS(Escalonador):
    """First-Come, First-Served — não-preemptivo."""

    def __init__(self):
        super().__init__("FCFS")

    def executar(self, processos: List[Processo]) -> MetricasSistema:
        procs = sorted(copy.deepcopy(processos), key=lambda p: p.tempo_chegada)
        cache = CacheArquivos()
        fs    = SistemaArquivos(cache)
        met   = MetricasSistema(self.nome)
        tempo = 0

        for p in procs:
            tempo = max(tempo, p.tempo_chegada)

            if p.tempo_inicio == -1:
                p.tempo_inicio = tempo

            # processa requisições de I/O que chegam antes do fim do burst
            custo_io = 0
            for req in p.requisicoes:
                if req.tempo_chegada <= tempo + p.burst_total:
                    custo_io += fs.processar(req, tempo + p.burst_restante // 2)
                    met.log_eventos.append(
                        f"[t={tempo}] {p.nome}: {req.operacao.value} bloco {req.bloco_disco}"
                    )

            tempo += p.burst_total + custo_io
            p.tempo_conclusao = tempo
            p.tempo_retorno   = p.tempo_conclusao - p.tempo_chegada
            p.tempo_espera    = p.tempo_retorno - p.burst_total

        self._calcular_metricas(procs, fs, met)
        return met

    def _calcular_metricas(self, procs, fs, met):
        met.tempo_total_simulacao  = max(p.tempo_conclusao for p in procs)
        met.tempo_medio_espera     = sum(p.tempo_espera    for p in procs) / len(procs)
        met.tempo_medio_retorno    = sum(p.tempo_retorno   for p in procs) / len(procs)
        met.throughput             = len(procs) / met.tempo_total_simulacao
        met.taxa_cache_hit         = fs.cache.taxa_hit
        todas_reqs = [r for p in procs for r in p.requisicoes]
        met.total_io_operacoes     = len(todas_reqs)
        met.latencia_media_io      = (sum(r.latencia_io for r in todas_reqs) / len(todas_reqs)
                                      if todas_reqs else 0)
        met.throughput_io_kb       = (fs.total_kb_transferidos / met.tempo_total_simulacao
                                      if met.tempo_total_simulacao else 0)
        # starvation
        para_starvation = [p for p in procs if p.tempo_espera > self.LIMIAR_STARVATION]
        met.starvation_detectado   = len(para_starvation) > 0
        met.processos_com_starvation = [p.pid for p in para_starvation]


class EscalonadorSJF(Escalonador):
    """Shortest Job First — não-preemptivo."""

    def __init__(self):
        super().__init__("SJF")

    def executar(self, processos: List[Processo]) -> MetricasSistema:
        procs     = copy.deepcopy(processos)
        cache     = CacheArquivos()
        fs        = SistemaArquivos(cache)
        met       = MetricasSistema(self.nome)
        tempo     = 0
        concluidos = []
        prontos   = []

        restantes = sorted(procs, key=lambda p: p.tempo_chegada)

        while restantes or prontos:
            # adiciona à fila os que já chegaram
            novos = [p for p in restantes if p.tempo_chegada <= tempo]
            for p in novos:
                restantes.remove(p)
                prontos.append(p)

            if not prontos:
                tempo = restantes[0].tempo_chegada
                continue

            # seleciona o de menor burst
            prontos.sort(key=lambda p: p.burst_restante)
            p = prontos.pop(0)

            if p.tempo_inicio == -1:
                p.tempo_inicio = tempo

            custo_io = 0
            for req in p.requisicoes:
                custo_io += fs.processar(req, tempo + p.burst_restante // 2)
                met.log_eventos.append(
                    f"[t={tempo}] {p.nome}: {req.operacao.value} bloco {req.bloco_disco}"
                )

            tempo += p.burst_restante + custo_io
            p.burst_restante   = 0
            p.tempo_conclusao  = tempo
            p.tempo_retorno    = p.tempo_conclusao - p.tempo_chegada
            p.tempo_espera     = p.tempo_retorno - p.burst_total
            concluidos.append(p)

        self._calcular_metricas(concluidos, fs, met)
        return met

    def _calcular_metricas(self, procs, fs, met):
        EscalonadorFCFS._calcular_metricas(self, procs, fs, met)


class EscalonadorRoundRobin(Escalonador):
    """Round Robin preemptivo com quantum configurável."""

    def __init__(self, quantum: int = 4):
        super().__init__(f"Round Robin (Q={quantum})", quantum)

    def executar(self, processos: List[Processo]) -> MetricasSistema:
        procs      = copy.deepcopy(processos)
        cache      = CacheArquivos()
        fs         = SistemaArquivos(cache)
        met        = MetricasSistema(self.nome)
        fila       = []
        tempo      = 0
        concluidos = []

        restantes  = sorted(procs, key=lambda p: p.tempo_chegada)

        # insere primeiro lote
        chegando = [p for p in restantes if p.tempo_chegada == 0]
        for p in chegando:
            restantes.remove(p)
            fila.append(p)

        while fila or restantes:
            if not fila:
                tempo = restantes[0].tempo_chegada
                chegando = [p for p in restantes if p.tempo_chegada <= tempo]
                for p in chegando:
                    restantes.remove(p)
                    fila.append(p)

            p = fila.pop(0)

            if p.tempo_inicio == -1:
                p.tempo_inicio = tempo

            executa = min(self.quantum, p.burst_restante)

            # I/O que ocorre dentro deste quantum
            custo_io = 0
            for req in p.requisicoes:
                if tempo <= req.tempo_chegada < tempo + executa:
                    custo_io += fs.processar(req, req.tempo_chegada)
                    met.log_eventos.append(
                        f"[t={req.tempo_chegada}] {p.nome}: {req.operacao.value} "
                        f"bloco {req.bloco_disco}"
                    )

            tempo += executa + custo_io
            p.burst_restante -= executa

            # novos processos que chegaram durante este quantum
            chegando = [q for q in restantes if q.tempo_chegada <= tempo]
            for q in chegando:
                restantes.remove(q)
                fila.append(q)

            if p.burst_restante > 0:
                fila.append(p)
            else:
                p.tempo_conclusao = tempo
                p.tempo_retorno   = p.tempo_conclusao - p.tempo_chegada
                p.tempo_espera    = p.tempo_retorno - p.burst_total
                concluidos.append(p)

        EscalonadorFCFS._calcular_metricas(self, concluidos, fs, met)
        return met


class EscalonadorPrioridade(Escalonador):
    """Escalonamento por Prioridade — preemptivo."""

    def __init__(self):
        super().__init__("Prioridade (Preemptivo)")

    def executar(self, processos: List[Processo]) -> MetricasSistema:
        procs      = copy.deepcopy(processos)
        cache      = CacheArquivos()
        fs         = SistemaArquivos(cache)
        met        = MetricasSistema(self.nome)
        tempo      = 0
        concluidos = []
        fila       = []
        restantes  = sorted(procs, key=lambda p: p.tempo_chegada)

        while restantes or fila:
            chegando = [p for p in restantes if p.tempo_chegada <= tempo]
            for p in chegando:
                restantes.remove(p)
                fila.append(p)

            if not fila:
                tempo = restantes[0].tempo_chegada
                continue

            fila.sort(key=lambda p: p.prioridade)
            p = fila[0]

            if p.tempo_inicio == -1:
                p.tempo_inicio = tempo

            # executa 1 unidade (granularidade de preempção)
            custo_io = 0
            for req in p.requisicoes:
                if req.tempo_chegada == tempo:
                    custo_io += fs.processar(req, tempo)
                    met.log_eventos.append(
                        f"[t={tempo}] {p.nome}: {req.operacao.value} bloco {req.bloco_disco}"
                    )

            tempo += 1 + custo_io
            p.burst_restante -= 1

            chegando = [q for q in restantes if q.tempo_chegada <= tempo]
            for q in chegando:
                restantes.remove(q)
                fila.append(q)

            if p.burst_restante <= 0:
                fila.remove(p)
                p.tempo_conclusao = tempo
                p.tempo_retorno   = p.tempo_conclusao - p.tempo_chegada
                p.tempo_espera    = p.tempo_retorno - p.burst_total
                concluidos.append(p)

        # detecta starvation nos de baixa prioridade
        EscalonadorFCFS._calcular_metricas(self, concluidos, fs, met)
        para_starvation = [p for p in concluidos if p.tempo_espera > self.LIMIAR_STARVATION]
        met.starvation_detectado     = len(para_starvation) > 0
        met.processos_com_starvation = [p.pid for p in para_starvation]
        return met


# ──────────────────────────────────────────────
# Gerador de Carga de Trabalho
# ──────────────────────────────────────────────

def gerar_carga(n_processos: int = 8, seed: int = 42) -> List[Processo]:
    """
    Gera uma lista de processos com perfis variados:
    CPU-bound, I/O-bound e mistos.
    """
    random.seed(seed)
    processos = []
    perfis = ["CPU-bound", "I/O-bound", "Misto"]

    for i in range(n_processos):
        perfil  = perfis[i % len(perfis)]
        chegada = random.randint(0, 10)
        burst   = (random.randint(20, 40) if perfil == "CPU-bound"
                   else random.randint(5, 15)  if perfil == "I/O-bound"
                   else random.randint(10, 25))
        prio    = random.randint(1, 5)

        # gera requisições de arquivo
        n_reqs = (1 if perfil == "CPU-bound"
                  else random.randint(4, 6) if perfil == "I/O-bound"
                  else random.randint(2, 4))

        reqs = []
        for j in range(n_reqs):
            op     = random.choice(list(OperacaoArquivo))
            bloco  = random.randint(0, 499)
            tam_kb = round(random.uniform(1.0, 64.0), 1)
            t_req  = chegada + random.randint(0, burst)
            reqs.append(RequisicaoArquivo(i, op, bloco, tam_kb, t_req))

        p = Processo(
            pid=i,
            nome=f"P{i}({perfil[:3]})",
            burst_total=burst,
            prioridade=prio,
            tempo_chegada=chegada,
            requisicoes=reqs,
        )
        processos.append(p)

    return processos


# ──────────────────────────────────────────────
# Runner principal
# ──────────────────────────────────────────────

def executar_simulacao(n_processos: int = 8,
                       quantum_rr: int = 4,
                       seed: int = 42) -> List[MetricasSistema]:

    processos = gerar_carga(n_processos, seed)

    escalonadores = [
        EscalonadorFCFS(),
        EscalonadorSJF(),
        EscalonadorRoundRobin(quantum=quantum_rr),
        EscalonadorPrioridade(),
    ]

    resultados = []
    for esc in escalonadores:
        met = esc.executar(processos)
        resultados.append(met)

    return resultados


def imprimir_tabela(resultados: List[MetricasSistema]):
    """Imprime tabela comparativa no terminal."""
    print("\n" + "=" * 90)
    print(f"{'ALGORITMO':<28} {'T.Esp':>7} {'T.Ret':>7} {'Thrpt':>7} "
          f"{'Lat.IO':>8} {'IO Ops':>7} {'Thrpt.IO':>9} {'Cache%':>7} {'Starv':>6}")
    print("=" * 90)
    for m in resultados:
        starv = "SIM" if m.starvation_detectado else "NÃO"
        print(
            f"{m.algoritmo:<28} "
            f"{m.tempo_medio_espera:>7.2f} "
            f"{m.tempo_medio_retorno:>7.2f} "
            f"{m.throughput:>7.4f} "
            f"{m.latencia_media_io:>8.2f} "
            f"{m.total_io_operacoes:>7} "
            f"{m.throughput_io_kb:>9.2f} "
            f"{m.taxa_cache_hit*100:>7.1f} "
            f"{starv:>6}"
        )
    print("=" * 90)
    print("Legenda: T.Esp=Tempo médio de espera | T.Ret=Tempo médio de retorno |"
          " Thrpt=Throughput (proc/t)")
    print("         Lat.IO=Latência média de I/O | Thrpt.IO=Throughput de I/O (KB/t)"
          " | Cache%=Taxa de cache hit\n")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Simulação: Escalonamento de CPU × Sistema de Arquivos          ║")
    print("║  IFCE Maracanaú — Sistemas Operacionais 2026.1                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    resultados = executar_simulacao(n_processos=8, quantum_rr=4, seed=42)
    imprimir_tabela(resultados)