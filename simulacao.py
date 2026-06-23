"""
Simulação: Impacto da Política de Escalonamento de CPU
       sobre o Comportamento do Sistema de Arquivos
"""

import math
import random
import copy
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ══════════════════════════════════════════════════════════════════════
# Modelos de dados
# ══════════════════════════════════════════════════════════════════════

class TipoOperacao(Enum):
    LEITURA = "Leitura"
    ESCRITA = "Escrita"


class TipoFase(Enum):
    CPU = "CPU"
    IO = "IO"


@dataclass
class Fase:
    """Um trecho da vida de um processo: ou uma rajada de CPU, ou uma
    operação de E/S sobre um arquivo específico."""
    tipo: TipoFase
    duracao: int = 0                       # usado em fases de CPU
    arquivo: Optional[str] = None          # usado em fases de E/S
    operacao: Optional[TipoOperacao] = None
    tamanho_kb: float = 0.0
    offset_logico: int = 0                 # bloco lógico inicial (usado em LEITURA)


@dataclass
class Processo:
    """Processo com rajadas de CPU e requisições de E/S intercaladas.

    Diferente da versão anterior, as requisições de E/S não são uma lista
    solta com timestamps pré-definidos: elas estão embutidas em `fases`,
    e só passam a existir como `RequisicaoArquivo` no instante em que o
    motor de simulação efetivamente as dispara (ver `_motor_simulacao`).
    """
    pid: int
    nome: str
    perfil: str
    prioridade: int
    tempo_chegada: int
    fases: List[Fase]

    # ---- estado mutável, controlado pelo motor de simulação ----
    indice_fase: int = field(default=0, init=False)
    restante_fase: int = field(default=0, init=False)
    tempo_inicio: int = field(default=-1, init=False)
    tempo_conclusao: int = field(default=0, init=False)
    tempo_espera: int = field(default=0, init=False)

    def __post_init__(self):
        if self.fases and self.fases[0].tipo == TipoFase.CPU:
            self.restante_fase = self.fases[0].duracao

    @property
    def fase_atual(self) -> Optional[Fase]:
        if self.indice_fase >= len(self.fases):
            return None
        return self.fases[self.indice_fase]

    @property
    def concluido(self) -> bool:
        return self.indice_fase >= len(self.fases)

    @property
    def tempo_retorno(self) -> int:
        return self.tempo_conclusao - self.tempo_chegada

    @property
    def tempo_resposta(self) -> int:
        return (self.tempo_inicio - self.tempo_chegada) if self.tempo_inicio >= 0 else 0

    def burst_total(self) -> int:
        return sum(f.duracao for f in self.fases if f.tipo == TipoFase.CPU)

    def proxima_rajada_cpu(self) -> float:
        """Usado pelo SJF: duração restante da fase de CPU atual, ou
        infinito se o processo não está numa fase de CPU agora."""
        fase = self.fase_atual
        if fase is not None and fase.tipo == TipoFase.CPU:
            return self.restante_fase
        return float("inf")

    def avancar_fase(self):
        self.indice_fase += 1
        fase = self.fase_atual
        if fase is not None and fase.tipo == TipoFase.CPU:
            self.restante_fase = fase.duracao


@dataclass
class RequisicaoArquivo:
    """Representa uma operação de E/S em andamento ou concluída."""
    pid: int
    nome_processo: str
    arquivo: str
    operacao: TipoOperacao
    tamanho_kb: float
    tempo_chegada: int             # instante em que entrou na fila do disco
    offset_logico: int = 0
    tempo_inicio_io: int = 0
    tempo_fim_io: int = 0
    _custo_restante: int = field(default=0, repr=False)

    @property
    def latencia_io(self) -> int:
        """Latência TOTAL de E/S: tempo na fila + tempo de serviço no disco.
        latencia_io = tempo_fim_io - tempo_chegada
                    = (tempo_inicio_io - tempo_chegada)  [espera na fila]
                    + (tempo_fim_io - tempo_inicio_io)   [serviço no disco]
        """
        return self.tempo_fim_io - self.tempo_chegada

    @property
    def espera_fila_io(self) -> int:
        """Componente de fila: quanto tempo a requisição ficou parada
        esperando o disco ficar livre antes de começar a ser atendida."""
        return self.tempo_inicio_io - self.tempo_chegada

    @property
    def tempo_servico_io(self) -> int:
        """Componente de serviço: tempo realmente gasto pelo disco atendendo
        a requisição (seek + rotação + transferência), sem a fila."""
        return self.tempo_fim_io - self.tempo_inicio_io


@dataclass
class MetricasSistema:
    """Agrega os resultados de uma execução completa de um escalonador."""
    algoritmo: str
    tempo_medio_espera: float = 0.0
    tempo_max_espera: float = 0.0           # espera do processo mais penalizado (usado p/ starvation)
    tempo_medio_retorno: float = 0.0
    tempo_medio_resposta: float = 0.0
    throughput: float = 0.0                 # processos / unidade de tempo (vazão global)
    eficiencia_retorno: float = 0.0         # 1 / tempo_medio_retorno — mais sensível à política
    latencia_media_io: float = 0.0
    espera_media_fila_io: float = 0.0       # componente de fila da latência de E/S
    servico_medio_io: float = 0.0           # componente de serviço da latência de E/S
    total_io_operacoes: int = 0
    throughput_io_kb: float = 0.0           # KB / unidade de tempo
    taxa_cache_hit: float = 0.0
    cache_hits: int = 0                     # bruto — para auditoria/depuração
    cache_misses: int = 0                   # bruto — para auditoria/depuração
    distancia_seek_total: int = 0
    media_extents_por_arquivo: float = 0.0  # indicador de fragmentação
    tempo_total_simulacao: int = 0
    trocas_contexto: int = 0                # número de trocas de contexto (dispatches + preempções)
    utilizacao_cpu: float = 0.0             # fração do tempo com CPU ocupada
    historico_fila_io: List[int] = field(default_factory=list)  # tamanho da fila a cada tick
    starvation_detectado: bool = False
    processos_com_starvation: List[int] = field(default_factory=list)
    limiar_starvation: int = 0              # limiar usado para detectar starvation (exposto à visualização)
    log_eventos: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Cache de blocos de arquivo (LRU)
# ══════════════════════════════════════════════════════════════════════

class CacheArquivos:
    """Cache LRU com capacidade fixa de blocos. A chave é (arquivo,
    índice lógico do bloco dentro do arquivo) — assim a localidade é
    medida por arquivo, não por número de bloco físico bruto."""

    def __init__(self, capacidade: int = 24):
        self.capacidade = capacidade
        self._ordem: "OrderedDict" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def acessar(self, chave: Tuple[str, int]) -> bool:
        if chave in self._ordem:
            self._ordem.move_to_end(chave)
            self.hits += 1
            return True
        self.misses += 1
        self._ordem[chave] = True
        if len(self._ordem) > self.capacidade:
            self._ordem.popitem(last=False)
        return False

    @property
    def taxa_hit(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ══════════════════════════════════════════════════════════════════════
# Disco virtual: alocação de blocos por extents (não-contígua)
# ══════════════════════════════════════════════════════════════════════

class SistemaDiscoVirtual:
    """Disco simples: vetor de blocos + alocação por extents.

    Quando processos diferentes escrevem de forma intercalada (o que é
    mais comum sob Round Robin do que sob FCFS, por exemplo), os blocos
    livres já tendem a estar entremeados, então cada arquivo acaba
    espalhado em mais extents — ou seja, mais fragmentado. É esse efeito
    que a métrica `media_extents_por_arquivo` captura.
    """

    SEEK_BASE = 2
    SEEK_POR_BLOCO = 1
    LATENCIA_ROTACAO = 3

    def __init__(self, num_blocos: int = 300):
        self.num_blocos = num_blocos
        self.blocos_livres = deque(range(num_blocos))
        self.alocacao: Dict[str, List[Tuple[int, int]]] = {}
        self.posicao_cabeca = 0
        self.distancia_seek_total = 0

    def pre_alocar(self, arquivo: str, tamanho_blocos: int):
        """Cria um arquivo já existente no disco (estado inicial),
        simulando um sistema de arquivos recém-formatado: contíguo."""
        blocos = [self.blocos_livres.popleft() for _ in range(tamanho_blocos)]
        self.alocacao[arquivo] = [(blocos[0], len(blocos))] if blocos else []

    def _alocar(self, n: int) -> List[int]:
        if len(self.blocos_livres) < n:
            # Disco "cheio": nesta simulação não modelamos ENOSPC, mas
            # também não podemos reciclar blocos que já pertencem a outro
            # arquivo (isso corromperia a alocação: dois arquivos
            # "donos" do mesmo bloco físico). Damos margem realocando
            # apenas blocos extras ao final do disco virtual.
            faltam = n - len(self.blocos_livres)
            novo_inicio = self.num_blocos
            self.blocos_livres.extend(range(novo_inicio, novo_inicio + faltam))
            self.num_blocos += faltam
        alocados = [self.blocos_livres.popleft() for _ in range(n)]
        return alocados

    def escrever(self, arquivo: str, n_blocos: int) -> List[int]:
        novos = self._alocar(n_blocos)
        extents = self.alocacao.setdefault(arquivo, [])
        mesclados: List[Tuple[int, int]] = []
        inicio_atual, tam_atual = None, 0
        for b in novos:
            if inicio_atual is not None and b == inicio_atual + tam_atual:
                tam_atual += 1
            else:
                if inicio_atual is not None:
                    mesclados.append((inicio_atual, tam_atual))
                inicio_atual, tam_atual = b, 1
        if inicio_atual is not None:
            mesclados.append((inicio_atual, tam_atual))
        if extents and mesclados and extents[-1][0] + extents[-1][1] == mesclados[0][0]:
            ult_inicio, ult_tam = extents.pop()
            mesclados[0] = (ult_inicio, ult_tam + mesclados[0][1])
        extents.extend(mesclados)
        return novos

    def blocos_do_arquivo(self, arquivo: str, quantidade: Optional[int] = None,
                          offset: int = 0) -> List[int]:
        extents = self.alocacao.get(arquivo, [])
        blocos: List[int] = []
        for inicio, tam in extents:
            blocos.extend(range(inicio, inicio + tam))
        if offset:
            blocos = blocos[offset:]
        if quantidade is not None:
            blocos = blocos[:quantidade]
        return blocos

    def num_extents(self, arquivo: str) -> int:
        return len(self.alocacao.get(arquivo, []))

    def custo_seek(self, bloco_destino: int) -> int:
        distancia = abs(bloco_destino - self.posicao_cabeca)
        self.posicao_cabeca = bloco_destino
        self.distancia_seek_total += distancia
        # custo em tempo modelado como proporcional à RAIZ da distância
        # (aproxima a aceleração/desaceleração real do braço do disco —
        # um modelo puramente linear faria a E/S dominar irrealisticamente
        # a escala de tempo frente às rajadas de CPU). A distância "crua"
        # continua sendo acumulada acima, para as métricas de localidade.
        return math.isqrt(distancia) * self.SEEK_POR_BLOCO

    def media_extents_por_arquivo(self) -> float:
        if not self.alocacao:
            return 0.0
        return sum(len(v) for v in self.alocacao.values()) / len(self.alocacao)


# ══════════════════════════════════════════════════════════════════════
# Sistema de Arquivos: orquestra fila de E/S + disco + cache
# ══════════════════════════════════════════════════════════════════════

BLOCO_KB = 4.0
CUSTO_COMMIT_ESCRITA = 2


class SistemaArquivos:
    """Modela o subsistema de E/S como um único "braço de disco" que
    atende uma requisição por vez (fila FCFS no disco — o foco do
    trabalho é o efeito do escalonador de CPU, não do escalonador de
    disco). `tick()` avança 1 unidade de tempo de atendimento."""

    def __init__(self, num_blocos: int = 300, capacidade_cache: int = 8):
        self.disco = SistemaDiscoVirtual(num_blocos=num_blocos)
        self.cache = CacheArquivos(capacidade=capacidade_cache)
        self.fila: List[RequisicaoArquivo] = []
        self.em_atendimento: Optional[RequisicaoArquivo] = None
        self.concluidas: List[RequisicaoArquivo] = []
        self.kb_transferidos: float = 0.0
        self.total_requisicoes: int = 0

    def preparar_arquivo(self, arquivo: str, tamanho_blocos: int):
        self.disco.pre_alocar(arquivo, tamanho_blocos)

    def submeter(self, req: RequisicaoArquivo):
        self.fila.append(req)
        self.total_requisicoes += 1

    @staticmethod
    def _blocos_para_kb(kb: float) -> int:
        return max(1, math.ceil(kb / BLOCO_KB))

    def tick(self, agora: int) -> List[RequisicaoArquivo]:
        concluidas_agora: List[RequisicaoArquivo] = []

        if self.em_atendimento is None and self.fila:
            req = self.fila.pop(0)
            req.tempo_inicio_io = agora
            n_blocos = self._blocos_para_kb(req.tamanho_kb)

            if req.operacao == TipoOperacao.ESCRITA:
                blocos = self.disco.escrever(req.arquivo, n_blocos)
                seek = self.disco.custo_seek(blocos[0])
                custo = (self.disco.SEEK_BASE + seek + self.disco.LATENCIA_ROTACAO
                         + n_blocos + CUSTO_COMMIT_ESCRITA)
            else:  # LEITURA
                blocos = self.disco.blocos_do_arquivo(req.arquivo, n_blocos, offset=req.offset_logico)
                seek = self.disco.custo_seek(blocos[0]) if blocos else 0
                transferencia = 0
                for posicao, bloco_fisico in enumerate(blocos):
                    indice_logico = req.offset_logico + posicao
                    acertou = self.cache.acessar((req.arquivo, indice_logico))
                    if not acertou:
                        transferencia += 1
                custo = self.disco.SEEK_BASE + seek + self.disco.LATENCIA_ROTACAO + transferencia

            self.kb_transferidos += req.tamanho_kb
            req._custo_restante = max(custo, 1)
            self.em_atendimento = req

        if self.em_atendimento is not None:
            self.em_atendimento._custo_restante -= 1
            if self.em_atendimento._custo_restante <= 0:
                self.em_atendimento.tempo_fim_io = agora + 1
                self.concluidas.append(self.em_atendimento)
                concluidas_agora.append(self.em_atendimento)
                self.em_atendimento = None

        return concluidas_agora

    def resumo_metricas(self) -> dict:
        if self.concluidas:
            latencias = [r.latencia_io for r in self.concluidas]
            esperas_fila = [r.espera_fila_io for r in self.concluidas]
            servicos = [r.tempo_servico_io for r in self.concluidas]
            latencia_media = sum(latencias) / len(latencias)
            espera_media_fila = sum(esperas_fila) / len(esperas_fila)
            servico_medio = sum(servicos) / len(servicos)
        else:
            latencia_media = 0.0
            espera_media_fila = 0.0
            servico_medio = 0.0
        return {
            "latencia_media_io": latencia_media,
            "espera_media_fila_io": espera_media_fila,
            "servico_medio_io": servico_medio,
            "total_requisicoes": self.total_requisicoes,
            "kb_transferidos": self.kb_transferidos,
            "taxa_cache_hit": self.cache.taxa_hit,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "distancia_seek_total": self.disco.distancia_seek_total,
            "media_extents_por_arquivo": self.disco.media_extents_por_arquivo(),
        }


# ══════════════════════════════════════════════════════════════════════
# Motor de simulação único (por ticks), reutilizado por TODOS os
# escalonadores — é isso que garante uma comparação justa entre eles.
# ══════════════════════════════════════════════════════════════════════

def _motor_simulacao(processos: List[Processo], escalonador: "Escalonador",
                      fs: SistemaArquivos, max_ticks: int = 200_000):
    nao_chegados = sorted(processos, key=lambda p: p.tempo_chegada)
    prontos: List[Processo] = []
    executando: Optional[Processo] = None
    bloqueados: Dict[int, Processo] = {}
    concluidos: List[Processo] = []
    eventos: List[str] = []
    ticks_quantum = 0
    cpu_ociosa = 0
    trocas_contexto = 0
    historico_fila_io: List[int] = []
    agora = 0
    is_rr = isinstance(escalonador, EscalonadorRoundRobin)

    while len(concluidos) < len(processos) and agora < max_ticks:
        # 1) chegadas de novos processos
        while nao_chegados and nao_chegados[0].tempo_chegada <= agora:
            prontos.append(nao_chegados.pop(0))

        # 2) avança o subsistema de E/S; quem termina volta para "prontos"
        for req in fs.tick(agora):
            proc = bloqueados.pop(req.pid)
            eventos.append(
                f"[t={req.tempo_fim_io}] {proc.nome}: {req.operacao.value} em "
                f"'{req.arquivo}' concluída (latência={req.latencia_io})"
            )
            proc.avancar_fase()
            if proc.concluido:
                proc.tempo_conclusao = agora + 1
                concluidos.append(proc)
            else:
                prontos.append(proc)

        # 2b) registra tamanho da fila de E/S neste tick
        historico_fila_io.append(len(fs.fila) + (1 if fs.em_atendimento else 0))

        # 3) escolhe quem ocupa a CPU neste tick.
        # O escalonador recebe a lista completa de prontos (sem o processo
        # em execução) E o processo em execução separado, para que políticas
        # não-preemptivas (FCFS, SJF) possam simplesmente retornar `executando`
        # e políticas preemptivas (SRTF, Prioridade) possam comparar os dois.
        escolhido = escalonador.selecionar(list(prontos), executando, agora)

        if escolhido is None:
            cpu_ociosa += 1
            agora += 1
            continue

        if escolhido is not executando:
            # Houve troca de contexto: qualquer mudança de processo na CPU
            # (preempção, dispatch após término/bloqueio, ou primeiro dispatch).
            # Em sistemas operacionais reais, TODA substituição de processo na
            # CPU envolve salvar/restaurar contexto — inclusive FCFS e SJF
            # trocam contexto cada vez que um processo termina e o próximo entra.
            trocas_contexto += 1
            if executando is not None:
                # Preempção: devolve o processo atual para a fila de prontos
                prontos.append(executando)
            if escolhido in prontos:
                prontos.remove(escolhido)
            executando = escolhido
            if executando.tempo_inicio == -1:
                executando.tempo_inicio = agora
            ticks_quantum = 0

        for p in prontos:
            p.tempo_espera += 1

        # 4) executa 1 unidade de tempo
        proc = executando
        proc.restante_fase -= 1
        ticks_quantum += 1

        fase_terminou = proc.restante_fase <= 0
        quantum_terminou = is_rr and ticks_quantum >= escalonador.quantum

        if fase_terminou:
            proc.avancar_fase()
            if proc.concluido:
                proc.tempo_conclusao = agora + 1
                concluidos.append(proc)
                executando = None
            else:
                fase = proc.fase_atual
                if fase.tipo == TipoFase.CPU:
                    pass  # fases de CPU consecutivas: continua executando
                else:
                    req = RequisicaoArquivo(
                        pid=proc.pid, nome_processo=proc.nome, arquivo=fase.arquivo,
                        operacao=fase.operacao, tamanho_kb=fase.tamanho_kb,
                        tempo_chegada=agora + 1, offset_logico=fase.offset_logico,
                    )
                    fs.submeter(req)
                    eventos.append(
                        f"[t={agora + 1}] {proc.nome}: solicita {fase.operacao.value} "
                        f"em '{fase.arquivo}' ({fase.tamanho_kb:.1f} KB)"
                    )
                    bloqueados[proc.pid] = proc
                    executando = None
            ticks_quantum = 0
        elif quantum_terminou:
            # Quantum esgotado: devolve à fila de prontos. A troca de
            # contexto (se houver) é contada no próximo tick, no bloco
            # `escolhido is not executando` — contar aqui TAMBÉM duplica
            # a troca quando o próprio processo é reescolhido em seguida
            # (fila vazia ou só ele pronto), o que inflava o RR ~2x.
            prontos.append(proc)
            executando = None
            ticks_quantum = 0

        agora += 1

    return concluidos, cpu_ociosa, trocas_contexto, historico_fila_io, agora, eventos


# ══════════════════════════════════════════════════════════════════════
# Escalonadores
# ══════════════════════════════════════════════════════════════════════

class Escalonador:
    """Classe base. Cada subclasse só precisa decidir QUEM roda agora
    (`selecionar`); o motor de simulação (`_motor_simulacao`) é o mesmo
    para todas, então a comparação entre políticas é justa."""

    # Limiar de starvation: baseado no burst médio da carga (não no número de
    # processos). Um processo está em starvation quando sua espera acumulada
    # supera 2× o burst médio esperado da carga — ou seja, ele deveria ter
    # sido executado pelo menos duas vezes completas, mas ainda está esperando.
    # Burst médio ≈ 50 ticks (média ponderada entre CPU-bound ~85, I/O-bound ~6,
    # Misto ~27, com proporção 1/3 cada). 2× burst_medio ≈ 100 ticks.
    # Esse limiar é FIXO e INDEPENDENTE do número de processos, assim como
    # na definição clássica de starvation em literatura de SO.
    LIMIAR_STARVATION = 100  # ticks — fixo, baseado em 2× burst médio da carga

    def __init__(self, nome: str):
        self.nome = nome
        self._limiar_starvation = self.LIMIAR_STARVATION

    def selecionar(self, prontos: List[Processo], executando: Optional[Processo],
                   agora: int) -> Optional[Processo]:
        raise NotImplementedError

    def executar(self, processos: List[Processo], fs: SistemaArquivos) -> MetricasSistema:
        procs = copy.deepcopy(processos)
        concluidos, cpu_ociosa, trocas_contexto, historico_fila_io, makespan, eventos = (
            _motor_simulacao(procs, self, fs)
        )
        return self._calcular_metricas(concluidos, fs, makespan, eventos,
                                        cpu_ociosa, trocas_contexto, historico_fila_io)

    def _calcular_metricas(self, procs: List[Processo], fs: SistemaArquivos,
                            makespan: int, eventos: List[str],
                            cpu_ociosa: int = 0, trocas_contexto: int = 0,
                            historico_fila_io: Optional[List[int]] = None) -> MetricasSistema:
        met = MetricasSistema(self.nome)
        n = len(procs)
        met.tempo_total_simulacao = makespan
        met.tempo_medio_espera = sum(p.tempo_espera for p in procs) / n if n else 0.0
        met.tempo_max_espera = max((p.tempo_espera for p in procs), default=0.0)
        met.tempo_medio_retorno = sum(p.tempo_retorno for p in procs) / n if n else 0.0
        met.tempo_medio_resposta = sum(p.tempo_resposta for p in procs) / n if n else 0.0
        met.throughput = n / makespan if makespan else 0.0
        # Throughput (n/makespan) é dominado pelo makespan, que por sua vez é
        # dominado pela soma fixa de trabalho (bursts de CPU + serviço de E/S)
        # — isso é CONCEITUALMENTE correto, mas pouco discriminativo entre
        # políticas quando todas terminam numa janela de tempo parecida.
        # eficiencia_retorno é mais sensível à política, pois captura o efeito
        # médio sentido por processo (tempo de retorno), não só o instante do
        # último processo a terminar.
        met.eficiencia_retorno = (1.0 / met.tempo_medio_retorno) if met.tempo_medio_retorno else 0.0
        met.trocas_contexto = trocas_contexto
        met.utilizacao_cpu = (makespan - cpu_ociosa) / makespan if makespan else 0.0
        met.historico_fila_io = historico_fila_io or []

        resumo = fs.resumo_metricas()
        met.latencia_media_io = resumo["latencia_media_io"]
        met.espera_media_fila_io = resumo["espera_media_fila_io"]
        met.servico_medio_io = resumo["servico_medio_io"]
        met.total_io_operacoes = resumo["total_requisicoes"]
        met.throughput_io_kb = resumo["kb_transferidos"] / makespan if makespan else 0.0
        met.taxa_cache_hit = resumo["taxa_cache_hit"]
        met.cache_hits = resumo["cache_hits"]
        met.cache_misses = resumo["cache_misses"]
        met.distancia_seek_total = resumo["distancia_seek_total"]
        met.media_extents_por_arquivo = resumo["media_extents_por_arquivo"]

        com_starvation = [p.pid for p in procs if p.tempo_espera > self._limiar_starvation]
        met.starvation_detectado = len(com_starvation) > 0
        met.processos_com_starvation = com_starvation
        met.limiar_starvation = self._limiar_starvation  # expõe à visualização
        met.log_eventos = eventos
        return met


class EscalonadorFCFS(Escalonador):
    """First-Come, First-Served — não-preemptivo."""

    def __init__(self):
        super().__init__("FCFS")

    def selecionar(self, prontos, executando, agora):
        if executando is not None:
            return executando
        return prontos[0] if prontos else None


class EscalonadorSJF(Escalonador):
    """Shortest Job First — não-preemptivo, decide pela próxima rajada de CPU."""

    def __init__(self):
        super().__init__("SJF")

    def selecionar(self, prontos, executando, agora):
        if executando is not None:
            return executando
        if not prontos:
            return None
        return min(prontos, key=lambda p: (p.proxima_rajada_cpu(), p.tempo_chegada, p.pid))


class EscalonadorSRTF(Escalonador):
    """Shortest Remaining Time First — variante preemptiva do SJF.

    A cada tick avalia se existe um processo em 'prontos' com MENOS tempo
    restante de CPU do que o processo atualmente em execução. Se sim,
    preempta o atual. Isso fragmenta mais as rajadas de E/S do que o SJF
    não-preemptivo — o que é interessante para comparar o impacto no cache.
    """

    def __init__(self):
        super().__init__("SRTF (SJF Preemptivo)")

    def selecionar(self, prontos, executando, agora):
        candidatos = list(prontos)
        if executando is not None:
            candidatos.append(executando)
        if not candidatos:
            return None
        return min(candidatos, key=lambda p: (p.proxima_rajada_cpu(), p.tempo_chegada, p.pid))


class EscalonadorRoundRobin(Escalonador):
    """Round Robin preemptivo com quantum configurável."""

    def __init__(self, quantum: int = 4):
        super().__init__(f"Round Robin (Q={quantum})")
        self.quantum = quantum

    def selecionar(self, prontos, executando, agora):
        if executando is not None:
            return executando
        return prontos[0] if prontos else None


class EscalonadorPrioridade(Escalonador):
    """Escalonamento por prioridade com envelhecimento (aging).

    A prioridade efetiva de um processo decresce 1 nível a cada
    INTERVALO_AGING ticks de espera acumulada. Isso evita starvation:
    processos de baixa prioridade que esperam muito ficam progressivamente
    mais competitivos, sem alterar a prioridade original do processo.

    Preemptivo apenas por NÍVEL: um processo pronto só toma a CPU do
    processo em execução se sua prioridade efetiva for ESTRITAMENTE
    melhor (menor). Em caso de empate de nível, o processo em execução
    continua executando.

    BUG CORRIGIDO (regressão de uma "correção" anterior):
    Desempatar globalmente por `-tempo_espera` (quem esperou mais vence)
    e aplicar esse desempate também CONTRA o processo em execução causava
    preempção espúria a cada tick: como `tempo_espera` do processo em
    execução fica parado (só quem está em `prontos` acumula espera) e o
    dos processos prontos cresce 1 a cada tick, o desempate quase sempre
    favorecia algum processo pronto sobre o que estava rodando, mesmo
    com prioridade efetiva IDÊNTICA — gerando troca de contexto a cada
    1-2 ticks (203 de 204 trocas no cenário padrão eram espúrias, sem
    nenhuma mudança real de urgência).
    Correção: a comparação contra o processo em execução usa só a
    prioridade efetiva (sem o desempate por espera). O desempate por
    maior tempo de espera é usado apenas para decidir, ENTRE os
    candidatos prontos empatados em nível, qual deles entra quando a
    CPU está livre — preservando a ideia original de não travar em
    FCFS rígido dentro do grupo já saturado em prioridade 1.
    """

    INTERVALO_AGING = 10   # a cada 10 ticks de espera, sobe 1 nível de prioridade

    def __init__(self):
        super().__init__("Prioridade (Aging)")

    def _prioridade_efetiva(self, p: "Processo") -> int:
        """Quanto mais o processo esperou, menor (= mais urgente) a prioridade efetiva."""
        bonus = p.tempo_espera // self.INTERVALO_AGING
        return max(1, p.prioridade - bonus)

    def selecionar(self, prontos, executando, agora):
        if not prontos and executando is None:
            return None

        melhor_pronto = None
        if prontos:
            melhor_pronto = min(
                prontos,
                key=lambda p: (self._prioridade_efetiva(p), -p.tempo_espera,
                                p.tempo_chegada, p.pid),
            )

        if executando is None:
            return melhor_pronto

        if melhor_pronto is None:
            return executando

        # Só preempta por NÍVEL estritamente melhor — empate de prioridade
        # efetiva mantém o processo em execução (evita troca espúria).
        if self._prioridade_efetiva(melhor_pronto) < self._prioridade_efetiva(executando):
            return melhor_pronto
        return executando


# ══════════════════════════════════════════════════════════════════════
# Gerador de carga de trabalho
# ══════════════════════════════════════════════════════════════════════

ARQUIVOS_SISTEMA = ["dados.db", "log_sistema.txt", "config.cfg", "cache_app.tmp", "indice.idx"]


def _particionar(total: int, partes: int, rng: random.Random) -> List[int]:
    """Divide `total` em `partes` pedaços inteiros >= 1 que somam `total`."""
    if partes <= 1:
        return [total]
    if total <= partes:
        return [max(1, total - (partes - 1))] + [1] * (partes - 1)
    cortes = sorted(rng.sample(range(1, total), partes - 1))
    pedacos, anterior = [], 0
    for c in cortes + [total]:
        pedacos.append(c - anterior)
        anterior = c
    return pedacos


def gerar_carga(n_processos: int = 8, seed: int = 42) -> Tuple[List[Processo], Dict[str, int]]:
    """Gera processos com perfis variados (CPU-bound, I/O-bound, Misto) e
    a lista de arquivos que devem existir previamente no disco.

    Parâmetros calibrados para maximizar a diferença observável entre políticas:

    - CPU-bound: burst longo (50-120 ticks) → FCFS sofre efeito comboio real;
      políticas preemptivas (RR, SRTF) interrompem frequentemente e fragmentam
      mais a fila de E/S em comparação com FCFS/SJF.

    - I/O-bound: burst curto (3-10 ticks), muitas fases de E/S → a cadência de
      chegada de requisições varia bastante conforme o escalonador.

    - Acesso com localidade: 60% das leituras usam offset dentro de um pequeno
      conjunto FIXO de "blocos quentes" (não um intervalo contínuo qualquer)
      em 2 arquivos "quentes". Usar pontos de ancoragem fixos (em vez de um
      offset aleatório livre dentro de uma janela) é essencial: como cada
      leitura cobre uma janela de tamanho variável, duas leituras que apenas
      "começam dentro da mesma faixa" quase nunca pedem exatamente os MESMOS
      blocos lógicos — e é a igualdade exata de bloco que o cache LRU exige
      para registrar um hit. Ancorar em poucos pontos fixos garante reuso real.

    Retorna (processos, tamanhos_iniciais_dos_arquivos_em_blocos).
    """
    rng = random.Random(seed)
    tamanhos_iniciais = {a: rng.randint(30, 80) for a in ARQUIVOS_SISTEMA}

    # Dois arquivos "quentes": a maioria das leituras com localidade vai para eles
    arquivos_quentes = ARQUIVOS_SISTEMA[:2]   # dados.db, log_sistema.txt

    # Pontos de ancoragem FIXOS por arquivo quente: um pequeno conjunto de
    # offsets lógicos (não um intervalo contínuo) que processos diferentes
    # vão reutilizar literalmente — é isso que cria sobreposição de blocos
    # real e, portanto, hits de cache reais e comparáveis entre algoritmos.
    N_PONTOS_QUENTES = 2   # poucos pontos → mais reuso exato do MESMO bloco
    pontos_quentes: Dict[str, List[int]] = {
        a: [0, 2][:N_PONTOS_QUENTES] for a in arquivos_quentes
    }

    processos = []
    perfis = ["CPU-bound", "I/O-bound", "Misto"]

    for i in range(n_processos):
        perfil = perfis[i % len(perfis)]
        chegada = rng.randint(0, 10)

        # Bursts maiores → a política de CPU tem mais tempo para fazer diferença
        burst_total = (rng.randint(50, 120) if perfil == "CPU-bound"
                       else rng.randint(3, 10)  if perfil == "I/O-bound"
                       else rng.randint(15, 40))
        prioridade = rng.randint(1, 5)

        n_reqs = (1 if perfil == "CPU-bound"
                  else rng.randint(4, 7) if perfil == "I/O-bound"
                  else rng.randint(2, 4))
        n_reqs = min(n_reqs, max(1, burst_total - 1))

        pedacos_cpu = _particionar(burst_total, n_reqs + 1, rng)

        fases: List[Fase] = []
        for k, duracao in enumerate(pedacos_cpu):
            fases.append(Fase(tipo=TipoFase.CPU, duracao=duracao))
            if k < n_reqs:
                operacao = rng.choice([TipoOperacao.LEITURA, TipoOperacao.ESCRITA])

                # Localidade: 60% das leituras vão para arquivos quentes,
                # ancoradas num offset fixo reutilizado entre processos →
                # mesmos blocos lógicos absolutos, dando ao cache LRU chance
                # real de produzir hits (ver nota acima sobre ancoragem fixa).
                usar_localidade = (operacao == TipoOperacao.LEITURA
                                   and rng.random() < 0.80)
                if usar_localidade:
                    arquivo = rng.choice(arquivos_quentes)
                    tamanho_kb = round(rng.uniform(1.0, 12.0), 1)   # pequeno → poucos blocos
                    offset_logico = rng.choice(pontos_quentes[arquivo])
                else:
                    arquivo = rng.choice(ARQUIVOS_SISTEMA)
                    tamanho_kb = round(rng.uniform(1.0, 64.0), 1)
                    n_blocos_fase = max(1, math.ceil(tamanho_kb / BLOCO_KB))
                    offset_logico = 0
                    if operacao == TipoOperacao.LEITURA:
                        espaco = max(0, tamanhos_iniciais[arquivo] - n_blocos_fase)
                        offset_logico = rng.randint(0, espaco) if espaco > 0 else 0

                fases.append(Fase(tipo=TipoFase.IO, arquivo=arquivo, operacao=operacao,
                                   tamanho_kb=tamanho_kb, offset_logico=offset_logico))

        nome = f"P{i}({perfil[:3]})"
        processos.append(Processo(pid=i, nome=nome, perfil=perfil, prioridade=prioridade,
                                   tempo_chegada=chegada, fases=fases))

    return processos, tamanhos_iniciais


def _criar_sistema_arquivos(tamanhos_iniciais: Dict[str, int]) -> SistemaArquivos:
    # disco dimensionado em função dos arquivos reais (com margem para escritas
    # que crescem os arquivos durante a simulação) — evita distâncias de seek
    # artificialmente enormes que dominariam qualquer efeito do escalonador de CPU
    num_blocos = max(150, sum(tamanhos_iniciais.values()) * 3)
    fs = SistemaArquivos(num_blocos=num_blocos)
    for arquivo, tamanho in tamanhos_iniciais.items():
        fs.preparar_arquivo(arquivo, tamanho)
    return fs


# ══════════════════════════════════════════════════════════════════════
# Runner principal
# ══════════════════════════════════════════════════════════════════════

def executar_simulacao(n_processos: int = 8, quantum_rr: int = 4,
                        seed: int = 42) -> List[MetricasSistema]:
    """Gera UMA carga de trabalho (mesma para todos) e a executa sob os
    quatro escalonadores, cada um com seu próprio disco/cache zerados —
    só assim a comparação é justa: mesmos processos, mesmas requisições,
    mesmo estado inicial de disco para todos."""
    processos_modelo, tamanhos_iniciais = gerar_carga(n_processos, seed)

    escalonadores: List[Escalonador] = [
        EscalonadorFCFS(),
        EscalonadorSJF(),
        EscalonadorSRTF(),
        EscalonadorRoundRobin(quantum=quantum_rr),
        EscalonadorPrioridade(),
    ]

    resultados = []
    for esc in escalonadores:
        fs = _criar_sistema_arquivos(tamanhos_iniciais)
        met = esc.executar(processos_modelo, fs)
        resultados.append(met)
    return resultados


def imprimir_tabela(resultados: List[MetricasSistema]):
    """Imprime tabela comparativa no terminal."""
    print("\n" + "=" * 134)
    print(f"{'ALGORITMO':<26} {'T.Esp':>7} {'T.EspMax':>9} {'T.Ret':>7} {'T.Resp':>7} {'Thrpt':>7} {'CPU%':>6} {'Ctx':>5} "
          f"{'Lat.IO':>7} {'IO Ops':>7} {'Thrpt.IO':>9} {'Cache%':>7} {'Frag':>6} {'Starv':>6}")
    print("=" * 134)
    for m in resultados:
        # Critério EXPLÍCITO de starvation: por PROCESSO INDIVIDUAL, usando a
        # maior espera entre todos os processos (não a média). Um algoritmo
        # pode ter espera MÉDIA baixa e ainda conter um outlier em starvation.
        starv = "SIM" if m.starvation_detectado else "NÃO"
        print(
            f"{m.algoritmo:<26} "
            f"{m.tempo_medio_espera:>7.2f} "
            f"{m.tempo_max_espera:>9.2f} "
            f"{m.tempo_medio_retorno:>7.2f} "
            f"{m.tempo_medio_resposta:>7.2f} "
            f"{m.throughput:>7.4f} "
            f"{m.utilizacao_cpu * 100:>6.1f} "
            f"{m.trocas_contexto:>5} "
            f"{m.latencia_media_io:>7.2f} "
            f"{m.total_io_operacoes:>7} "
            f"{m.throughput_io_kb:>9.2f} "
            f"{m.taxa_cache_hit * 100:>7.1f} "
            f"{m.media_extents_por_arquivo:>6.2f} "
            f"{starv:>6}"
        )
    print("=" * 134)
    print("Legenda: T.Esp=Espera MÉDIA (CPU) | T.EspMax=Espera MÁXIMA entre processos (base da starvation)")
    print("         T.Ret=Retorno médio | T.Resp=Resposta média")
    print("         Thrpt=Throughput CPU | CPU%=Utilização da CPU | Ctx=Trocas de contexto")
    print("         Lat.IO=Latência média de E/S | Thrpt.IO=Throughput de E/S (KB/t)")
    print("         Cache%=Taxa de acerto de cache | Frag=Extents médios por arquivo")
    print(f"         Starv=SIM se ALGUM processo individual teve tempo_espera > "
          f"{resultados[0].limiar_starvation if resultados else 100} ticks (não é sobre a média)\n")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Simulação: Escalonamento de CPU × Sistema de Arquivos          ║")
    print("║  IFCE Maracanaú — Sistemas Operacionais 2026.1                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    resultados = executar_simulacao(n_processos=8, quantum_rr=4, seed=42)
    imprimir_tabela(resultados)