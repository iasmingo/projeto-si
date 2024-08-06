import connection as con
import random as rd
import numpy as np

# Inicializa a conexão
cn = con.connect(2037)

# Função para recuperar o progresso da Q-table
def recupera_tabela():
    tabela = []
    try:
        with open("resultado.txt", "r") as file:
            linhas = file.readlines()
        for linha in linhas:
            tabela.append([float(valor) for valor in linha.split()])
        # Garantir que a tabela tenha o tamanho correto
        while len(tabela) < 96:
            tabela.append([0.0, 0.0, 0.0])
    except IOError:
        tabela = [[0.0, 0.0, 0.0] for _ in range(96)]
    return tabela

# Função para salvar a Q-table
def salva_tabela(tabela):
    with open("resultado.txt", "w") as file:
        for linha in tabela:
            file.write(" ".join(map(str, linha)) + "\n")

# Função de atualização Q-learning
def atualiza_q(valor, alfa, gama, recompensa, maximo_futuro):
    return (1 - alfa) * valor + alfa * (recompensa + gama * maximo_futuro)

# Inicializa a Q-table
q_table = recupera_tabela()

# Parâmetros
alfa = 0.7  # Taxa de aprendizado
gama = 0.95  # Fator de desconto
epsilon = 0.1  # Taxa de exploração
episodios = 3  # Número de episódios

# Ações possíveis
acoes = ['jump', 'left', 'right']

# Função para decodificar o estado
def decodifica_estado(estado):
    plataforma = int(estado[2:7], 2)
    direcao = int(estado[7:9], 2)
    return plataforma, direcao

# Loop principal
for episodio in range(episodios):
    print(f"Episódio {episodio + 1}/{episodios}")

    estado, recompensa = con.get_state_reward(cn, "")
    estado_atual = (int(estado[2:7], 2) * 4) + (int(estado[7:9], 2) % 4)

    while True:
        print('==============================')

        # Política epsilon-greedy
        if rd.random() < epsilon:
            acao = rd.randint(0, 2)  # Exploração
        else:
            acao = np.argmax(q_table[estado_atual])  # Exploração

        estado, recompensa = con.get_state_reward(cn, acoes[acao])
        print(f'Estado: {estado} | Recompensa: {recompensa}')

        if recompensa < -100:
            alfa = 0.05

        plataforma, direcao = decodifica_estado(estado)
        estado_int = (plataforma * 4) + (direcao % 4)
        max_q_valor = max(q_table[estado_int])
        q_atual = atualiza_q(q_table[estado_atual][acao], alfa, gama, recompensa, max_q_valor)

        q_table[estado_atual][acao] = q_atual
        salva_tabela(q_table)

        estado_atual = estado_int

        if recompensa == -1:
            break

# Salva a Q-table final
salva_tabela(q_table)
