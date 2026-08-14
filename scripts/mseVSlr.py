import matplotlib.pyplot as plt
import numpy as np

taxas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.28]

mse_sem_reg = [15.910409, 0.073032, 19.365642, 10.700440, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
               float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

mse_com_reg = [16.011359, 0.086586, 19.395846, 11.616592, 56.086361, 12.649248, 5.624026, 4.281715, 23.569748,
               3.230303, 21.279254, 15.709958, 0.000381, float('nan')]
LIMITE = 100

def tratar(lista):
    return [x if (not np.isnan(x) and x < LIMITE) else LIMITE for x in lista]

sem_reg = tratar(mse_sem_reg)
com_reg = tratar(mse_com_reg)

plt.figure(figsize=(10,6))

# linhas
plt.plot(taxas, sem_reg, marker='o', linestyle='--', label='Sem regularização')
plt.plot(taxas, com_reg, marker='o', label='Com regularização')

# linha de explosão
plt.axhline(y=LIMITE, color='red', linestyle='--')
plt.text(taxas[0], LIMITE + 2, 'Divergência', color='red')

# marcar pontos de falha
for i, v in enumerate(mse_sem_reg):
    if np.isnan(v):
        plt.scatter(taxas[i], LIMITE, color='blue', marker='x', s=100)

for i, v in enumerate(mse_com_reg):
    if np.isnan(v):
        plt.scatter(taxas[i], LIMITE, color='orange', marker='x', s=100)

# eixo X discreto (como você queria)
plt.xticks(taxas)

plt.xlabel('Taxa de aprendizado')
plt.ylabel('MSE')
plt.title('Impacto da regularização na estabilidade do treinamento')

plt.legend()
plt.grid(True)

plt.show()