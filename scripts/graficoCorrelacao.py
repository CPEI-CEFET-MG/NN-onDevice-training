import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar dados
df_p = pd.read_csv('resultados_artigo004.csv')
df_l = pd.read_csv('resultados_artigo025.csv')

# 2. Criar a figura
plt.figure(figsize=(9, 9))

plt.rcParams['font.family'] = 'serif'

# 3. Plotar a linha ideal (Y = X)
# Definimos a faixa de interesse: 15 a 45 graus
faixa = np.linspace(15, 45, 100)
plt.plot(faixa, faixa, color='black', linestyle='--', label='Ideal (Fidelidade Total)', alpha=0.8, zorder=1)

# 4. Plotar as predições COM LIMITAÇÃO DE VISUALIZAÇÃO
# Usamos s=30 para pontos mais visíveis e alpha para ver a densidade
plt.scatter(df_p['Sensor'], df_p['PredPadrao'], 
            alpha=0.4, label=r'MLP Padrão ($\eta=0,04$)', color='#1f77b4', s=30, zorder=2)

plt.scatter(df_l['Sensor'], df_l['PredLipschitz'], 
            alpha=0.6, label=r'MLP Lipschitz ($\eta=0,25$)', color='#ff7f0e', s=30, zorder=3)

# 5. O SEGREDO: Limitar os eixos para a zona de operação real
# Isso ignora os "chutes" de -400 ou 1100 que estragam a escala
plt.xlim(18, 35) # Ajuste conforme a variação real da sua estufa
plt.ylim(18, 35)

# 6. Estética Profissional
plt.title('Zoom de Fidelidade: Operação em Regime Permanente', fontsize=14)
plt.xlabel('Temperatura Real do Sensor (°C)', fontsize=12)
plt.ylabel('Temperatura Predita pela Rede (°C)', fontsize=12)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

# Adiciona uma caixa de texto explicando os pontos fora do gráfico
plt.text(25, 19, "*Outliers de predição omitidos para clareza da escala", 
         fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('grafico_fidelidade.pdf', bbox_inches='tight')
plt.show()
