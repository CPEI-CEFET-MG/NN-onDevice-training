import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
ARQUIVO_CSV = 'resultados_artigo.csv'

def gerar_grafico_artigo():
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        
        # 1. Tratamento de Nomes e Tipos
        if 'Sensor' in df.columns:
            df.rename(columns={'Sensor': 'Real'}, inplace=True)
            
        for col in ['PredPadrao', 'PredLipschitz', 'Real']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Cálculo do MSE Acumulado (Running MSE)
        # Calculamos o erro quadrático e depois a média acumulada até aquele ponto
        df['SE_P'] = (df['Real'] - df['PredPadrao'])**2
        df['SE_L'] = (df['Real'] - df['PredLipschitz'])**2
        
        df['CumMSE_P'] = df['SE_P'].expanding().mean()
        df['CumMSE_L'] = df['SE_L'].expanding().mean()

        df[['CumMSE_P','CumMSE_L']].to_csv('mse_plot.csv', index=True)

        # --- ESTÉTICA DO GRÁFICO (Padrão IEEE/Nature) ---
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot das linhas
        ax.plot(df.index, df['CumMSE_P'], label='MLP Padrão (Divergente)', 
                color='#d62728', linewidth=1.8)
        ax.plot(df.index, df['CumMSE_L'], label='MLP Lipschitz (Estável)', 
                color='#1f77b4', linewidth=2)

        # Configuração da Escala Logarítmica
        ax.set_yscale('log')
        
        # 3. Identificação do Ponto de Colapso (NaN)
        idx_nan = df[df['PredPadrao'].isna()].index
        if not idx_nan.empty:
            ponto_morte = idx_nan[0]
            ax.axvline(x=ponto_morte, color='gray', linestyle='--', alpha=0.7)
            ax.annotate('Divergência Numérica\n(Overflow)', 
                        xy=(ponto_morte, 10**20), xytext=(ponto_morte-150, 10**30),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Títulos e Eixos (em Português ou Inglês conforme sua tese)
        ax.set_title('Análise de Robustez Numérica sob Estresse de Entrada', fontsize=14, pad=15)
        ax.set_xlabel('Número de Amostras (Treinamento Online)', fontsize=12)
        ax.set_ylabel('MSE Acumulado (Escala Log$_{10}$)', fontsize=12)
        
        # Grade refinada
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.legend(frameon=True, loc='upper left')

        plt.tight_layout()
        plt.savefig('grafico_erro_artigo.pdf') # Salvar em PDF para alta qualidade
        plt.show()

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    gerar_grafico_artigo()