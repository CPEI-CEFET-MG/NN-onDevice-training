import serial
import time
import csv
import numpy as np

# --- CONFIGURAÇÃO ---
PORTA = 'COM3'  # <--- VEJA NO SEU DISPOSITIVO QUAL É A PORTA
BAUD = 115200

# Criar dados: Senoide (estufa aquecendo) com ruído e um OUTLIER
amostras = 400
t = np.linspace(0, 10, amostras)
dados = 25 + 10 * np.sin(t * 0.5) # Temperatura variando entre 25 e 35 graus

# INJEÇÃO DO ERRO (O IMPACTO): Na amostra 200, o sensor "enlouquece"
dados[200:205] = 999.0 

try:
    # Abre a conexão com o ESP32
    esp32 = serial.Serial(PORTA, BAUD, timeout=1)
    time.sleep(2) # Espera o ESP32 reiniciar após a conexão
    
    with open('resultados_artigo.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Amostra", "Sensor", "PredPadrao", "PredLipschitz", "ErroP", "ErroL", "NormaP", "NormaL"])
        
        print("Iniciando simulação... O ESP32 está aprendendo agora!")
        
        for i in range(amostras):
            valor_sensor = f"{dados[i]:.2f}\n"
            
            # 1. Envia o número para o ESP32 via Serial
            esp32.write(valor_sensor.encode())
            
            # 2. Espera o ESP32 processar e responder
            resposta = esp32.readline().decode('utf-8').strip()
            
            if resposta:
                # O ESP32 responde no formato CSV que configuramos no printf
                print(f"Amostra {i} | Sensor: {dados[i]:.2f} | Resposta ESP32: {resposta}")
                writer.writerow([i] + resposta.split(','))
            
            # Simula um sensor que lê a cada 50ms
            time.sleep(0.05)

    esp32.close()
    print("Fim do experimento. Verifique o arquivo 'resultados_artigo.csv'!")

except Exception as e:
    print(f"Erro na conexão: {e}")