#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h" 
#include "SimpleMLP.h"

// Mantendo as redes estáticas para segurança de memória
static SimpleMLP redePadrao(42); 
static SimpleMLP redeLipschitz(42);

struct EstufaState {
    float t_prev1 = 25.0f;
    float t_prev2 = 25.0f;
};


extern "C" void app_main(void) {
    // --- CONFIGURAÇÃO DO WATCHDOG (Versão Minimalista) ---
    // Removemos .enabled e .trigger_proactive que causaram erro nas tentativas anteriores
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms = 15000,                             // 15 segundos de folga
        .idle_core_mask = (1 << portNUM_PROCESSORS) - 1  // Monitora todos os núcleos
    };
    
    // Tenta reconfigurar o Watchdog padrão do sistema
    esp_task_wdt_reconfigure(&twdt_config);
    esp_task_wdt_add(NULL); // Adiciona a tarefa atual (main) ao monitoramento

    // --- VARIÁREIS DO EXPERIMENTO ---
    const float lr = 0.25f;     // Taxa de aprendizado
    const float L_max = 1.0f;   // Limite de Lipschitz
    EstufaState stP, stL;
    char buffer[64];

    printf("Sistema Pronto. Iniciando Ciclo de Treino...\n");

    while (1) {
        // Alimenta o Watchdog no início do loop
        esp_task_wdt_reset();

        if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
            float temp_raw = atof(buffer);
            float temp_norm = temp_raw / 100.0f;

            float entradaP[3] = {stP.t_prev1/100.0f, stP.t_prev2/100.0f, 1.0f};
            float entradaL[3] = {stL.t_prev1/100.0f, stL.t_prev2/100.0f, 1.0f};

            // TREINO REDE PADRÃO (Usando a variável lr para evitar o warning)
            float predP_norm = redePadrao.forward(entradaP);
            redePadrao.train(entradaP, temp_norm, lr);
            
            // Pequena pausa para o sistema operacional não travar
            vTaskDelay(pdMS_TO_TICKS(1)); 

            // TREINO REDE LIPSCHITZ
            float predL_norm = redeLipschitz.forward(entradaL);
            redeLipschitz.train(entradaL, temp_norm, lr);
            redeLipschitz.aplicar_restricao_lipschitz(L_max);

            // CÁLCULO DE OUTPUT PARA O CSV
            float predP_celsius = predP_norm * 100.0f;
            float predL_celsius = predL_norm * 100.0f;

            printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    temp_raw, predP_celsius, predL_celsius, 
                    (predP_celsius - temp_raw), (predL_celsius - temp_raw), 
                    redePadrao.calcular_norma_total(), redeLipschitz.calcular_norma_total());

            // ATUALIZAÇÃO DOS ESTADOS
            stP.t_prev2 = stP.t_prev1; stP.t_prev1 = temp_raw;
            stL.t_prev2 = stL.t_prev1; stL.t_prev1 = temp_raw;
        }

        // Delay final para ceder tempo para a tarefa IDLE do sistema
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}