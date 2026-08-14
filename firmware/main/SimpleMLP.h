#ifndef SIMPLE_MLP_H
#define SIMPLE_MLP_H

#include <math.h>
#include <stdlib.h>

/**
 * @brief Classe SimpleMLP para predicao de séries temporais.
 * Arquitetura: 3 (Entradas) -> 16 (Oculta 1) -> 8 (Oculta 2) -> 1 (Saída)
 */
class SimpleMLP {
public:
    // Pesos e Biases expostos para facilitar o acesso durante o treinamento/log
    // w1[16][3]: 16 neuronios, cada um com 3 entradas (t-1, t-2, bias)
    float w1[16][3], b1[16];
    
    // w2[8][16]: 8 neuronios conectados aos 16 da camada anterior
    float w2[8][16], b2[8];
    
    // w_out[1][8]: Camada de saida linear
    float w_out[1][8], b_out[1];

    // Buffers para armazenar as ativacoes (necessários para o backpropagation)
    float out1[16];
    float out2[8];
    float last_pred;

    /**
     * @brief Construtor que inicializa os pesos com a semente fornecida.
     * @param seed Semente para o gerador pseudo-aleatorio (isonomia experimental).
     */
    SimpleMLP(unsigned int seed);

    /**
     * @brief Realiza o passo a frente (inferencia).
     * @param inputs Array de 3 floats {t-1, t-2, 1.0f}.
     * @return Valor predito da temperatura.
     */
    float forward(float* inputs);

    /**
     * @brief Realiza o treinamento SGD (amostra por amostra).
     * @param inputs Dados de entrada.
     * @param target Valor real do sensor (alvo).
     * @param lr Taxa de aprendizado (learning rate).
     * @return Erro absoluto da predicao atual.
     */
    float train(float* inputs, float target, float lr);

    /**
     * @brief Aplica a restricao de Lipschitz via Weight Clipping.
     * @param L_max Limite superior para a Norma de Frobenius da primeira camada.
     */
    void aplicar_restricao_lipschitz(float L_max);

    /**
     * @brief Calcula a Norma de Frobenius total da rede (W1 e W2).
     * Utilizada para gerar os graficos de estabilidade parametrica.
     * @return Valor escalar da norma ||W||.
     */
    float calcular_norma_total();

private:
    /**
     * @brief Inicializa os pesos usando a técnica de Xavier/Glorot.
     */
    void inicializar_pesos();
};

#endif // SIMPLE_MLP_H