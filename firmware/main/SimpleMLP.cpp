#include "SimpleMLP.h"

SimpleMLP::SimpleMLP(unsigned int seed) {
    srand(seed);
    inicializar_pesos();
}

void SimpleMLP::inicializar_pesos() {
    // Camada 1: 3 -> 16 (Xavier Initialization)
    float scale1 = sqrt(2.0f / (3.0f + 16.0f));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 3; j++) {
            w1[i][j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale1;
        }
        b1[i] = 0.05f; 
    }

    // Camada 2: 16 -> 8
    float scale2 = sqrt(2.0f / (16.0f + 8.0f));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) {
            w2[i][j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale2;
        }
        b2[i] = 0.05f;
    }

    // Camada Saida: 8 -> 1
    float scale_out = sqrt(2.0f / (8.0f + 1.0f));
    for (int j = 0; j < 8; j++) {
        w_out[0][j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale_out;
    }
    b_out[0] = 0.0f;
}

float SimpleMLP::forward(float* inputs) {
    // Camada Oculta 1 com Leaky ReLU
    for (int i = 0; i < 16; i++) {
        float sum = b1[i];
        for (int j = 0; j < 3; j++) sum += inputs[j] * w1[i][j];
        out1[i] = (sum > 0) ? sum : sum * 0.01f; 
    }

    // Camada Oculta 2 com Leaky ReLU
    for (int i = 0; i < 8; i++) {
        float sum = b2[i];
        for (int j = 0; j < 16; j++) sum += out1[j] * w2[i][j];
        out2[i] = (sum > 0) ? sum : sum * 0.01f;
    }

    // Saida Linear (Regressao)
    last_pred = b_out[0];
    for (int j = 0; j < 8; j++) last_pred += out2[j] * w_out[0][j];
    
    return last_pred;
}

float SimpleMLP::train(float* inputs, float target, float lr) {
    float pred = forward(inputs);
    float error_out = pred - target; 

    // 1. Retropropagacao do Erro - Camada 2 para Saida
    float delta2[8];
    for (int i = 0; i < 8; i++) {
        float d_lrelu2 = (out2[i] > 0) ? 1.0f : 0.01f;
        delta2[i] = error_out * w_out[0][i] * d_lrelu2;
    }

    // 2. Retropropagacao do Erro - Camada 1 para Camada 2
    float delta1[16];
    for (int i = 0; i < 16; i++) {
        float sum_deltas = 0;
        for (int k = 0; k < 8; k++) sum_deltas += delta2[k] * w2[k][i];
        float d_lrelu1 = (out1[i] > 0) ? 1.0f : 0.01f;
        delta1[i] = sum_deltas * d_lrelu1;
    }

    // 3. Atualizacao dos Pesos e Biases (SGD Unit Batch)
    float b_lr = lr * 5.0f; // Multiplicador para o Bias convergir mais rápido
    
    // Ajuste Saida
    for (int j = 0; j < 8; j++) w_out[0][j] -= lr * error_out * out2[j];
    b_out[0] -= b_lr * error_out;

    // Ajuste Camada 2
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) w2[i][j] -= lr * delta2[i] * out1[j];
        b2[i] -= b_lr * delta2[i];
    }

    // Ajuste Camada 1
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 3; j++) w1[i][j] -= lr * delta1[i] * inputs[j];
        b1[i] -= b_lr * delta1[i];
    }

    return fabsf(error_out);
}

void SimpleMLP::aplicar_restricao_lipschitz(float L_max) {
    // Restricao na Camada 1 (Projecao de Norma de Frobenius)
    float soma1 = 0;
    for(int i=0; i<16; i++) for(int j=0; j<3; j++) soma1 += w1[i][j]*w1[i][j];
    float norma1 = sqrt(soma1);
    
    if (norma1 > L_max) {
        float fator = L_max / norma1;
        for(int i=0; i<16; i++) for(int j=0; j<3; j++) w1[i][j] *= fator;
    }

    // Restricao na Camada 2 (Usando 1.5x L_max como folga estrutural)
    float soma2 = 0;
    for(int i=0; i<8; i++) for(int j=0; j<16; j++) soma2 += w2[i][j]*w2[i][j];
    float norma2 = sqrt(soma2);
    
    if (norma2 > (L_max * 1.5f)) {
        float fator = (L_max * 1.5f) / norma2;
        for(int i=0; i<8; i++) for(int j=0; j<16; j++) w2[i][j] *= fator;
    }
}

float SimpleMLP::calcular_norma_total() {
    float soma = 0;
    for(int i=0; i<16; i++) for(int j=0; j<3; j++) soma += w1[i][j]*w1[i][j];
    for(int i=0; i<8; i++) for(int j=0; j<16; j++) soma += w2[i][j]*w2[i][j];
    return sqrt(soma);
}