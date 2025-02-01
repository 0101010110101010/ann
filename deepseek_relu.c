#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 定义神经网络结构
#define INPUT_NODES 2
#define HIDDEN_LAYERS 2
#define HIDDEN_NODES 8
#define OUTPUT_NODES 3

// 定义学习率和训练次数
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// ReLU 激活函数
double relu(double x) {
    return x > 0 ? x : 0;
}

// ReLU 的导数
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// 初始化权重和偏置
void initialize(double *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // 随机初始化在[-1, 1]之间
    }
}

// 前向传播
void forward(double input[INPUT_NODES], double hidden[HIDDEN_LAYERS][HIDDEN_NODES], double output[OUTPUT_NODES],
             double weights_input_hidden[HIDDEN_NODES][INPUT_NODES],
             double weights_hidden_hidden[HIDDEN_LAYERS - 1][HIDDEN_NODES][HIDDEN_NODES],
             double weights_hidden_output[OUTPUT_NODES][HIDDEN_NODES],
             double bias_hidden[HIDDEN_LAYERS][HIDDEN_NODES], double bias_output[OUTPUT_NODES]) {

    // 计算第一个隐藏层的输出
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden[0][i] = 0;
        for (int j = 0; j < INPUT_NODES; j++) {
            hidden[0][i] += input[j] * weights_input_hidden[i][j];
        }
        hidden[0][i] += bias_hidden[0][i];
        hidden[0][i] = relu(hidden[0][i]); // 使用 ReLU 激活
    }

    // 计算后续隐藏层的输出
    for (int l = 1; l < HIDDEN_LAYERS; l++) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            hidden[l][i] = 0;
            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden[l][i] += hidden[l - 1][j] * weights_hidden_hidden[l - 1][i][j];
            }
            hidden[l][i] += bias_hidden[l][i];
            hidden[l][i] = relu(hidden[l][i]); // 使用 ReLU 激活
        }
    }

    // 计算输出层的输出（无激活函数）
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            output[i] += hidden[HIDDEN_LAYERS - 1][j] * weights_hidden_output[i][j];
        }
        output[i] += bias_output[i];
        // 无激活函数，直接输出
    }
}

// 反向传播
void backward(double input[INPUT_NODES], double hidden[HIDDEN_LAYERS][HIDDEN_NODES], double output[OUTPUT_NODES],
              double target[OUTPUT_NODES],
              double weights_input_hidden[HIDDEN_NODES][INPUT_NODES],
              double weights_hidden_hidden[HIDDEN_LAYERS - 1][HIDDEN_NODES][HIDDEN_NODES],
              double weights_hidden_output[OUTPUT_NODES][HIDDEN_NODES],
              double bias_hidden[HIDDEN_LAYERS][HIDDEN_NODES], double bias_output[OUTPUT_NODES], double learning_rate) {

    double output_error[OUTPUT_NODES];
    double hidden_error[HIDDEN_LAYERS][HIDDEN_NODES];

    // 计算输出层的误差（线性激活，导数为1）
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output_error[i] = (target[i] - output[i]) * 1.0; // 线性激活，导数为1
    }

    // 计算最后一个隐藏层的误差
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden_error[HIDDEN_LAYERS - 1][i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden_error[HIDDEN_LAYERS - 1][i] += output_error[j] * weights_hidden_output[j][i];
        }
        hidden_error[HIDDEN_LAYERS - 1][i] *= relu_derivative(hidden[HIDDEN_LAYERS - 1][i]); // ReLU 导数
    }

    // 计算其他隐藏层的误差
    for (int l = HIDDEN_LAYERS - 2; l >= 0; l--) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            hidden_error[l][i] = 0;
            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden_error[l][i] += hidden_error[l + 1][j] * weights_hidden_hidden[l][j][i];
            }
            hidden_error[l][i] *= relu_derivative(hidden[l][i]); // ReLU 导数
        }
    }

    // 更新输出层的权重和偏置
    for (int i = 0; i < OUTPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weights_hidden_output[i][j] += learning_rate * output_error[i] * hidden[HIDDEN_LAYERS - 1][j];
        }
        bias_output[i] += learning_rate * output_error[i];
    }

    // 更新隐藏层之间的权重和偏置
    for (int l = HIDDEN_LAYERS - 1; l > 0; l--) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weights_hidden_hidden[l - 1][i][j] += learning_rate * hidden_error[l][i] * hidden[l - 1][j];
            }
            bias_hidden[l][i] += learning_rate * hidden_error[l][i];
        }
    }

    // 更新输入层到第一个隐藏层的权重和偏置
    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            weights_input_hidden[i][j] += learning_rate * hidden_error[0][i] * input[j];
        }
        bias_hidden[0][i] += learning_rate * hidden_error[0][i];
    }
}

int main() {
    srand(time(NULL));

    // 定义权重和偏置
    double weights_input_hidden[HIDDEN_NODES][INPUT_NODES];
    double weights_hidden_hidden[HIDDEN_LAYERS - 1][HIDDEN_NODES][HIDDEN_NODES];
    double weights_hidden_output[OUTPUT_NODES][HIDDEN_NODES];
    double bias_hidden[HIDDEN_LAYERS][HIDDEN_NODES];
    double bias_output[OUTPUT_NODES];

    // 初始化权重和偏置
    initialize((double *)weights_input_hidden, HIDDEN_NODES * INPUT_NODES);
    initialize((double *)weights_hidden_hidden, (HIDDEN_LAYERS - 1) * HIDDEN_NODES * HIDDEN_NODES);
    initialize((double *)weights_hidden_output, OUTPUT_NODES * HIDDEN_NODES);
    initialize((double *)bias_hidden, HIDDEN_LAYERS * HIDDEN_NODES);
    initialize(bias_output, OUTPUT_NODES);

    // 训练数据（已归一化到 [0,1]）
    double inputs[][INPUT_NODES] = {
        {0.333, 0.333}, {1.0, 1.0}, {0.333, 1.0}, {0.0, 0.0}, {0.0, 0.00333},
        {0.00333, 0.0}, {0.00333, 0.00333}, {0.00267, 0.00267}, {0.002, 0.002},
        {0.00133, 0.00133}, {0.00067, 0.00067}, {0.00333, 0.00267}, {0.00333, 0.002},
        {0.00333, 0.00133}, {0.00333, 0.00067}, {0.00267, 0.002}, {0.002, 0.00133},
        {0.00133, 0.00067}, {0.00067, 0.0}, {0.00333, 0.00222}, {0.00222, 0.00111},
        {0.00111, 0.0}, {0.00267, 0.00133}, {0.00133, 0.0}, {0.0, 0.00041},
        {0.0004, 0.00077}, {0.00077, 0.00113}, {0.00113, 0.0015}, {0.0015, 0.00187},
        {0.00187, 0.00223}, {0.00223, 0.0026}, {0.0026, 0.00297}, {0.00297, 0.0033}
    };
    double targets[][OUTPUT_NODES] = {
        {0, 0, 0}, {0, 0, 0}, {0.333, 0.666, 1.0}, {0, 0, 0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
        {0, 0, 0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0},
        {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}, {0.333, 0.666, 1.0}
    };

    // 训练神经网络
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double learning_rate = LEARNING_RATE * (1.0 - (double)epoch / EPOCHS); // 动态学习率
        for (int i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
            double hidden[HIDDEN_LAYERS][HIDDEN_NODES] = {0};
            double output[OUTPUT_NODES] = {0};

            // 前向传播
            forward(inputs[i], hidden, output, weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);

            // 反向传播
            backward(inputs[i], hidden, output, targets[i], weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate);
        }
    }

    // 测试神经网络
    printf("Testing the trained neural network:\n");
    for (int i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        double hidden[HIDDEN_LAYERS][HIDDEN_NODES] = {0};
        double output[OUTPUT_NODES] = {0};

        forward(inputs[i], hidden, output, weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);

        // 反归一化输出
        double output_denormalized[OUTPUT_NODES];
        for (int j = 0; j < OUTPUT_NODES; j++) {
            output_denormalized[j] = output[j] * 3.0; // 反归一化到 [0,3]
        }

        printf("Input: [%f, %f] -> Output: [%f, %f, %f]\n", inputs[i][0], inputs[i][1], output_denormalized[0], output_denormalized[1], output_denormalized[2]);
    }

    return 0;
}
