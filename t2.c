#include <stdio.h>  
#include <stdlib.h>  
#include <math.h>  

#define INPUT_NEURONS 2  
#define HIDDEN1_NEURONS 3  
#define HIDDEN2_NEURONS 2  
#define OUTPUT_NEURONS 1  
#define LEARNING_RATE 0.1  
#define ERROR_THRESHOLD 1e-6

typedef struct {  
	double weights[INPUT_NEURONS][HIDDEN1_NEURONS];  
	double biases[HIDDEN1_NEURONS];  
	double output[HIDDEN1_NEURONS];
} HiddenLayer1;  

typedef struct {  
	double weights[HIDDEN1_NEURONS][HIDDEN2_NEURONS];  
	double biases[HIDDEN2_NEURONS];  
	double output[HIDDEN2_NEURONS];
} HiddenLayer2;  

typedef struct {  
	double weights[HIDDEN2_NEURONS][OUTPUT_NEURONS];  
	double biases[OUTPUT_NEURONS];  
	double output[OUTPUT_NEURONS];
} OutputLayer;  

double sigmoid(double x) {  
	return 1.0 / (1.0 + exp(-x));  
}  

double sigmoid_derivative(double x) {  
	return x * (1.0 - x);  
}  

void forward_propagation(HiddenLayer1 *hidden_layer1, HiddenLayer2 *hidden_layer2, OutputLayer *output_layer, double *input, double *output) {  
	double hidden1_input[HIDDEN1_NEURONS];  
	double hidden1_output[HIDDEN1_NEURONS];  
	double hidden2_input[HIDDEN2_NEURONS];  
	double hidden2_output[HIDDEN2_NEURONS];  
	double output_input[OUTPUT_NEURONS];  

	// 第一隐藏层  
	for (int i = 0; i < HIDDEN1_NEURONS; i++) {  
		hidden1_input[i] = 0.0;  
		for (int j = 0; j < INPUT_NEURONS; j++) {  
			hidden1_input[i] += input[j] * hidden_layer1->weights[j][i];  
		}  
		hidden1_input[i] += hidden_layer1->biases[i];  
		hidden1_output[i] = sigmoid(hidden1_input[i]);  
		hidden_layer1->output[i] = hidden1_output[i];
	}  

	// 第二隐藏层  
	for (int i = 0; i < HIDDEN2_NEURONS; i++) {  
		hidden2_input[i] = 0.0;  
		for (int j = 0; j < HIDDEN1_NEURONS; j++) {  
			hidden2_input[i] += hidden1_output[j] * hidden_layer2->weights[j][i];  
		}  
		hidden2_input[i] += hidden_layer2->biases[i];  
		hidden2_output[i] = sigmoid(hidden2_input[i]);  
		hidden_layer2->output[i] = hidden2_output[i];
	}  

	// 输出层  
	for (int i = 0; i < OUTPUT_NEURONS; i++) {  
		output_input[i] = 0.0;  
		for (int j = 0; j < HIDDEN2_NEURONS; j++) {  
			output_input[i] += hidden2_output[j] * output_layer->weights[j][i];  
		}  
		output_input[i] += output_layer->biases[i];  
		output[i] = sigmoid(output_input[i]);  
		output_layer->output[i] = output[i];
	}  
}  

void backward_propagation(HiddenLayer1 *hidden_layer1, HiddenLayer2 *hidden_layer2, OutputLayer *output_layer, double *input, double *target_data, double *output_data) {  
	double output_delta[OUTPUT_NEURONS];  
	double hidden2_error[HIDDEN2_NEURONS];  
	double hidden2_delta[HIDDEN2_NEURONS];  
	double hidden1_error[HIDDEN1_NEURONS];  
	double hidden1_delta[HIDDEN1_NEURONS];  

	// 输出层  
	for (int i = 0; i < OUTPUT_NEURONS; i++) {  
		// 计算输出层的误差（即损失函数关于输出的导数）  
		output_delta[i] = (target_data[i] - output_data[i]) * sigmoid_derivative(output_layer->output[i]);  

		// 更新输出层到隐藏层的权重和偏置  
		for (int j = 0; j < HIDDEN2_NEURONS; j++) {  
			output_layer->weights[j][i] += LEARNING_RATE * output_delta[i] * hidden_layer2->output[j];  
		}  
		output_layer->biases[i] += LEARNING_RATE * output_delta[i];  
	}
	// 第二隐藏层  
	for (int i = 0; i < HIDDEN2_NEURONS; i++) {  
		hidden2_error[i] = 0.0;  
		for (int j = 0; j < OUTPUT_NEURONS; j++) {  
			hidden2_error[i] += output_delta[j] * output_layer->weights[i][j];  
		}  
		hidden2_delta[i] = hidden2_error[i] * sigmoid_derivative(hidden_layer2->output[i]);  

		// 更新第二隐藏层到输出层的权重和偏置  
		for (int j = 0; j < HIDDEN1_NEURONS; j++) {  
			hidden_layer2->weights[j][i] += LEARNING_RATE * hidden2_delta[i] * hidden_layer1->output[j];  
		}  
		hidden_layer2->biases[i] += LEARNING_RATE * hidden2_delta[i];  
	}  

	// 第一隐藏层  
	for (int i = 0; i < HIDDEN1_NEURONS; i++) {  
		hidden1_error[i] = 0.0;  
		for (int j = 0; j < HIDDEN2_NEURONS; j++) {  
			hidden1_error[i] += hidden2_delta[j] * hidden_layer2->weights[i][j];  
		}  
		hidden1_delta[i] = hidden1_error[i] * sigmoid_derivative(hidden_layer1->output[i]);  

		// 更新输入层到第一隐藏层的权重和偏置  
		for (int j = 0; j < INPUT_NEURONS; j++) {  
			hidden_layer1->weights[j][i] += LEARNING_RATE * hidden1_delta[i] * input[j];  
		}  
		hidden_layer1->biases[i] += LEARNING_RATE * hidden1_delta[i];  
	}  
}  

int main() {  
	HiddenLayer1 hidden_layer1;  
	HiddenLayer2 hidden_layer2;  
	OutputLayer output_layer;  

	// 初始化权重和偏置为随机值（这里简化为0）  
	for (int i = 0; i < INPUT_NEURONS; i++) {  
		for (int j = 0; j < HIDDEN1_NEURONS; j++) {  
			hidden_layer1.weights[i][j] = 0.0;  
		}  
	}  
	for (int i = 0; i < HIDDEN1_NEURONS; i++) {  
		hidden_layer1.biases[i] = 0.0;  
	}  
	for (int i = 0; i < HIDDEN1_NEURONS; i++) {  
		for (int j = 0; j < HIDDEN2_NEURONS; j++) {  
			hidden_layer2.weights[i][j] = 0.0;  
		}  
	}  
	for (int i = 0; i < HIDDEN2_NEURONS; i++) {  
		hidden_layer2.biases[i] = 0.0;  
	}  
	for (int i = 0; i < HIDDEN2_NEURONS; i++) {  
		for (int j = 0; j < OUTPUT_NEURONS; j++) {  
			output_layer.weights[i][j] = 0.0;  
		}  
	}  
	for (int i = 0; i < OUTPUT_NEURONS; i++) {  
		output_layer.biases[i] = 0.0;  
	}  

	// 训练数据  
	double input_data[INPUT_NEURONS] = {0.5, 0.3};  
	double target_data[OUTPUT_NEURONS] = {0.15};  
	double output_data[OUTPUT_NEURONS];  

	// 训练
	// 训练神经网络  
	int epochs = 10000; // 假设我们训练10000轮  

	for (int epoch = 0; epoch < epochs; epoch++) {  
		// 前向传播  
		forward_propagation(&hidden_layer1, &hidden_layer2, &output_layer, input_data, output_data);  

		// 计算输出层的误差  
		double output_error = 0.0;  
		for (int i = 0; i < OUTPUT_NEURONS; i++) {  
			output_error += (target_data[i] - output_data[i]) * (target_data[i] - output_data[i]);  
			printf("target: %f  output: %f\n", target_data[i], output_data[i]);  
		}  
		sleep(1);

		// 如果误差低于某个阈值，我们可以提前停止训练  
		if (output_error < ERROR_THRESHOLD) {  
			printf("Epoch %d complete. Error: %f\n", epoch+1, output_error);  
			break;  
		}  

		// 反向传播  
		backward_propagation(&hidden_layer1, &hidden_layer2, &output_layer, input_data, target_data, output_data);  

		// 打印进度  
		if ((epoch+1) % 1000 == 0) {  
			printf("Epoch %d complete. Error: %f\n", epoch+1, output_error);  
		}  
	}  

	// 训练完成后，可以使用训练好的网络进行预测  
	// ...  

	// 清理工作，释放动态分配的内存（如果有的话）  
	// ...  

	return 0;  
}  
