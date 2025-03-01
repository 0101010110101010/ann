#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 定义神经网络结构
#define INPUT_NODES 2
#define HIDDEN_LAYERS 2
#define HIDDEN_NODES 10
#define OUTPUT_NODES 3

// 定义学习率和训练次数
#define LEARNING_RATE 0.1
#define EPOCHS 200000

#define MAX_NUM 300

double trainSize = 0;   
double testSize = 0;    

//sample                                     
typedef struct Sample{                       
  double out[MAX_NUM][OUTPUT_NODES]; //output 
  double in[MAX_NUM][INPUT_NODES]; //input    
}Sample;                                     

Sample * getTrainData(const char * filename)        
{                                                   
  Sample * result = malloc(sizeof(Sample));         
	memset(result, 0, sizeof(Sample));
  FILE * file = fopen(filename, "r");               
  if(NULL != file)                                  
  {                                                 
    int count = 0;                                  
    while(fscanf(file,"%lf %lf %lf %lf %lf",                
          &result->in[count][0],                    
          &result->in[count][1],                    
          &result->out[count][0],
          &result->out[count][1],
          &result->out[count][2]) != EOF)           
    {                                               
      count++;                                      
    }                                               
    trainSize = count;                              
    printf("read ok\n");                            
    fclose(file);                                   
    return result;                                  
                                                    
  }                                                 
  else                                              
  {                                                 
    fclose(file);                                   
    printf("file open error\n");                    
    return NULL;                                    
  }                                                 
  return result;                                    
}                                                   

Sample * getTestData(const char * filename)      
{                                                
  Sample * result = malloc(sizeof(Sample));      
	memset(result, 0, sizeof(Sample));
  FILE * file = fopen(filename, "r");            
  if(NULL != file)                               
  {                                              
    int count = 0;                               
    while(fscanf(file,"%lf %lf",                 
          &result->in[count][0],                 
          &result->in[count][1]) != EOF)         
    {                                            
      count++;                                   
    }                                            
    testSize = count;                            
    printf("read ok\n");                         
    fclose(file);                                
    return result;                               
  }                                              
  else                                           
  {                                              
    fclose(file);                                
    printf("file open error\n");                 
    return NULL;                                 
  }                                              
                                                 
  return result;                                 
}                                                

void printData(Sample * data, int size)              
{                                                    
  int i;                                             
  if(data == NULL)                                   
  {                                                  
    printf("null data\n");                           
    return;                                          
  }                                                  
  for(i = 0; i < size; i++)                          
  {                                                  
    printf("%d %lf %lf %lf,%lf,%lf\n", i,                    
          data->in[i][0],                            
          data->in[i][1],                            
          data->out[i][0],                          
          data->out[i][1],                          
          data->out[i][2]);                          
  }                                                  
}                                                    

// 激活函数：Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 激活函数的导数
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
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
        hidden[0][i] = sigmoid(hidden[0][i]);
    }

    // 计算后续隐藏层的输出
    for (int l = 1; l < HIDDEN_LAYERS; l++) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            hidden[l][i] = 0;
            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden[l][i] += hidden[l - 1][j] * weights_hidden_hidden[l - 1][i][j];
            }
            hidden[l][i] += bias_hidden[l][i];
            hidden[l][i] = sigmoid(hidden[l][i]);
        }
    }

    // 计算输出层的输出
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            output[i] += hidden[HIDDEN_LAYERS - 1][j] * weights_hidden_output[i][j];
        }
        output[i] += bias_output[i];
        output[i] = sigmoid(output[i]);
    }
}

// 反向传播
void backward(double input[INPUT_NODES], double hidden[HIDDEN_LAYERS][HIDDEN_NODES], double output[OUTPUT_NODES],
              double target[OUTPUT_NODES],
              double weights_input_hidden[HIDDEN_NODES][INPUT_NODES],
              double weights_hidden_hidden[HIDDEN_LAYERS - 1][HIDDEN_NODES][HIDDEN_NODES],
              double weights_hidden_output[OUTPUT_NODES][HIDDEN_NODES],
              double bias_hidden[HIDDEN_LAYERS][HIDDEN_NODES], double bias_output[OUTPUT_NODES]) {

    double output_error[OUTPUT_NODES];
    double hidden_error[HIDDEN_LAYERS][HIDDEN_NODES];

    // 计算输出层的误差
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output_error[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
        //output_error[i] = (target[i] - output[i]) ;
    }

    // 计算最后一个隐藏层的误差
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden_error[HIDDEN_LAYERS - 1][i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden_error[HIDDEN_LAYERS - 1][i] += output_error[j] * weights_hidden_output[j][i];
        }
        hidden_error[HIDDEN_LAYERS - 1][i] *= sigmoid_derivative(hidden[HIDDEN_LAYERS - 1][i]);
    }

    // 计算其他隐藏层的误差
    for (int l = HIDDEN_LAYERS - 2; l >= 0; l--) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            hidden_error[l][i] = 0;
            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden_error[l][i] += hidden_error[l + 1][j] * weights_hidden_hidden[l][j][i];
            }
            hidden_error[l][i] *= sigmoid_derivative(hidden[l][i]);
        }
    }

    // 更新输出层的权重和偏置
    for (int i = 0; i < OUTPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weights_hidden_output[i][j] += LEARNING_RATE * output_error[i] * hidden[HIDDEN_LAYERS - 1][j];
        }
        bias_output[i] += LEARNING_RATE * output_error[i];
    }

    // 更新隐藏层之间的权重和偏置
    for (int l = HIDDEN_LAYERS - 1; l > 0; l--) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weights_hidden_hidden[l - 1][i][j] += LEARNING_RATE * hidden_error[l][i] * hidden[l - 1][j];
            }
            bias_hidden[l][i] += LEARNING_RATE * hidden_error[l][i];
        }
    }

    // 更新输入层到第一个隐藏层的权重和偏置
    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            weights_input_hidden[i][j] += LEARNING_RATE * hidden_error[0][i] * input[j];
        }
        bias_hidden[0][i] += LEARNING_RATE * hidden_error[0][i];
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

		Sample * trainSample =getTrainData("TrainDataDeep.txt");    
		printData(trainSample, trainSize);                      
		Sample * testSample = getTestData("TestData.txt");

    // 初始化权重和偏置
    initialize((double *)weights_input_hidden, HIDDEN_NODES * INPUT_NODES);
    initialize((double *)weights_hidden_hidden, (HIDDEN_LAYERS - 1) * HIDDEN_NODES * HIDDEN_NODES);
    initialize((double *)weights_hidden_output, OUTPUT_NODES * HIDDEN_NODES);
    initialize((double *)bias_hidden, HIDDEN_LAYERS * HIDDEN_NODES);
    initialize(bias_output, OUTPUT_NODES);

    // 训练数据

    // 训练神经网络
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
				//printf("train[%d]\r", epoch);
        for (int i = 0; i < trainSize; i++) {
            double hidden[HIDDEN_LAYERS][HIDDEN_NODES] = {0};
            double output[OUTPUT_NODES] = {0};

            // 前向传播
            forward(trainSample->in[i], hidden, output, weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);

            // 反向传播
            backward(trainSample->in[i], hidden, output, trainSample->out[i], weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);
        }

			#if 0
			// 测试神经网络
			system("clear");
			printf("Testing the trained neural network[%u - %u]:\n", epoch, EPOCHS);
			for (int i = 0; i < trainSize; i++) {
					double hidden[HIDDEN_LAYERS][HIDDEN_NODES] = {0};
					double output[OUTPUT_NODES] = {0};

					forward(trainSample->in[i], hidden, output, weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);
			
					// 计算输出层的误差
					double output_error[OUTPUT_NODES] = {0};
					for (int j = 0; j < OUTPUT_NODES; j++)
							output_error[j] += (trainSample->out[i][j] - output[j]) * sigmoid_derivative(output[j]);

					printf("Input: [%f, %f] -> Output: [%f, %f, %f] [%f, %f, %f] err:[%f,%f,%f] sub:[%f,%f,%f]\n", 
							trainSample->in[i][0], trainSample->in[i][1], 
							output[0], output[1], output[2], 
							trainSample->out[i][0], trainSample->out[i][1], trainSample->out[i][2], 
							output_error[0], output_error[1], output_error[2],
							trainSample->out[i][0] - output[0],
							trainSample->out[i][1] - output[1],
							trainSample->out[i][2] - output[2]
							);
			}
			#else
				printf("train[%d]\r", epoch);
			#endif
    }

    // 测试神经网络
    printf("Testing the trained neural network:\n");
    for (int i = 0; i < testSize; i++) {
        double hidden[HIDDEN_LAYERS][HIDDEN_NODES] = {0};
        double output[OUTPUT_NODES] = {0};

        forward(testSample->in[i], hidden, output, weights_input_hidden, weights_hidden_hidden, weights_hidden_output, bias_hidden, bias_output);

        printf("Input: [%f, %f] -> Output: [%f, %f, %f]\n", testSample->in[i][0], testSample->in[i][1], output[0], output[1], output[2]);
    }

    return 0;
}
