#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"

#define INPUT_NODE  2 // input  neuron num
#define HIDE_NODE   4 // hide   neuron num
#define OUTPUT_NODE 1 // output neuron num

#define MAX_NUM 300

double studyRate = 0.8;  //study rate
double threshold = 1e-4; //max mistake
double mostTimes = 1e6; //max study times 
double trainSize = 0; 
double testSize = 0; 

//sample
typedef struct Sample{
	double out[MAX_NUM][OUTPUT_NODE]; //output
	double in[MAX_NUM][INPUT_NODE]; //input
}Sample;

typedef struct Node{
	double value; //current value
	double loss_value; //loss value
	double bias; //bias value
	double bias_delta; // modify bias
	double *weight; //weight value
	double *weight_delta; //modify weight 
	double d_param; //
}Node;

Node inpuLayer[INPUT_NODE];
Node hideLayer[HIDE_NODE];
Node outputLayer[OUTPUT_NODE];

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double x)
{
	return x * (1 - x);
}

double loss(double x)
{
	return 0.5 * x * x;
}

double d_loss(double x)
{
	return -x;
}

double d_bias()
{
	return -1;
}

double d_weight(double x)
{
	return x;
}
double d_value(double x)
{
	return x;
}

Sample * getTrainData(const char * filename)
{
	Sample * result = malloc(sizeof(Sample));
	FILE * file = fopen(filename, "r");
	if(NULL != file)
	{
		int count = 0;
		while(fscanf(file,"%lf %lf %lf",
					&result->in[count][0], 
					&result->in[count][1], 
					&result->out[count][0]) != EOF)
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
		printf("%d %lf %lf %lf\n", i,
					data->in[i][0], 
					data->in[i][1], 
					data->out[i][0]);
	}
}

void init()
{
	int i,j;
	srand(time(0));

	//input init
	for(i = 0; i< INPUT_NODE; i++)
	{
		inpuLayer[i].weight = malloc(sizeof(double) * HIDE_NODE);
		inpuLayer[i].weight_delta = malloc(sizeof(double) * HIDE_NODE);
		inpuLayer[i].bias = 0.0;
		inpuLayer[i].bias_delta = 0.0;
		for(j = 0; j< HIDE_NODE; j++)
		{
			inpuLayer[i].weight[j] = rand() % 10000 / (double)10000 * 2 - 1;
			inpuLayer[i].weight_delta[j] = 0.0;
		}
	}
	
	//hide init
	for(i = 0; i< HIDE_NODE; i++)
	{
		hideLayer[i].weight = malloc(sizeof(double) * OUTPUT_NODE);
		hideLayer[i].weight_delta = malloc(sizeof(double) * OUTPUT_NODE);
		hideLayer[i].bias = rand() % 10000 / (double)10000 * 2 - 1;
		hideLayer[i].bias_delta = 0.0;
		for(j = 0; j< OUTPUT_NODE; j++)
		{
			hideLayer[i].weight[j] = rand() % 10000 / (double)10000 * 2 - 1;
			hideLayer[i].weight_delta[j] = 0.0;
		}
	}
	
	//output init
	for(i = 0; i< OUTPUT_NODE; i++)
	{
		outputLayer[i].bias = rand() % 10000 / (double)10000 * 2 - 1;
		outputLayer[i].bias_delta = 0.0;
	}
}

void resetDelta()
{
	int i;
	int j;
	for(i = 0; i < INPUT_NODE; i++)
	{
		for(j = 0; j < HIDE_NODE; j++)
		{
			inpuLayer[i].weight_delta[j] = 0.0;
		}
	}

	for(i = 0; i < HIDE_NODE; i++)
	{
		hideLayer[i].bias_delta = 0.0;
		for(j = 0; j < OUTPUT_NODE; j++)
		{
			hideLayer[i].weight_delta[j] = 0.0;
		}
	}

	for(j = 0; j < OUTPUT_NODE; j++)
	{
		outputLayer[j].bias_delta = 0.0;
	}
}


double Max (double a, double b)
{
	return a > b ? a : b;
}
int main()
{
	init();
	Sample * trainSample =getTrainData("TrainData.txt");
	printData(trainSample, trainSize);
	
	int trainTime;
	int currTrainSample_pos;
	int inputLayer_post;
	int outputlayer_pos;
	int hidelayer_pos;
	for(trainTime = 0; trainTime < mostTimes; trainTime++)
	{
		resetDelta();
    
		//max error
		double error_max = 0.0;

		for(currTrainSample_pos = 0; currTrainSample_pos < trainSize; currTrainSample_pos++)
		{
			//init input
			for(inputLayer_post= 0; inputLayer_post < INPUT_NODE; inputLayer_post++)
			{
				inpuLayer[inputLayer_post].value = trainSample->in[currTrainSample_pos][inputLayer_post];
			}

			//forward spread input -> hide 
			for(hidelayer_pos= 0; hidelayer_pos < HIDE_NODE; hidelayer_pos++)
			{
				double sum = 0.0;
				for(inputLayer_post= 0; inputLayer_post < INPUT_NODE; inputLayer_post++)
				{
					sum += inpuLayer[inputLayer_post].value * inpuLayer[inputLayer_post].weight[hidelayer_pos];
				}

				sum -= hideLayer[hidelayer_pos].bias;
				hideLayer[hidelayer_pos].value = sigmoid(sum);
			}
			//forward spread hide -> output
			for(outputlayer_pos = 0; outputlayer_pos < OUTPUT_NODE; outputlayer_pos++)
			{
				double sum = 0.0;
				for(hidelayer_pos= 0; hidelayer_pos < HIDE_NODE; hidelayer_pos++)
				{
					sum += hideLayer[hidelayer_pos].value * hideLayer[hidelayer_pos].weight[outputlayer_pos];
				}

				sum -= outputLayer[outputlayer_pos].bias;
				outputLayer[outputlayer_pos].value = sigmoid(sum);
			}

			//calculus error
			double error = 0.0;
			for(outputlayer_pos = 0; outputlayer_pos < OUTPUT_NODE; outputlayer_pos++)
			{
				double temp = fabs(outputLayer[outputlayer_pos].value - 
						trainSample->out[currTrainSample_pos][outputlayer_pos]
						);
				//loss func
				error += temp * temp / 2.0;
			}
			
			error_max = Max(error_max, error);

			for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
			{
				outputLayer[outputlayer_pos].loss_value = 
						trainSample->out[currTrainSample_pos][outputlayer_pos] - 
						outputLayer[outputlayer_pos].value;
			}
			//backward spread output -> hide check output bias
			for(outputlayer_pos = 0; outputlayer_pos < OUTPUT_NODE; outputlayer_pos++)
			{
				double bias_delta = 1;

				bias_delta *= d_loss(outputLayer[outputlayer_pos].loss_value);
				bias_delta *= d_sigmoid(outputLayer[outputlayer_pos].value);
				outputLayer[outputlayer_pos].d_param = bias_delta;

				bias_delta *= d_bias();
				bias_delta *= -1;
				outputLayer[outputlayer_pos].bias_delta += bias_delta;
			}
			//backward spread output -> hide check hide weight
			for(hidelayer_pos = 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
				{
					double weight_delta = 1;
					weight_delta *= outputLayer[outputlayer_pos].d_param;
					weight_delta *= d_weight(hideLayer[hidelayer_pos].value);
					weight_delta *= -1;
					hideLayer[hidelayer_pos].weight_delta[outputlayer_pos] += weight_delta;
				}
			}
			//backward spread output -> hide check hide bias 
			for(hidelayer_pos = 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				double sum_delta = 0;
				for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
				{
				  double sum = 1;
					sum *= outputLayer[outputlayer_pos].d_param;
					sum *= d_value(hideLayer[hidelayer_pos].weight[outputlayer_pos]);
					sum *= d_sigmoid(hideLayer[hidelayer_pos].value);
					hideLayer[hidelayer_pos].d_param = sum;
					sum *= d_bias();
					sum_delta += sum;
				}
				sum_delta *= -1;
				hideLayer[hidelayer_pos].bias_delta += sum_delta;
			}

			//backward spread hide -> input check input weight
			for(inputLayer_post= 0; inputLayer_post< INPUT_NODE; inputLayer_post++)
			{
				for(hidelayer_pos= 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
				{
					double sum = 1;
					sum *= hideLayer[hidelayer_pos].d_param;
					sum *= d_weight(inpuLayer[inputLayer_post].value);
					sum *= -1;
					inpuLayer[inputLayer_post].weight_delta[hidelayer_pos] += sum;
				}
			}
		}
		if(error_max < threshold)
		{
			printf("train complete, train:%d max error:%lf\n", trainTime + 1, error_max);
			break;
		}

		//modify
		for(inputLayer_post = 0 ; inputLayer_post < INPUT_NODE; inputLayer_post++)
		{
			for(hidelayer_pos = 0 ; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				inpuLayer[inputLayer_post].weight[hidelayer_pos ] += studyRate * 
					inpuLayer[inputLayer_post].weight_delta[hidelayer_pos] /(double) trainSize;
			}
		}
		for(hidelayer_pos = 0 ; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
		{
			hideLayer[hidelayer_pos].bias +=studyRate * hideLayer[hidelayer_pos].bias_delta / (double)trainSize;
		  for(outputlayer_pos= 0 ; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
		  {
				hideLayer[hidelayer_pos].weight[outputlayer_pos] += studyRate * 
					hideLayer[hidelayer_pos].weight_delta[outputlayer_pos] /(double) trainSize;
			}
		}
		for(outputlayer_pos= 0 ; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
		{
			outputLayer[outputlayer_pos].bias +=studyRate * outputLayer[outputlayer_pos].bias_delta / (double)trainSize;
		}
	}

	Sample * testSample = getTestData("TestData.txt");
	for(currTrainSample_pos = 0; currTrainSample_pos < testSize; currTrainSample_pos++)
	{
		for(inputLayer_post = 0; inputLayer_post < INPUT_NODE; inputLayer_post++)
		{
			inpuLayer[inputLayer_post].value = testSample->in[currTrainSample_pos][inputLayer_post];
		}
		for(hidelayer_pos = 0; hidelayer_pos < HIDE_NODE; hidelayer_pos++)
		{
			double sum = 0.0;
			for(inputLayer_post= 0; inputLayer_post< INPUT_NODE; inputLayer_post++)
			{
				sum += inpuLayer[inputLayer_post].value * inpuLayer[inputLayer_post].weight[hidelayer_pos];
			}
			sum -= hideLayer[hidelayer_pos].bias;
			hideLayer[hidelayer_pos].value = sigmoid(sum);
		}
		for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
		{
			double sum = 0.0;
			for(hidelayer_pos= 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				sum += hideLayer[hidelayer_pos].value * hideLayer[hidelayer_pos].weight[outputlayer_pos];
			}
			sum -= outputLayer[outputlayer_pos].bias;
			outputLayer[outputlayer_pos].value = sigmoid(sum);
		}

		for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
		{
			testSample->out[currTrainSample_pos][outputlayer_pos] = outputLayer[outputlayer_pos].value;
		}
	}

	printData(testSample, testSize);
	return 0;
}
