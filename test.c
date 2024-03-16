#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"

#define INPUT_NODE  2 // input  neuron num
#define HIDE_NODE   4 // hide   neuron num
#define HIDE_NODE_COL   2 // hide   neuron num
#define OUTPUT_NODE 1 // output neuron num

#define MAX_NUM 300

double studyRate = 1.8;  //study rate
double threshold = 1e-4; //max mistake
double mostTimes = 1e10; //max study times 
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
Node hideLayer[HIDE_NODE][HIDE_NODE_COL];
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
	int i,j,k;
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
			//inpuLayer[i].weight[j] = rand() % 10000 / (double)10000 * 2 - 1;
			inpuLayer[i].weight[j] = 1;
			inpuLayer[i].weight_delta[j] = 0.0;
		}
	}

	for(i = 0; i< HIDE_NODE; i++)
	{
		for(j = 0; j< HIDE_NODE_COL - 1; j++)
		{
			hideLayer[i][j].weight = malloc(sizeof(double) * HIDE_NODE);
			hideLayer[i][j].weight_delta = malloc(sizeof(double) * HIDE_NODE);
			hideLayer[i][j].bias = rand() % 10000 / (double)10000 * 2 - 1;
			hideLayer[i][j].bias_delta = 0.0;
			for(k = 0; k< HIDE_NODE; k++)
			{
				//hideLayer[i][j].weight[k] = rand() % 10000 / (double)10000 * 2 - 1;
				hideLayer[i][j].weight[k] = 1;
				hideLayer[i][j].weight_delta[k] = 0.0;
			}
		}
	}
	
	//hide init
	for(i = 0; i< HIDE_NODE; i++)
	{
		hideLayer[i][HIDE_NODE_COL - 1].weight = malloc(sizeof(double) * OUTPUT_NODE);
		hideLayer[i][HIDE_NODE_COL - 1].weight_delta = malloc(sizeof(double) * OUTPUT_NODE);
		hideLayer[i][HIDE_NODE_COL - 1].bias = rand() % 10000 / (double)10000 * 2 - 1;
		hideLayer[i][HIDE_NODE_COL - 1].bias_delta = 0.0;
		for(j = 0; j< OUTPUT_NODE; j++)
		{
			//hideLayer[i][HIDE_NODE_COL - 1].weight[j] = rand() % 10000 / (double)10000 * 2 - 1;
			hideLayer[i][HIDE_NODE_COL - 1].weight[j] = 1;
			hideLayer[i][HIDE_NODE_COL - 1].weight_delta[j] = 0.0;
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
	int k;
	for(i = 0; i < INPUT_NODE; i++)
	{
		for(j = 0; j < HIDE_NODE; j++)
		{
			inpuLayer[i].weight_delta[j] = 0.0;
		}
	}

	for(i = 0; i< HIDE_NODE; i++)
	{
		for(j = 0; j< HIDE_NODE_COL - 1; j++)
		{
			hideLayer[i][j].bias_delta = 0.0;
			for(k = 0; k< HIDE_NODE; k++)
				hideLayer[i][j].weight_delta[k] = 0.0;
		}
	}

	for(i = 0; i < HIDE_NODE; i++)
	{
		hideLayer[i][HIDE_NODE_COL - 1].bias_delta = 0.0;
		for(j = 0; j < OUTPUT_NODE; j++)
		{
			hideLayer[i][HIDE_NODE_COL - 1].weight_delta[j] = 0.0;
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

				sum -= hideLayer[hidelayer_pos][0].bias;
				hideLayer[hidelayer_pos][0].value = sigmoid(sum);
			}

			//forward spread hide -> hide 
			int i,j,k;
			for(j = 1; j< HIDE_NODE_COL; j++)
			{
				for(i = 0; i< HIDE_NODE; i++)
				{
					double sum = 0.0;
					for(k = 0; k< HIDE_NODE; k++)
						sum += hideLayer[k][j - 1].value * hideLayer[k][j - 1].weight[i];
					sum -= hideLayer[i][j].bias;
					hideLayer[i][j].value = sigmoid(sum);
				}
			}

			//forward spread hide -> output
			for(outputlayer_pos = 0; outputlayer_pos < OUTPUT_NODE; outputlayer_pos++)
			{
				double sum = 0.0;
				for(hidelayer_pos= 0; hidelayer_pos < HIDE_NODE; hidelayer_pos++)
				{
					sum += hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].value * hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight[outputlayer_pos];
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

				bias_delta *= d_loss(outputLayer[outputlayer_pos].loss_value); // a(loss) / a(y^) 
				bias_delta *= d_sigmoid(outputLayer[outputlayer_pos].value);   // a(y^)   / a(I)
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
					weight_delta *= d_weight(hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].value);
					weight_delta *= -1;
					hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight_delta[outputlayer_pos] += weight_delta;
				}
			}
			//backward spread output -> hide check hide bias 
			for(hidelayer_pos = 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				double sum_delta = 0;
				double sum_param = 0;
				double sum_all = 0;
				for(k = 0; k < OUTPUT_NODE; k++)
						sum_param += hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight[k];
				for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
				{
				  double sum = 1;
					sum *= outputLayer[outputlayer_pos].d_param;
					sum *= d_value(hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight[outputlayer_pos]);
					sum *= d_sigmoid(hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].value);
					sum_all += sum * (hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight[outputlayer_pos] / sum_param);
					sum *= d_bias();
					sum_delta += sum;
				}
				hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].d_param = (sum_all);
				sum_delta *= -1;
				hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].bias_delta += sum_delta;
			}
		
			if(HIDE_NODE_COL > 1)
			{
				for(j = HIDE_NODE_COL - 2; j >= 0; j--)
				{
					for(i = 0; i < HIDE_NODE; i++)
					{
						double sum_delta = 0;
						double sum_param = 0;
						double sum_all = 0;
						for(k = 0; k < HIDE_NODE; k++)
								sum_param += hideLayer[i][j].weight[k];

						//backward spread hide -> hide check hide weight
						for(k = 0; k < HIDE_NODE; k++)
						{
							double weight_delta = 1;
							weight_delta *= hideLayer[k][j + 1].d_param;
							weight_delta *= d_weight(hideLayer[i][j].value);
							weight_delta *= -1;
							hideLayer[i][j].weight_delta[k] += weight_delta;

							//backward spread hide -> hide check hide bias 
							{
								double sum = 1;
								sum *= hideLayer[k][j + 1].d_param;
								sum *= d_value(hideLayer[i][j].weight[k]);
								sum *= d_sigmoid(hideLayer[i][j].value);
								sum_all += sum * (hideLayer[i][j].weight[k] / sum_param);
								sum *= d_bias();
								sum_delta += sum;
							}
						}
					
						hideLayer[i][j].d_param = sum_all;
						sum_delta *= -1;
						hideLayer[i][j].bias_delta += sum_delta;
					}
				}
			}

			//backward spread hide -> input check input weight
			for(inputLayer_post= 0; inputLayer_post< INPUT_NODE; inputLayer_post++)
			{
				for(hidelayer_pos= 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
				{
					double sum = 1;
					sum *= hideLayer[hidelayer_pos][0].d_param;
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
		printf("\r[%d]times:%d %lf %lf%%  error:%lf", currTrainSample_pos, trainTime, mostTimes, trainTime / mostTimes, error_max);

		//modify
		for(inputLayer_post = 0 ; inputLayer_post < INPUT_NODE; inputLayer_post++)
		{
			for(hidelayer_pos = 0 ; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				inpuLayer[inputLayer_post].weight[hidelayer_pos ] += studyRate * 
					inpuLayer[inputLayer_post].weight_delta[hidelayer_pos] /(double) trainSize;
			}
		}
		int j;
		for(hidelayer_pos = 0 ; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
		{
			for(j = 0 ; j < HIDE_NODE_COL; j++)
			{
				hideLayer[hidelayer_pos][j].bias +=studyRate * hideLayer[hidelayer_pos][j].bias_delta / (double)trainSize;
				for(outputlayer_pos= 0 ; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
				{
					hideLayer[hidelayer_pos][j].weight[outputlayer_pos] += studyRate * 
						hideLayer[hidelayer_pos][j].weight_delta[outputlayer_pos] /(double) trainSize;
				}
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
			sum -= hideLayer[hidelayer_pos][0].bias;
			hideLayer[hidelayer_pos][0].value = sigmoid(sum);
		}
		int j,k;
		for(j = 1; j < HIDE_NODE_COL; j++)
			for(hidelayer_pos = 0; hidelayer_pos < HIDE_NODE; hidelayer_pos++)
			{
				double sum = 0.0;
				for(k = 1; k < HIDE_NODE; k++)
				{
						sum += hideLayer[k][j - 1].value * hideLayer[k][j - 1].weight[hidelayer_pos];
				}
				sum -= hideLayer[hidelayer_pos][j].bias;
				hideLayer[hidelayer_pos][j].value = sigmoid(sum);
			}
		for(outputlayer_pos= 0; outputlayer_pos< OUTPUT_NODE; outputlayer_pos++)
		{
			double sum = 0.0;
			for(hidelayer_pos= 0; hidelayer_pos< HIDE_NODE; hidelayer_pos++)
			{
				sum += hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].value * hideLayer[hidelayer_pos][HIDE_NODE_COL - 1].weight[outputlayer_pos];
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
