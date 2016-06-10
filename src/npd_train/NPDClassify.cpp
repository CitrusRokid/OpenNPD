#include "common.h"
#include <opencv2/core/core.hpp>
#include <omp.h>
#include <math.h>
#include <vector>

using namespace std;


void NPDClassify(vector<double>& score, vector<int>& passCount, NpdModel& npdModel, vector<cv::Mat>& noFaceDB, vector<int>& negIndex,int numThreads)
{
	// Set the number of threads
	int numProcs = omp_get_num_procs();

	if (numThreads > numProcs)
		omp_set_num_threads(numProcs);
	else
		omp_set_num_threads(numThreads);

	int objSize = npdModel.objSize;
	int numStages = npdModel.numStages;
	//int numBranchNodes = npdModel.numBranchNodes;
	double* stageThreshold = npdModel.stageThreshold;
	int* treeRoot = npdModel.treeRoot;

	int** Points1 = npdModel.pixelx;
	int** Points2 = npdModel.pixely;

	unsigned  char** ppCutpoint;
	ppCutpoint = npdModel.cutpoint;

	int* pLeftChild = npdModel.leftChild;
	int* pRightChild = npdModel.rightChild;
	double* pFit = npdModel.fit;

	if (noFaceDB[0].channels() != 1 || noFaceDB[0].depth() != CV_8UC1) {
		printf("The data type of the noFaceDB must be uint8 and 1 channel");
		return ;
	}

	int height = noFaceDB[0].rows;
	int width = noFaceDB[0].cols;

	if (height != objSize || width != objSize)
	{
		printf("height=%d, width=%d, objSize=%d\n", height, width, objSize);
		printf("The patch dimensions disagree.");
		return ;
	}

	//int numPixels = objSize * objSize;
	int numRects = 1;

	numRects = noFaceDB.size();

	int numIndex = 0;

	numIndex = negIndex.size();
	

	if (numIndex > numRects) {
		printf("Error index.");
		return ;
	}

	int numRun = numRects;
	if (numIndex > 0) numRun = numIndex;

	if (numRects >= pow(2.0, 31)) {
		printf("The number of samples is out of the int32 range (required by OpenMP).");
	}

	score.resize(numRun);
	passCount.resize(numRun);

	//原来是以列展开，转换成以行展开
	vector<int> offset(width*height);
	int p1 = 0, p2 = 0, gap = width;
	
	for (int j = 0; j < width; j++) // column coordinate
	{
		p2 = j;
		for (int i = 0; i < height; i++) // row coordinate
		{
			offset[p1++] = p2;
			p2 += gap;
		}
	}

#pragma omp parallel for

	// process each subwindow (rect)
	for (int i = 0; i < numRun; i++)
	{
		bool run = true;

		int index;
		if (numIndex > 0)
		{
			if (negIndex[i] >= numRects) {
				printf("Error index.");
				run = false;
			}
			index = negIndex[i];
		}
		else 
		{
			index = i;
		}

		if (run) {

			int treeIndex = 0;
			double Fx = 0;
			int count = 0;

			unsigned char* I = noFaceDB[index].data;

			// test each tree classifier
			for (int s = 0; s < numStages; s++)
			{
				int node = treeRoot[treeIndex];

				// test the current tree classifier
				while (node > -1) // branch node
				{
					unsigned char p1 = I[offset[Points1[0][node]]];
					unsigned char p2 = I[offset[Points2[0][node]]];
					unsigned char fea = npd::npdTable[p2][p1];

					if (fea < ppCutpoint[0][node] || fea > ppCutpoint[1][node]) node = pLeftChild[node];
					else node = pRightChild[node];
				}

				// leaf node
				node = -node - 1;
				Fx = Fx + pFit[node];
				treeIndex++;

				if (Fx < stageThreshold[s]) break; // negative samples
				else count++;
			}

			score[i] = Fx;
			passCount[i] = count;
		}
	}
}
