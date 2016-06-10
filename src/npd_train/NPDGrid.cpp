#include "common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <math.h>
#include <vector>

using namespace std;

vector<cv::Mat> NPDGrid(NpdModel& npdModel, cv::Mat& img, int minFace, int maxFace, int numThreads)
{

	// Set the number of threads
	int numProcs = omp_get_num_procs();

	if (numThreads > numProcs) {
		omp_set_num_threads(numProcs);
	}
	else {
		omp_set_num_threads(numThreads);
	}

	// get the NPD detector
	int objSize = npdModel.objSize;
	int numStages = npdModel.numStages;
	//int numBranchNodes = npdModel.numBranchNodes;
	double* stageThreshold = npdModel.stageThreshold;
	int* treeRoot = npdModel.treeRoot;

	int numScales = npdModel.numScales;
	int** ppPoints1 = npdModel.pixelx;
	int** ppPoints2 = npdModel.pixely;

	unsigned char** ppCutpoint;
	ppCutpoint = npdModel.cutpoint;

	int* leftChild = npdModel.leftChild;
	int* rightChild = npdModel.rightChild;
	double* fit = npdModel.fit;

	int* winSize = npdModel.winSize;

	int height = img.rows;
	int width = img.cols;
	const unsigned char *I = img.data;

	minFace = max(minFace, objSize);
	maxFace = min(maxFace, min(height, width));

	if (min(height, width) < minFace)
	{
		// create a structure vector for the output data
		return vector<cv::Mat>();
	}

	// containers for the detected faces
	vector<double> row, col, size, score;
	vector<cv::Mat> mats;

	for (int k = 0; k < numScales; k++) // process each scale
	{
		if (winSize[k] < minFace) continue;
		else if (winSize[k] > maxFace) break;

		int winStep = (int)floor(winSize[k] * 0.5);

		//原来是以列展开，转换成以行展开
		vector<int> offset(winSize[k] * winSize[k]);
		int p1 = 0, p2 = 0, gap = winSize[k];

		for (int j = 0; j < winSize[k]; j++) // column coordinate
		{
			p2 = j;
			for (int i = 0; i < winSize[k]; i++) // row coordinate
			{
				offset[p1++] = p2;
				p2 += gap;
			}
		}

		int colMax = width - winSize[k] + 1;
		int rowMax = height - winSize[k] + 1;

#pragma omp parallel for

		// process each subwindow
		for (int r = 0; r < rowMax; r += winStep) // slide in row
		{
			const unsigned char *pPixel = I + r * width;

			for (int c = 0; c < colMax; c += winStep, pPixel += winStep) // slide in column
			{
				int treeIndex = 0;
				double _score = 0;
				int s;

				//=======================

				cv::Mat rectMat(winSize[k], winSize[k], CV_8UC1);
				cv::Mat temMat(img, cv::Rect(c, r, winSize[k], winSize[k]));
				temMat.copyTo(rectMat);

				//cv::resize(rectMat, rectMat, cv::Size(objSize, objSize));
				const unsigned char * pixelPtr = rectMat.data;

				//=======================

				// test each tree classifier
				for (s = 0; s < numStages; s++)
				{
					int node = treeRoot[treeIndex];

					// test the current tree classifier
					while (node > -1) // branch node
					{

						//====================================================
						unsigned char p1 = pixelPtr[offset[ppPoints1[k][node]]];
						unsigned char p2 = pixelPtr[offset[ppPoints2[k][node]]];
						unsigned char fea = npd::npdTable[p2][p1];
						//====================================================

						if (fea < ppCutpoint[0][node] || fea > ppCutpoint[1][node])
						{
							node = leftChild[node];
						}
						else
						{
							node = rightChild[node];
						}

					}

					// leaf node
					node = -node - 1;
					_score = _score + fit[node];
					treeIndex++;

					if (_score < stageThreshold[s])
					{
						break; // negative samples
					}

				}

				if (s == numStages) // a face detected
				{

					cv::Mat scaleRect;
					npd::resizeMat(rectMat, scaleRect, objSize);

#pragma omp critical // modify the record by a single thread
					{
						mats.push_back(scaleRect);
					}
				}
			}
		}
	}

	int numFaces = (int)mats.size();

	if (numFaces == 0) return mats;

	return mats;
}
