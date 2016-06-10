#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <omp.h>
#include <math.h>
#include <vector>

using namespace std;

void WeightHist(const cv::Mat& X, vector<double>& W, vector<int>& index, int n, int count[256], double wHist[256])
{
	memset(wHist, 0, 256 * sizeof(double));

	for (int j = 0; j < n; j++)
	{
		unsigned char bin = X.ptr(index[j])[0];
		count[bin]++;
		wHist[bin] += W[index[j]];
	}
}

double LearnQuadStump(cv::Mat &posX, cv::Mat &negX, vector<double>& posW, vector<double>& negW, vector<double>& posFx, vector<double>& negFx,
	vector<int>& posIndex, vector<int>& negIndex, int nPos, int nNeg, int minLeaf, int numThreads, double parentFit,
	int &feaId, unsigned char(&cutpoint)[2], double(&fit)[2])
{
	double w = 0;

	for (int i = 0; i < nPos; i++)
		w += posW[posIndex[i]];
	double minCost = w * (parentFit - 1) * (parentFit - 1);

	w = 0;
	for (int i = 0; i < nNeg; i++)
		w += negW[negIndex[i]];
	minCost += w * (parentFit + 1) * (parentFit + 1);

	feaId = -1;
	if (nPos == 0 || nNeg == 0 || nPos + nNeg < 2 * minLeaf)
		return minCost;

	int feaDims = (int)posX.cols;
	minCost = 1e16f;

	omp_set_num_threads(numThreads);
#pragma omp parallel for

	// process each dimension
	for (int i = 0; i < feaDims; i++)
	{
		int count[256];
		double posWHist[256];
		double negWHist[256];

		memset(count, 0, 256 * sizeof(int));

		WeightHist(posX.col(i), posW, posIndex, nPos, count, posWHist);
		WeightHist(negX.col(i), negW, negIndex, nNeg, count, negWHist);

		double posWSum = 0;
		double negWSum = 0;

		for (int bin = 0; bin < 256; bin++)
		{
			posWSum += posWHist[bin];
			negWSum += negWHist[bin];
		}

		int totalCount = nPos + nNeg;
		//double wSum = posWSum + negWSum;

		double minMSE = 3.4e38f;
		int thr0 = -1, thr1 = -1;
		double fit0 = 0, fit1 = 0;

		for (int v = 0; v < 256; v++) // lower threshold
		{
			int rightCount = 0;
			double rightPosW = 0;
			double rightNegW = 0;

			for (int u = v; u < 256; u++) // upper threshold
			{
				rightCount += count[u];
				rightPosW += posWHist[u];
				rightNegW += negWHist[u];

				if (rightCount < minLeaf) continue;

				int leftCount = totalCount - rightCount;
				if (leftCount < minLeaf) break;

				double leftPosW = posWSum - rightPosW;
				double leftNegW = negWSum - rightNegW;

				double leftFit, rightFit;

				if (leftPosW + leftNegW <= 0) leftFit = 0.0f;
				else leftFit = (leftPosW - leftNegW) / (leftPosW + leftNegW);

				if (rightPosW + rightNegW <= 0) rightFit = 0.0f;
				else rightFit = (rightPosW - rightNegW) / (rightPosW + rightNegW);

				//Mean squared error ???
				double leftMSE = leftPosW * (leftFit - 1) * (leftFit - 1) + leftNegW * (leftFit + 1) * (leftFit + 1);
				double rightMSE = rightPosW * (rightFit - 1) * (rightFit - 1) + rightNegW * (rightFit + 1) * (rightFit + 1);

				double mse = leftMSE + rightMSE;

				if (mse < minMSE)
				{
					minMSE = mse;
					thr0 = v;
					thr1 = u;
					fit0 = leftFit;
					fit1 = rightFit;
				}
			}
		}

		if (thr0 == -1) continue;

		if (minMSE <= minCost)
		{
#pragma omp critical // modify the record by a single thread
		{
			minCost = minMSE;
			feaId = i;
			cutpoint[0] = (unsigned char)thr0;
			cutpoint[1] = (unsigned char)thr1;
			fit[0] = fit0;
			fit[1] = fit1;
		}
		}
	} // end omp parallel for feaDims

	return minCost;
}

double MyLearnDQT(cv::Mat &posX, cv::Mat &negX, vector<double>& posW, vector<double>& negW, vector<double>& posFx, vector<double>& negFx,
	vector<int>& posIndex, vector<int>& negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, double parentFit,
	vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<double> &fit)
{
	int _feaId;
	unsigned char _cutpoint[2];
	double _fit[2];

	//select feature
	double minCost = LearnQuadStump(posX, negX, posW, negW, posFx, negFx, posIndex, negIndex, nPos, nNeg, minLeaf, numThreads, parentFit,
		_feaId, _cutpoint, _fit);

	if (_feaId < 0) return minCost;		//if select no feature, terminate

	feaId.push_back(_feaId);
	cutpoint.push_back(vector<unsigned char>(_cutpoint, _cutpoint + 2));
	leftChild.push_back(-1);
	rightChild.push_back(-2);

	if (treeLevel <= 1)
	{
		fit.push_back(_fit[0]);
		fit.push_back(_fit[1]);
		return minCost;
	}

	int nPos1 = 0, nNeg1 = 0, nPos2 = 0, nNeg2 = 0;
	vector<int> posIndex1, posIndex2, negIndex1, negIndex2;

	for (int j = 0; j < nPos; j++)
	{
		if ( posX.at<uchar>(size_t(posIndex[j]), size_t(_feaId)) < _cutpoint[0] || posX.at<uchar>(size_t(posIndex[j]), size_t(_feaId)) > _cutpoint[1])
		{
			posIndex1.push_back(posIndex[j]);	//the pos judge as left
			nPos1++;
		}
		else
		{
			posIndex2.push_back(posIndex[j]);	//the pos judge as right
			nPos2++;
		}
	}

	for (int j = 0; j < nNeg; j++)
	{
		if (negX.at<uchar>(size_t(negIndex[j]), size_t(_feaId)) < _cutpoint[0] || negX.at<uchar>(size_t(negIndex[j]), size_t(_feaId)) > _cutpoint[1])
		{
			negIndex1.push_back(negIndex[j]);	//the neg judge as left
			nNeg1++;
		}
		else
		{
			negIndex2.push_back(negIndex[j]);	//the neg judge as right
			nNeg2++;
		}
	}

	vector<int> feaId1, feaId2, leftChild1, leftChild2, rightChild1, rightChild2;
	vector< vector<unsigned char> > cutpoint1, cutpoint2;
	vector<double> fit1, fit2;

	double minCost1 = MyLearnDQT(posX, negX, posW, negW, posFx, negFx, posIndex1, negIndex1, nPos1, nNeg1, treeLevel - 1, minLeaf, numThreads, _fit[0],
		feaId1, cutpoint1, leftChild1, rightChild1, fit1);

	double minCost2 = MyLearnDQT(posX, negX, posW, negW, posFx, negFx, posIndex2, negIndex2, nPos2, nNeg2, treeLevel - 1, minLeaf, numThreads, _fit[1],
		feaId2, cutpoint2, leftChild2, rightChild2, fit2);

	if (feaId1.empty() && feaId2.empty())	//select no feature in sub sets , terminate
	{
		fit.push_back(_fit[0]);
		fit.push_back(_fit[1]);
		return minCost;
	}

	if (minCost1 + minCost2 >= minCost)
	{
		fit.push_back(_fit[0]);
		fit.push_back(_fit[1]);
		return minCost;
	}

	minCost = minCost1 + minCost2;

	if (feaId1.empty())
	{
		fit.push_back(_fit[0]);
	}
	else
	{
		feaId.insert(feaId.end(), feaId1.begin(), feaId1.end());
		cutpoint.insert(cutpoint.end(), cutpoint1.begin(), cutpoint1.end());
		fit = fit1;

		for (int i = 0; i < int(leftChild1.size()); i++)
		{
			if (leftChild1[i] >= 0) leftChild1[i]++;
			if (rightChild1[i] >= 0) rightChild1[i]++;
		}

		leftChild[0] = 1;
		leftChild.insert(leftChild.end(), leftChild1.begin(), leftChild1.end());
		rightChild.insert(rightChild.end(), rightChild1.begin(), rightChild1.end());
	}

	int numBranchNodes = (int)feaId.size();
	int numLeafNodes = (int)fit.size();

	if (feaId2.empty())
	{
		fit.push_back(_fit[1]);
		rightChild[0] = -(numLeafNodes + 1);
	}
	else
	{
		feaId.insert(feaId.end(), feaId2.begin(), feaId2.end());
		cutpoint.insert(cutpoint.end(), cutpoint2.begin(), cutpoint2.end());
		fit.insert(fit.end(), fit2.begin(), fit2.end());

		for (int i = 0; i < int(leftChild2.size()); i++)
		{
			if (leftChild2[i] >= 0) leftChild2[i] += numBranchNodes;
			else leftChild2[i] -= numLeafNodes;

			if (rightChild2[i] >= 0) rightChild2[i] += numBranchNodes;
			else rightChild2[i] -= numLeafNodes;
		}

		leftChild.insert(leftChild.end(), leftChild2.begin(), leftChild2.end());
		rightChild[0] = numBranchNodes;
		rightChild.insert(rightChild.end(), rightChild2.begin(), rightChild2.end());
	}

	return minCost;
}

void LearnDQT(vector<int>& feaId, vector<int>& leftChild,vector<int>& rightChild,
		vector< vector<uchar> >& cutpoint, vector<double>& fit, double& minCost,
		cv::Mat posX, cv::Mat negX, vector<double> posW, vector<double> negW,
		vector<double> posFx, vector<double> negFx, vector<int> posIndex, vector<int> negIndex,
		int treeLevel, int minLeaf, int numThreads)
{

	// Set the number of threads
	int numProcs = omp_get_num_procs();
	numThreads = numProcs < numThreads ? numProcs:numThreads;

	/* Validate the input data type */
	if (posX.depth()!=CV_8U || negX.depth()!=CV_8U )
	{
		printf("The data types of the input arrays are not acceptable.");
	}

	//int nTotalPos = posX.rows;
	//int nTotalNeg = negX.rows;
	//int feaDims = posX.cols;
	int nPos = posIndex.size();
	int nNeg = negIndex.size();

	vector< vector<unsigned char> > cutpointArray;

	minCost = MyLearnDQT(posX, negX, posW, negW, posFx, negFx, posIndex, negIndex,
		nPos, nNeg, treeLevel, minLeaf, numThreads, 0,
		feaId, cutpointArray, leftChild, rightChild, fit);

	int numBranchNodes = (int)feaId.size();
	//int numLeafNodes = (int)fit.size();

	cutpoint.resize(2);
	cutpoint[0].resize(numBranchNodes);
	cutpoint[1].resize(numBranchNodes);
	for (int i = 0; i < numBranchNodes; i++)
	{
		cutpoint[0][i] = cutpointArray[i][0];
		cutpoint[1][i] = cutpointArray[i][1];
	}

}
