#include "common.h"
#include <iostream>
#include "time.h"

using namespace std;

int CalcTreeDepth(vector<int>& leftChild, vector<int>& rightChild, int node = 0) {

	int ld, rd;

	if (node > int(leftChild.size()) - 1 ) {
		return 0;
	}

	if (leftChild[node] < 0) {
		ld = 0;
	}
	else {
		ld = CalcTreeDepth(leftChild, rightChild, leftChild[node]);
	}

	if (rightChild[node] < 0) {
		rd = 0;
	}
	else {
		rd = CalcTreeDepth(leftChild, rightChild, rightChild[node]);
	}

	return (max(ld, rd) + 1);
}

vector<double> CalcWeight(vector<double>& fx, int flag, double maxW) {
	int n = fx.size();

	vector<double> weight(n);

	double sum = 0;

	for (int i = 0; i < n; i++) {
		weight[i] = min(exp(-flag * fx[i]), maxW);
		sum += weight[i];
	}

	if (sum == 0) {
		for (int i = 0; i < n; i++) {
			weight[i] = 1./n;
		}
	}
	else{
		for (int i = 0; i < n; i++) {
			weight[i] /= sum;
		}
	}

	return weight;
}

//function to test the learned soft cascade and DQT weak classifier based Gentle AdaBoost classifier.
void TestGAB(vector<double>& Fx, vector<int>& passCount, Model& model, cv::Mat& X) {
	int n = X.rows;

	Fx.clear();
	passCount.clear();
	Fx.resize(n);
	passCount.resize(n);


	for (int i = 0; i < n; i++) {	//for each image

		bool run = true;

		for (int j = 0; j < int(model.stages.size()) && run; j++) {	//for each weak classifier

			double fx = TestDQT(model.stages[j], X.row(i));

			Fx[i] += fx;
			if (Fx[i] >= model.stages[j].threshold) {
				passCount[i]++;
			}
			else {
				run = false;
			}
		}
	}

};

void LearnGAB(Model& model, cv::Mat& faceFea, cv::Mat& nonfaceFea,NpdModel& npdModel, vector<cv::Mat>& nonFaceRects,  BoostOpt& option) {
	int		treeLevel = 4;			// the maximal depth of the DQT trees to be learned
	int		maxNumWeaks = 1000;		// maximal number of weak classifiers to be learned
	double	minDR = 1.0;			// minimal detection rate required
	double	maxFAR = 1e-16;			// maximal FAR allowed; stop the training if reached
	int		minSamples = 1000;		// minimal samples required to continue training
	double	minNegRatio = 0.2;		// minimal fraction of negative samples required to remain,

									// w.r.t.the total number of negative samples.This is a signal of
									// requiring new negative sample bootstrapping.Also used to avoid
									// overfitting.
	double	trimFrac = 0.05;		// weight trimming in AdaBoost
	double	samFrac = 1.0;			// the fraction of samples randomly selected in each iteration
									// for training; could be used to avoid overfitting.
	double	minLeafFrac = 0.01;		// minimal sample fraction w.r.t.the total number of
									// samples required in each leaf node.This is used to avoid overfitting.
	int		minLeaf = 100;			// minimal samples required in each leaf node.This is used to avoid overfitting.
	double	maxWeight = 100;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
	int		numThreads = 24;		// the number of computing threads in tree learning

	if (true) {
		maxWeight = option.maxWeight;
		samFrac = option.samFrac;
		treeLevel = option.treeLevel;
		minLeafFrac = option.minLeafFrac;
		minLeaf = option.minLeaf;
		maxNumWeaks = option.maxNumWeak;
		maxFAR = option.maxFAR;
		minNegRatio = option.minNegRatio;
		minSamples = option.minSamples;
		trimFrac = option.trimFrac;
		numThreads = option.numThreads;
		minDR = option.minDR;
	}

	int nPos = faceFea.rows;
	int nNeg = nonfaceFea.rows;

	clock_t t0 = clock();

	vector<double> posFx, negFx;
	vector<int> passCount;

	cv::Mat posX(nPos, faceFea.cols, CV_8UC1, faceFea.data);
	cv::Mat negX(nNeg, nonfaceFea.cols, CV_8UC1, nonfaceFea.data);

	vector<double> newPosFx, newNegFx, posW, negW;
	vector<int> posPassIndex, negPassIndex;

	int startIter;

	if (!model.isEmpty()) {
		cout << "Test the current model..." << endl;
		int T = model.stages.size();

		//test pos samples
		TestGAB(posFx, passCount, model, faceFea);

		nPos = 0;
		for (int i = 0; i < int(passCount.size()); i++) {
			if (passCount[i] == T) {
				//faceFea.row(i).copyTo(posX.row(nPos++));
				posPassIndex.push_back(i);
				nPos++;
				newPosFx.push_back(posFx[i]);
			}
		}

		if (nPos < faceFea.rows) {
			cout << "Warning: some positive samples cannot pass all stages. pass rate is "
				<< ((nPos)/(double)(faceFea.rows)) << endl;
		}

		//test neg samples
		passCount.clear();
		TestGAB(negFx, passCount, model, nonfaceFea);

		nNeg = 0;
		for (int i = 0; i < int(passCount.size()); i++) {
			if (passCount[i] == T) {
				//nonfaceFea.row(i).copyTo(negX.row(nNeg++));
				negPassIndex.push_back(i);
				nNeg++;
				newNegFx.push_back(negFx[i]);
			}
		}

		if (nNeg < nonfaceFea.rows) {
			cout << "Warning: some negative samples cannot pass all stages, pass rate is "
				 << ((nNeg)/(double)(nonfaceFea.rows)) << endl;
		}

		vector<double> newPosW = CalcWeight(newPosFx, 1, maxWeight);
		vector<double> newNegW = CalcWeight(newNegFx, -1, maxWeight);

		posW.resize(faceFea.rows);
		for (int i = 0; i < int(newPosW.size()); i++)
		{
			posW[posPassIndex[i]] = newPosW[i];
		}

		negW.resize(nonfaceFea.rows);
		for (int i = 0; i < int(newNegW.size()); i++)
		{
			negW[negPassIndex[i]] = newNegW[i];
		}

		startIter = T;

		cout << (clock() - t0) << " seconds." << endl;

	}
	else {
		posW = vector<double>(nPos);
		negW = vector<double>(nNeg);
		posFx = vector<double>(nPos);
		negFx = vector<double>(nNeg);

		for (int i = 0; i < nPos; i++) {
			posW[i] = 1. / nPos;
			posFx[i] = 0;
		}

		for (int i = 0; i < nNeg; i++) {
			negW[i] = 1. / nNeg;
			negFx[i] = 0;
		}

		posPassIndex.resize(nPos);
		for (int i = 0; i < nPos; i++) {
			posPassIndex[i] = i;
		}

		negPassIndex.resize(nNeg);
		for (int i = 0; i < nNeg; i++) {
			negPassIndex[i] = i;
		}

		startIter = 0;

	}

	int nNegPass = nNeg;

	cout << "\nStart to train adaboost, nPos=" << nPos << ", nNeg=" << nNeg << endl << endl;
	int t;
	for (t = startIter; t < maxNumWeaks; t++) {
		if (nNegPass < minSamples) {
			cout << endl << "No enough negative samples. The Adaboost learning terninates at iteration "
				<< t << ". nNegPass = " << nNegPass << endl;
			break;
		}

		//posIndex
		int nPosSam = max((int)round(nPos*samFrac), minSamples);
		vector<int> posIndex(nPos);
		for (int i = 0; i < nPos; i++) {
			posIndex[i] = posPassIndex[i];
		}
		random_shuffle(posIndex.begin(), posIndex.end());
		posIndex.resize(nPosSam);

		//negIndex
		int nNegSam = max((int)round(nNegPass*samFrac), minSamples);
		vector<int> negIndex(nNegPass);
		for (int i = 0; i < nNegPass; i++) {
			negIndex[i] = negPassIndex[i];
		}
		random_shuffle(negIndex.begin(), negIndex.end());
		negIndex.resize(nNegSam);

		//trim pos weight
		vector<int> trimedPosIndex;
		vector<int> posIndexSort(posIndex);
		IndexSort(posIndexSort, posW);

		double cumsum = 0;
		int k = 0;
		for (int i = 0; i < int(posIndexSort.size()); i++) {
			cumsum += posW[posIndexSort[i]];
			if (cumsum >= trimFrac) {
				k = i;
				break;
			}
		}
		k = min(k, nPosSam - minSamples);

		double trimWeight = posW[posIndexSort[k]];

		for (int i = 0; i < int(posIndex.size()); i++) {
			if (posW[posIndex[i]] >= trimWeight)
				trimedPosIndex.push_back(posIndex[i]);
		}

		posIndex.swap(trimedPosIndex);

		//trim neg weight
		vector<int> trimedNegIndex;
		vector<int> negIndexSort(negIndex);
		IndexSort(negIndexSort, negW);

		cumsum = 0;
		//int k;
		for (int i = 0; i < int(negIndexSort.size()); i++) {
			cumsum += negW[negIndexSort[i]];
			if (cumsum >= trimFrac) {
				k = i;
				break;
			}
		}
		k = min(k, nNegSam - minSamples);

		trimWeight = negW[negIndexSort[k]];

		for (int i = 0; i < int(negIndex.size()); i++) {
			if (negW[negIndex[i]] >= trimWeight)
				trimedNegIndex.push_back(negIndex[i]);
		}

		negIndex.swap(trimedNegIndex);

		nPosSam = posIndex.size();
		nNegSam = negIndex.size();

		int minLeaf_t = max( (int)round((nPosSam+nNegSam)*minLeafFrac), minLeaf);

		printf("\nIter %d: nPos=%d, nNeg=%d, ", t, nPosSam, nNegSam);

		vector<int> feaId, leftChild, rightChild;
		vector< vector<uchar> > cutpoint;
		vector<double> fit;
		double minCost;

		LearnDQT(feaId, leftChild, rightChild, cutpoint, fit, minCost, posX, negX, posW, negW, posFx, negFx, posIndex,
				negIndex, treeLevel, minLeaf_t, numThreads);

		if (int(feaId.size()) == 0)
		{
			printf("\n\nNo available features to satisfy the split. The AdaBoost learning terminates.\n");
			break;
		}

		model.stages.push_back(Stage());	//add a new stage
		model.stages[t].feaId = feaId;
		model.stages[t].cutpoint = cutpoint;
		model.stages[t].leftChild = leftChild;
		model.stages[t].rightChild = rightChild;
		model.stages[t].fit = fit;
		model.stages[t].depth = CalcTreeDepth(leftChild, rightChild);

		vector<double> v;
		for (int i = 0; i < int(posPassIndex.size()); i++) {
			posFx[posPassIndex[i]] += TestDQT(model.stages[t], posX.row(posPassIndex[i]));
			v.push_back(posFx[posPassIndex[i]]);
		}

		for (int i = 0; i < int(negPassIndex.size()); i++) {
			negFx[negPassIndex[i]] += TestDQT(model.stages[t], negX.row(negPassIndex[i]));
		}

		sort(v.begin(), v.end());
		int index = max((int)floor(nPos*(1-minDR)), 0);
		model.stages[t].threshold = v[index];

		vector<int> temNegPassIndex;
		for (int i = 0; i < int(negPassIndex.size()); i++) {
			if (negFx[negPassIndex[i]] >= model.stages[t].threshold) {
				temNegPassIndex.push_back(negPassIndex[i]);
			}
		}

		negPassIndex.swap(temNegPassIndex);
		model.stages[t].far = negPassIndex.size() / (double)nNegPass;
		nNegPass = negPassIndex.size();

		double FAR = 1;
		for (int i = 0; i < int(model.stages.size()); i++) {
			FAR *= model.stages[i].far;
		}

		//TODO: calc aveEval

		printf("FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n",
			model.stages[t].far * 100, FAR, model.stages[t].depth, int(feaId.size()), int(model.stages[t].feaId.size()), minCost);
		printf("\t\tnNegPass=%d, aveEval=%.3f, time=%.0fs, meanT=%.3fs.\n", nNegPass, 0.0/*aveEval*/, double(clock() - t0), double(clock()- t0) / (t - startIter + 1));

		if (FAR <= maxFAR) {
			printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, FAR * 100);
			break;
		}

		if (nNegPass < nNeg * minNegRatio || nNegPass < minSamples) {
			printf("\n\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t, nNegPass);
			break;
		}

		vector<double> temposW;
		vector<double> temposFx;
		for (int i = 0; i < int(posPassIndex.size()); i++) {
			temposFx.push_back(posFx[posPassIndex[i]]);
		}
		temposW = CalcWeight(temposFx, 1, maxWeight);
		for (int i = 0; i < int(posPassIndex.size()); i++) {
			posW[posPassIndex[i]] = temposW[i];
		}

		vector<double> temNegW;
		vector<double> temNegFx;
		for (int i = 0; i < int(negPassIndex.size()); i++) {
			temNegFx.push_back(negFx[negPassIndex[i]]);
		}
		temNegW = CalcWeight(temNegFx, -1, maxWeight);
		for (int i = 0; i < int(negPassIndex.size()); i++) {
			negW[negPassIndex[i]] = temNegW[i];
		}
	}

	printf("\n\nThe adaboost training is finished. Total time: %.0f seconds. Mean time: %.3f seconds.\n\n", double(clock() - t0), double(clock() - t0) / (t - startIter + 1));

};
