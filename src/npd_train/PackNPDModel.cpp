#include "common.h"
#include <iostream>

using namespace std;

NpdModel PackNPDModel(Model &model, int objSize, double scaleFactor, int maxFace) 
{
	
	int numStages = model.stages.size();
	if (numStages == 0) {
		return NpdModel();
	}

	vector<int> feaId;
	vector<int> treeRoot;
	int treeRootIndex = 0;
	for (int i = 0; i < numStages; i++) {
		feaId.insert(feaId.end(), model.stages[i].feaId.begin(), model.stages[i].feaId.end());
		treeRoot.push_back(treeRootIndex);
		treeRootIndex += model.stages[i].feaId.size();
	}
	int numBranchNodes = feaId.size();

	vector<double> fit;
	vector<int> leafRoot;
	int leafRootIndex = 0;
	for (int i = 0; i < numStages; i++) {
		fit.insert(fit.end(), model.stages[i].fit.begin(), model.stages[i].fit.end());
		leafRoot.push_back(leafRootIndex);
		leafRootIndex += model.stages[i].fit.size();
	}
	int numLeafNodes = fit.size();

	vector<double> stageThreshold;
	for (int i = 0; i < numStages; i++) {
		stageThreshold.push_back(model.stages[i].threshold);
	}

	int numPixels = objSize * objSize;
	vector<int> pt1;
	vector<int> pt2;
	vector<int> pixel1;
	vector<int> pixel2;
	for (int x = 0; x < numPixels; x++) {
		for (int y = x + 1; y < numPixels; y++) {
			pt1.push_back(x);
			pt2.push_back(y);
		}
	}
	for (int i = 0; i < int(feaId.size()); i++) {
		pixel1.push_back(pt1[feaId[i]]);
		pixel2.push_back(pt2[feaId[i]]);
	}

	vector< vector<uchar> > cutpoint;
	cutpoint.resize(2);
	for (int i = 0; i < numStages; i++) {
		cutpoint[1].insert(cutpoint[1].end(), model.stages[i].cutpoint[1].begin(), model.stages[i].cutpoint[1].end());
		cutpoint[0].insert(cutpoint[0].end(), model.stages[i].cutpoint[0].begin(), model.stages[i].cutpoint[0].end());
	}

	vector<int> leftChild;
	vector<int> rightChild;
	for (int i = 0; i < numStages; i++) {
		leftChild.insert(leftChild.end(), model.stages[i].leftChild.begin(), model.stages[i].leftChild.end());
		rightChild.insert(rightChild.end(), model.stages[i].rightChild.begin(), model.stages[i].rightChild.end());
	}

	for (int i = 1; i < numStages; i++) {
		int branchOffset = treeRoot[i];
		int leafOffset = leafRoot[i];

		for (int index = treeRoot[i]; index < treeRoot[i] + int(model.stages[i].feaId.size()) ; index++) {
			
			if (leftChild[index] < 0) {
				leftChild[index] -= leafOffset;
			}
			else {
				leftChild[index] += branchOffset;
			}

			if (rightChild[index] < 0) {
				rightChild[index] -= leafOffset;
			}
			else {
				rightChild[index] += branchOffset;
			}

		}

	}

	NpdModel outModel;

	outModel.objSize = objSize;
	outModel.numStages = numStages;
	outModel.numBranchNodes = numBranchNodes;
	outModel.numLeafNodes = numLeafNodes;
	
	outModel.stageThreshold = (double*)malloc(numStages * sizeof(double));
	outModel.treeRoot = (int*)malloc(numStages * sizeof(int));

	for (int i = 0; i < numStages; i++)
	{
		outModel.stageThreshold[i] = stageThreshold[i];
		outModel.treeRoot[i] = treeRoot[i];
	}
	
	outModel.cutpoint = (unsigned char**)malloc(2 * sizeof(unsigned char*));
	for (int i = 0; i < 2; i++) 
	{
		outModel.cutpoint[i] = (unsigned char*)malloc(numBranchNodes * sizeof(unsigned char));
	}
	outModel.leftChild = (int*)malloc(numBranchNodes * sizeof(int));
	outModel.rightChild = (int*)malloc(numBranchNodes * sizeof(int));

	for (int i = 0; i < numBranchNodes; i++)
	{
		outModel.cutpoint[0][i] = cutpoint[0][i];
		outModel.cutpoint[1][i] = cutpoint[1][i];
		outModel.leftChild[i] = leftChild[i];
		outModel.rightChild[i] = rightChild[i];
	}

	outModel.fit = (double*)malloc(numLeafNodes * sizeof(double));

	for (int i = 0; i < numLeafNodes; i++) {
		outModel.fit[i] = fit[i];
	}

	//TODO::aveEval
	printf("#weaks: %d\n", numStages);
	printf("#features: %d\n", numBranchNodes);

	if (scaleFactor > 1) 
	{
		int numScales = 0;
		int size = objSize;
		double s = 1;

		vector<int> winSize;
		vector<double> scale;

		while (size < maxFace) 
		{
			numScales++;
			winSize.push_back(size);
			scale.push_back( (size/(double)objSize) );

			s *= scaleFactor;
			size =  round(objSize * s);

		}

		outModel.pixelx = (int**)malloc(numScales * sizeof(int*));
		outModel.pixely = (int**)malloc(numScales * sizeof(int*));

		for (int i = 0; i < numScales; i++)
		{
			outModel.pixelx[i] = (int*)malloc(numBranchNodes * sizeof(int));
			outModel.pixely[i] = (int*)malloc(numBranchNodes * sizeof(int));

			if (i == 0)
			{
				for (int j = 0; j < numBranchNodes; j++)
				{
					outModel.pixelx[i][j] = pixel1[j];
					outModel.pixely[i][j] = pixel2[j];
				}
			}
			else
			{
				for (int j = 0; j < numBranchNodes; j++)
				{
					outModel.pixelx[i][j] = round( floor(pixel1[j] / objSize) * scale[i] ) * winSize[i] + round( (pixel1[j] % objSize) * scale[i] );
					outModel.pixely[i][j] = round( floor(pixel2[j] / objSize) * scale[i] ) * winSize[i] + round( (pixel2[j] % objSize) * scale[i] );
				}
			}

		}

		outModel.scaleFactor = scaleFactor;
		outModel.numScales = numScales;

		outModel.winSize = (int*)malloc(numScales * sizeof(int));
		for (int i = 0; i < numScales; i++) 
		{
			outModel.winSize[i] = winSize[i];
		}

	}

	return outModel;

}
