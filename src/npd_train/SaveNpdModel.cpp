
#include "common.h"
//#include <mat.h>

void SaveNpdModel(const char* fileName, NpdModel& npdModel, Model& model) {

	if(!npdModel.isEmpty())
	{
		char outName[50] = "";
		strcat(outName, fileName);
		strcat(outName, ".bin");
		FILE* fp = fopen(outName, "wb+");

		int numStages = npdModel.numStages;
		int objSize = npdModel.objSize;
		int numBranchNodes = npdModel.numBranchNodes;
		int numLeafNodes = npdModel.numLeafNodes;

		int		numScales = npdModel.numScales;
		float	scaleFactor = npdModel.scaleFactor;

		double*	stageThreshold = npdModel.stageThreshold;
		int*	treeRoot = npdModel.treeRoot;
		int**	pixel1 = npdModel.pixelx;
		int**	pixel2 = npdModel.pixely;
		unsigned char** cutpoint = npdModel.cutpoint;
		int*	leftChild = npdModel.leftChild;
		int*	rightChild = npdModel.rightChild;
		double*	fit = npdModel.fit;
		int*	winSize = npdModel.winSize;

		//write data
		fwrite(&objSize, sizeof(int), 1, fp);
		fwrite(&numStages, sizeof(int), 1, fp);
		fwrite(&numBranchNodes, sizeof(int), 1, fp);
		fwrite(&numLeafNodes, sizeof(int), 1, fp);
		fwrite(&scaleFactor, sizeof(float), 1, fp);
		fwrite(&numScales, sizeof(int), 1, fp);

		float stageThresholdFloat[5000];
		for (int i = 0; i < numStages; i++)
		{
			stageThresholdFloat[i] = (float)(stageThreshold[i]);
		}
		fwrite(stageThresholdFloat, sizeof(float), numStages, fp);

		fwrite(treeRoot, sizeof(int), numStages, fp);

		for (int i = 0; i < numScales; i++)
		{
			fwrite(pixel1[i], sizeof(int), numBranchNodes, fp);
		}

		for (int i = 0; i < numScales; i++)
		{
			fwrite(pixel2[i], sizeof(int), numBranchNodes, fp);
		}

		for (int i = 0; i < 2; i++)
		{
			fwrite(cutpoint[i], sizeof(unsigned char), numBranchNodes, fp);
		}

		fwrite(leftChild, sizeof(int), numBranchNodes, fp);
		fwrite(rightChild, sizeof(int), numBranchNodes, fp);

		float fitFloat[50000];
		for (int i = 0; i < numLeafNodes; i++)
		{
			fitFloat[i] = fit[i];
		}
		fwrite(fitFloat, sizeof(float), numLeafNodes, fp);

		fwrite(winSize, sizeof(int), numScales, fp);

		fclose(fp);

	}

	if (!model.isEmpty())
	{

		char tempName[100] = "";
		strcat(tempName, fileName);
		strcat(tempName, ".tem");

		FILE* fp = fopen(tempName, "wb+");

		int numStages = model.stages.size();
		fwrite(&numStages, sizeof(int), 1, fp);


		for (int i = 0; i < numStages; i++) {

			//fit
			int fitNum = model.stages[i].fit.size();
			fwrite(&fitNum, sizeof(int), 1, fp);
			float* fit = (float *)malloc(fitNum * sizeof(float));
			for (int j = 0; j < fitNum; j++)
			{
				fit[j] = model.stages[i].fit[j];
			}
			fwrite(fit, sizeof(float), fitNum, fp);

			//cutpoint
			int numBranchNodes = model.stages[i].cutpoint[0].size();
			fwrite(&numBranchNodes, sizeof(int), 1, fp);
			for (int j = 0; j < 2; j++)
			{
				fwrite(&(model.stages[i].cutpoint[j][0]), sizeof(unsigned char), numBranchNodes, fp);
			}

			//leftChild
			fwrite(&(model.stages[i].leftChild[0]), sizeof(int), numBranchNodes, fp);

			//rightChild
			fwrite(&(model.stages[i].rightChild[0]), sizeof(int), numBranchNodes, fp);

			//feaId
			int feaIdNum = model.stages[i].feaId.size();
			fwrite(&feaIdNum, sizeof(int), 1, fp);
			fwrite(&(model.stages[i].feaId[0]), sizeof(int), feaIdNum, fp);

			//depth
			fwrite(&model.stages[i].depth, sizeof(int), 1, fp);

			//thresshold
			float threshold = model.stages[i].threshold;
			fwrite(&threshold, sizeof(float), 1, fp);

			//far
			float far = model.stages[i].far;
			fwrite(&far, sizeof(float), 1, fp);

		}

		fclose(fp);

	}

}
