
#include "common.h"

bool LoadNpdModel(const char* file, NpdModel& npdModel, Model& model)
{

	//npdModel
	char npdModelName[100] = "";
	strcat(npdModelName, file);
	strcat(npdModelName, ".bin");
	FILE* fp = fopen(npdModelName, "rb");

	if (fp != NULL)
	{
		
		fread(&(npdModel.objSize), sizeof(int), 1, fp);
		fread(&(npdModel.numStages), sizeof(int), 1, fp);
		fread(&(npdModel.numBranchNodes), sizeof(int), 1, fp);
		fread(&(npdModel.numLeafNodes), sizeof(int), 1, fp);
		fread(&(npdModel.scaleFactor), sizeof(float), 1, fp);
		fread(&(npdModel.numScales), sizeof(int), 1, fp);

		float* stageThreshold = (float *)malloc(npdModel.numStages * sizeof(float));
		fread(stageThreshold, sizeof(float), npdModel.numStages, fp);
		npdModel.stageThreshold = (double *)malloc(npdModel.numStages * sizeof(double));
		for (int i = 0; i < npdModel.numStages; i++)
		{
			npdModel.stageThreshold[i] = stageThreshold[i];
		}

		npdModel.treeRoot = (int *)malloc(npdModel.numStages * sizeof(int));
		fread(npdModel.treeRoot, sizeof(int), npdModel.numStages, fp);

		npdModel.pixelx = (int **)malloc(npdModel.numScales * sizeof(int*));
		for (int i = 0; i < npdModel.numScales; i++) {
			npdModel.pixelx[i] = (int *)malloc(npdModel.numBranchNodes * sizeof(int));
			fread(npdModel.pixelx[i], sizeof(int), npdModel.numBranchNodes, fp);
		}

		npdModel.pixely = (int **)malloc(npdModel.numScales * sizeof(int*));
		for (int i = 0; i < npdModel.numScales; i++) {
			npdModel.pixely[i] = (int *)malloc(npdModel.numBranchNodes * sizeof(int));
			fread(npdModel.pixely[i], sizeof(int), npdModel.numBranchNodes, fp);
		}

		npdModel.cutpoint = (unsigned char**)malloc(2 * sizeof(unsigned char*));
		for (int i = 0; i < 2; i++) {
			npdModel.cutpoint[i] = (unsigned char *)malloc(npdModel.numBranchNodes * sizeof(unsigned char));
			fread(npdModel.cutpoint[i], sizeof(unsigned char), npdModel.numBranchNodes, fp);
		}

		npdModel.leftChild = (int *)malloc(npdModel.numBranchNodes * sizeof(int));
		fread(npdModel.leftChild, sizeof(int), npdModel.numBranchNodes, fp);

		npdModel.rightChild = (int *)malloc(npdModel.numBranchNodes * sizeof(int));
		fread(npdModel.rightChild, sizeof(int), npdModel.numBranchNodes, fp);

		
		float* fit = (float *)malloc(npdModel.numLeafNodes * sizeof(float));
		fread(fit, sizeof(float), npdModel.numLeafNodes, fp);
		npdModel.fit = (double *)malloc(npdModel.numLeafNodes * sizeof(double));
		for (int i = 0; i < npdModel.numLeafNodes; i++)
		{
			npdModel.fit[i] = fit[i];
		}

		npdModel.winSize = (int *)malloc(npdModel.numScales * sizeof(int));
		fread(npdModel.winSize, sizeof(int), npdModel.numScales, fp);

		fclose(fp);

	}

	//model
	char modelName[100] = "";
	strcat(modelName, file);
	strcat(modelName, ".tem");
	fp = fopen(modelName, "rb");

	if (fp != NULL)
	{

		int numStages;
		fread(&numStages, sizeof(int), 1, fp);
		model.stages.resize(numStages);

		for (int i = 0; i < numStages; i++) {

			Stage newStage;

			//fit
			int fitNum;
			fread(&fitNum, sizeof(int), 1, fp);
			newStage.fit.resize(fitNum);
			float *fit = (float *)malloc(fitNum * sizeof(float));
			fread(fit, sizeof(float), fitNum, fp);
			for (int j = 0; j < fitNum; j++)
			{
				newStage.fit[j] = fit[j];
			}

			//cutpoint
			int numBranchNodes;
			fread(&numBranchNodes, sizeof(int), 1, fp);
			newStage.cutpoint.resize(2);
			for (int j = 0; j < 2; j++)
			{
				newStage.cutpoint[j].resize(numBranchNodes);
				fread(&(newStage.cutpoint[j][0]), sizeof(unsigned char), numBranchNodes, fp);
			}

			//leftChild
			newStage.leftChild.resize(numBranchNodes);
			fread(&(newStage.leftChild[0]), sizeof(int), numBranchNodes, fp);

			//rightChild
			newStage.rightChild.resize(numBranchNodes);
			fread(&(newStage.rightChild[0]), sizeof(int), numBranchNodes, fp);

			//feaId
			int feaIdNum;
			fread(&feaIdNum, sizeof(int), 1, fp);
			newStage.feaId.resize(feaIdNum);
			fread(&(newStage.feaId[0]), sizeof(int), feaIdNum, fp);

			//depth
			fread(&newStage.depth, sizeof(int), 1, fp);

			//thresshold
			float threshold;
			fread(&threshold, sizeof(float), 1, fp);
			newStage.threshold = threshold;

			//far
			float far;
			fread(&far, sizeof(far), 1, fp);
			newStage.far = far;

			model.stages[i] = newStage;
		}

	}

	return true;
}