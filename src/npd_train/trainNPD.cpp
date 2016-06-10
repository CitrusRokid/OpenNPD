#include <time.h>
#include <stdio.h>
#include <iostream>
#include "common.h"

using namespace std;

void TrainDetector(const char* faceDBFile, const char* nonfaceDBFile, const char* outFile, Options options) {

	int objSize = 20,
		numFaces = 999999,
		finalNegs = 100,
		numThreads = 24;
	double negRatio = 1;

	objSize = options.objSize;
	numFaces = options.numFaces;
	negRatio = options.negRatio;
	finalNegs = options.finalNegs;
	numThreads = options.numThreads;

	BoostOpt boostOpt = options.boostOpt;
	boostOpt.numThreads = numThreads;

	DataBase dataBase(options.negImageStep);


	dataBase.getFaceRect(faceDBFile, objSize, options.ifFlip);
	dataBase.getNonFaceImage(nonfaceDBFile);

	random_shuffle(dataBase.FaceDB.begin(), dataBase.FaceDB.end());

	int numStages = 0;
	Model model;
	NpdModel npdModel;

	LoadNpdModel(outFile, npdModel, model);

	numStages = model.stages.size();

	numFaces = numFaces < int(dataBase.FaceDB.size()) ? numFaces : int(dataBase.FaceDB.size());
	if (numFaces < int(dataBase.FaceDB.size())) {
		dataBase.FaceDB.resize(numFaces);
	}

	int desireNumNegs = numFaces*negRatio;
	dataBase.getNonFaceRect(nonfaceDBFile, objSize, desireNumNegs);
	int numSamples = dataBase.NonFaceDB.size();

	cout << "Extract face features" << endl;

	/*get faces' NPD feature*/
	cv::Mat faceFea;
	NPD(dataBase.FaceDB, faceFea);

	vector<int> negIndex(numSamples);

	if (false) {	//TODO::

	}
	else {
		for (int i = 0; i < numSamples; i++) {
			negIndex[i] = i;
		}
	}

	cout << "Start to train detector" << endl;

	int T = model.stages.size();	//get current numStages

	clock_t trainTime = 0;

	bool finished = false;
	clock_t t0 = clock();

	while (true) {

		cv::Mat nonfaceFea;

		/*bootstrap method to select neg samples*/
		vector<cv::Mat> nonFaceRects;

		BootstrapNonfaces(nonFaceRects, nonfaceFea,negIndex, dataBase,
			npdModel, model, nonfaceDBFile, objSize, desireNumNegs,
		    numThreads);

		if (negIndex.size() < 10000) {
			negIndex.clear();
			dataBase.NonFaceDB.clear();
		}

		if (nonfaceFea.rows < finalNegs) {
			printf("\n\nNo enough negative examples to bootstrap (nNeg=%d). The detector training is terminated.\n", nonfaceFea.rows);
			trainTime = (clock() - t0);
			printf("\nTraining time: %.0fs.\n", float(trainTime));
			break;
		}

		if (nonfaceFea.rows == desireNumNegs) {
			LearnGAB(model, faceFea, nonfaceFea, npdModel, nonFaceRects,  boostOpt);
		}
		else {
			dataBase.NonFaceDB.clear();
			dataBase.NonFaceImages.clear();
			BoostOpt boostOpt2 = boostOpt;
			boostOpt2.minNegRatio = finalNegs/nonfaceFea.rows;
			LearnGAB(model, faceFea, nonfaceFea, npdModel, nonFaceRects, boostOpt2);
			finished = true;
		}

		//TODO: release nonfaceFea

		npdModel = PackNPDModel(model, objSize);

		if (int(model.stages.size()) == T) {		//no new features anymore
			printf("\n\nNo effective features for further detector learning.\n");
			break;
		}

		T = model.stages.size();
		numStages++;
		trainTime =  (clock() - t0);

		SaveNpdModel(outFile, npdModel, model);

		double far = 1;
		for (int i = 0; i < int(model.stages.size()); i++) {
			far *= model.stages[i].far;
		}

		printf("\nStage %d, #Weaks: %d, FAR: %lf, Training time: %.0fms, Time per stage: %.0fs, Time per weak: %.3fs.\n\n",
			numStages, T, far, float(trainTime), float(trainTime) / numStages, float(trainTime) / T);

		if (far <= boostOpt.maxFAR || T >= boostOpt.maxNumWeak || finished) {
			printf("\n\nThe detector training is finished.\n");
			break;
		}
	}

};

void help() {
	cout << "trainNPD -faceDB facelist -negDB nonfacelist [-outModel outputmodelname] [-objSize size] [-numPos num_of_pos] [-negRatio neg_ratio_to_pos] [-maxTreeLevel treeDepth] [-minDR min_detect_rate] [-maxFAR max_false_alarm_rate] [-maxNumStages max_num_of_stages] [-ifFlip if_flip_image]\n"
		"	-faceDB:         positive samples as list of image files, no default\n"
		"	-negDB:          negtive samples as list of image files, no default\n"
		"	-outModel:       result file name without format, default \"result\"\n"
		"	-objSize:        object size, default 20\n"
		"	-numPos:         number of pos face used for training, default as much in faceDB\n"
		"	-negRatio:       neg ratio to pos face, default 1\n"
		"	-maxTreeLevel:   max depth of tree, default 4\n"
		"	-minDR:          min final Detect Rate, default 1\n"
		"	-maxFAR:         max final false alarm rate, default 1e-6\n"
		"	-maxNumStages:   max number of stages, default not limited\n"
		"	-ifFlip:         if flip positive samples, set 1, default not flip\n"
		"	"<< endl;
}

int main(int argc, char** argv) {

	string faceDBName,NonfaceDBName, outName;
	int objSize = 20, numPos = 999999999, treeLevel = 4, maxNumStages = 200, negImageStep = 10000;
	double negRatio = 1., minDR = 1., maxFAR = 1e-16;
	bool ifFlip = false;
	if (argc < 17) {

		cout << "inseffisient args !" << endl << endl;
		help();
		return 0;
	}
	else {
		for (int i = 1; i < argc; i += 2) {
			if (strcmp(argv[i], "-faceDB") == 0) {
				faceDBName = argv[i + 1];
			}
			else
			if (strcmp(argv[i], "-negDB") == 0) {
				NonfaceDBName = argv[i + 1];
			}
			else
			if (strcmp(argv[i], "-outModel") == 0) {
				outName = argv[i + 1];
			}
			else
			if (strcmp(argv[i], "-objSize") == 0) {
				objSize = atoi(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-numPos") == 0) {
				numPos = atoi(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-negRatio") == 0) {
				negRatio = atof(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-maxTreeLevel") == 0) {
				treeLevel = atoi(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-minDR") == 0) {
				minDR = atof(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-maxFAR") == 0) {
				maxFAR = atof(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-maxNumStages") == 0) {
				maxNumStages = atoi(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-negImageStep") == 0) {
				negImageStep = atoi(argv[i + 1]);
			}
			else
			if (strcmp(argv[i], "-ifFlip") == 0) {
				ifFlip = ( atoi(argv[i + 1]) == 1 ? true : false );
			}
			else {
				cout << "arg " << argv[i] << " not accetalbe" << endl << endl;
				help();
				return 0;
			}
		}
	}

	Options options;
	options.objSize = objSize;
	options.negRatio = negRatio;
	options.finalNegs = 100;
	options.numFaces = numPos;
	options.numThreads = 40;
	options.negImageStep = negImageStep;
	options.ifFlip = ifFlip;

	options.boostOpt.treeLevel = treeLevel;
	options.boostOpt.maxNumWeak = maxNumStages;
	options.boostOpt.minDR = minDR;
	options.boostOpt.maxFAR = maxFAR;
	options.boostOpt.minSamples = 100;

	TrainDetector(faceDBName.c_str(), NonfaceDBName.c_str(), outName.c_str(), options);

	return 0;
}
