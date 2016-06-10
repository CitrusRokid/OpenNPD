#include "common.h"

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

void BootstrapNonfaces
(vector<cv::Mat>& noFaceRects, cv::Mat& nonfaceFea, vector<int>& negIndex, DataBase& dataBase,
	NpdModel& npdModel, Model& model ,const char* nonfaceDBFile,
	int objSize, int numNegs,
	int numThreads) {
	
	int numLimit = floor(numNegs / 1000) + 1;
	int dispStep = floor(numNegs / 10) + 1;
	int dispCount = 0;

	if (!negIndex.empty() && dataBase.NonFaceDB.empty())
	{
		printf("Warning: negIndex not empty but NonfaceDB empty !\n");
		dataBase.getNonFaceRect(nonfaceDBFile, objSize, numNegs);
	}

	vector<cv::Mat> nonfacePatches;
	noFaceRects.clear();

	/*if npdModel is empty*/
	if (npdModel.isEmpty()) 
	{
		random_shuffle(dataBase.NonFaceDB.begin(), dataBase.NonFaceDB.end());
		nonfacePatches = dataBase.NonFaceDB;
		if (numNegs > int(nonfacePatches.size())) 
		{
			printf("Number of NonfaceDB not enough. Please check.");
		}
		else 
		{
			nonfacePatches.resize(numNegs);
		}
		
	}
	/*if npdmodel is not empty*/
	else 
	{
		nonfacePatches.resize(numNegs);
		int T = npdModel.numStages;

		int n;

		/*select noface images from nofaceDB*/
		if (!negIndex.empty() && !dataBase.NonFaceDB.empty()) 
		{
			int numValid = negIndex.size();
			vector<int> passCount;
			vector<double> score; 
			NPDClassify(score, passCount, npdModel, dataBase.NonFaceDB, negIndex, numThreads);
			vector<int> newNegIndex;

			for (int i = 0; i < int(negIndex.size()); i++) 
			{
				if (passCount[i] == T) 
				{
					newNegIndex.push_back(negIndex[i]);
				}
			}

			n = newNegIndex.size();

			cout << endl << n << " of " << numValid << " NonfaceDB samples. total: "
				<< (n < numNegs ? n : numNegs) << " of " << numNegs << endl;

			if (n > numNegs) 
			{
				random_shuffle(newNegIndex.begin(), newNegIndex.end());
				newNegIndex.resize(numNegs);
				n = numNegs;
			}

			for (int i = 0; i < int(newNegIndex.size()); i++) 
			{
				nonfacePatches[i] = dataBase.NonFaceDB[newNegIndex[i]];
			};

		}
		else 
		{
			n = 0;
		}

		//add noface images from nofaceImages
		if (n < numNegs) 
		{

			dataBase.NonFaceDB.clear();
			negIndex.clear();

			vector<int> samIndex;
			double sum = 0;
			for (int i = 0; i < int(dataBase.numGridFace.size()); i++) 
			{
				if (dataBase.numGridFace[i] > 0) {
					samIndex.push_back(i);
				}
				sum += dataBase.numGridFace[i];
			}

			IndexSort(samIndex, dataBase.numGridFace, -1);

			cout << sum << " grid samples remained." << endl;

			if (sum > numNegs - n) {
				for (int i = 0; i < int(samIndex.size()); i++) {

					cv::Mat NonFaceImage = cv::imread(dataBase.negImageNames[samIndex[i]], 0);

					vector<cv::Mat> rects = NPDGrid(npdModel, NonFaceImage,
						objSize, 4000, numThreads);

					int k = rects.size();
					dataBase.numGridFace[samIndex[i]] = k;

					if (k == 0) continue;

					if (k > numLimit || k > numNegs - n) {
						random_shuffle(rects.begin(), rects.end());
						k = numLimit < (numNegs - n) ? numLimit : (numNegs - n);
						rects.resize(k);
					}

					for (int j = 0; j < k; j++) {
						nonfacePatches[n] = cv::Mat(objSize, objSize, CV_8UC1);
						nonfacePatches[n++] = rects[j];
					}

					if (n > (dispStep*(dispCount + 1))) {
						printf("+%d grid samples. total: %d of %d. Time: ... seconds.\n", k, n, numNegs);
						dispCount++;
					}

					if (n == numNegs) break;

				}
			}
			else
			{
				printf("not ennough grid samples...\n");			
			}

			//neg is not enough yet
			if (n < numNegs) {

				samIndex.clear();
				sum = 0;
				for (int i = 0; i < int(dataBase.numSlideFace.size()); i++) {
					if (dataBase.numSlideFace[i] > 0) {
						samIndex.push_back(i);
					}
					sum += dataBase.numSlideFace[i];
				}

				IndexSort(samIndex, dataBase.numSlideFace, -1);

				cout << sum << " slide samples remained." << endl;

				if (sum > (numNegs - n)) {
					for (int i = 0; i < int(samIndex.size()); i++) {

						cv::Mat NonFaceImage = cv::imread(dataBase.negImageNames[samIndex[i]], 0);

						vector<cv::Mat> rects = NPDScan(npdModel, NonFaceImage,
							objSize, 4000, numThreads);

						int k = rects.size();
						dataBase.numSlideFace[samIndex[i]] = k;

						if (k == 0) continue;

						if (k > numLimit || k > numNegs - n) {
							random_shuffle(rects.begin(), rects.end());
							k = numLimit < (numNegs - n) ? numLimit : (numNegs - n);
							rects.resize(k);
						}

						for (int j = 0; j < k; j++) {
							nonfacePatches[n] = cv::Mat(objSize, objSize, CV_8UC1);
							nonfacePatches[n++] = rects[j];
						}

						if (n > (dispStep*(dispCount + 1))) {
							printf("+%d slide samples. total: %d of %d. Time: ... seconds.\n", k, n, numNegs);
							dispCount++;
						}

						if (n == numNegs) break;

					}
				}
				else
				{
					printf("not ennough slide samples...\n");
				}

			}

			//neg is still not enough yet
			if (n < numNegs) {

				samIndex.clear();
				sum = 0;
				for (int i = 0; i < int(dataBase.numSlideFace.size()); i++) {
					if (dataBase.numSlideFace[i] > 0) {
						samIndex.push_back(i);
					}
					sum += dataBase.numSlideFace[i];
				}

				IndexSort(samIndex, dataBase.numSlideFace, -1);

				cout << sum << " slide samples remained." << endl;
				if (sum > numNegs - n)
				{
					for (int i = 0; i < int(samIndex.size()); i++) {

						cv::Mat NonFaceImage = cv::imread(dataBase.negImageNames[samIndex[i]], 0);

						vector<cv::Mat> rects = NPDScan(npdModel, NonFaceImage,
							objSize, 4000, numThreads);

						int k = rects.size();
						dataBase.numSlideFace[samIndex[i]] = k;

						if (k == 0) continue;

						if (k > numNegs - n) {
							random_shuffle(rects.begin(), rects.end());
							k = numNegs - n;
							rects.resize(k);
						}

						for (int j = 0; j < k; j++) {
							nonfacePatches[n] = cv::Mat(objSize, objSize, CV_8UC1);
							nonfacePatches[n++] = rects[j];

						}

						if (n > (dispStep*(dispCount + 1))) {
							printf("+%d slide samples. total: %d of %d. Time: ... seconds.\n", k, n, numNegs);
							dispCount++;
						}

						if (n == numNegs) break;

					}
				}
				
			}

		}

		//still not enough
		if (n < numNegs) {
			for (int i = n; i < numNegs; i++) {
				nonfacePatches[i] = cv::Mat(objSize, objSize, CV_8UC1, cv::Scalar::all(0));
			}
			printf("not enough neg samples, add %d black images...", (numNegs - n));
		}

	}

	cout << "Extract nonface feature..." << endl;

	NPD(nonfacePatches, nonfaceFea);

	cout << "done." << endl;
}
