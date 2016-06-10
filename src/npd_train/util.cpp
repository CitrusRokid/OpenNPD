#include <time.h>

#include <iostream>
#include <fstream>
#include "common.h"

void npd::resizeMat(const cv::Mat& oriMat, cv::Mat& outMat, int size)
{
	int oriSize = oriMat.rows;

	//int channels = oriMat.channels();

	if (oriSize == size)
	{
		outMat = oriMat;
		return;
	}
	else
	{
		double scaleFactor = oriSize / (double)size;

		outMat = cv::Mat(size, size, CV_8UC1, cv::Scalar::all(0));

#pragma omp parallel for
		for (int r = 0; r < size; r++)
		{

			int oriR = round(scaleFactor * r);

			const unsigned char * oriPtr = oriMat.ptr(oriR);

			unsigned char * ptr = outMat.ptr(r);

			for (int c = 0; c < size; c++)
			{
				int oriC = round(scaleFactor * c);
				ptr[c] = oriPtr[oriC];

			}

		}

	}

};

void printMat(const cv::Mat& oriMat, string name)
{
	printf("\n\n\%s\n", name.c_str());

	int oriSize = oriMat.rows;

	for (int r = 0; r < oriSize; r++)
	{

		const unsigned char * ptr = oriMat.ptr(r);

		for (int c = 0; c < oriSize; c++)
		{

			printf("%d\t", ptr[c]);
		}

		printf("\n");
	}

	for (int i = 0; i < oriSize*oriSize; i++)
	{
		if (i%oriSize == 0) {
			printf("\n");
		}
		printf("%d\t", oriMat.data[i]);

	}
};

DataBase::DataBase() :negImageStep(99999999), negImageStepIndex(0) {};

DataBase::DataBase(int NegStep):negImageStepIndex(0)
{
	negImageStep = NegStep;
}

bool DataBase::getFaceRect(const char * fileName, int objSize, bool ifFlip)
{
	printf("load FaceDB:%s\n", fileName);

	ifstream file(fileName);
	string imageName;

	while (getline(file, imageName))
	{
		cv::Mat image = cv::imread(imageName.c_str(),0);
        if(image.empty())
        {
            printf("empty file path:%s\n", imageName.c_str());
            continue;
        }

		if (image.cols != objSize || image.rows != objSize)
		{
			cv::resize(image, image, cv::Size(objSize, objSize));
		}

		FaceDB.push_back(image);

		if (ifFlip)
		{
			cv::Mat f_image;
			cv::flip(image, f_image, 1);
			FaceDB.push_back(f_image);
		}
	}

	printf("done\n");
	return true;
}

bool DataBase::getNonFaceRect(const char * fileName, int objSize, int numNeg)
{
	printf("load NonFaceDB...\n");

	int num = 0;

	ifstream file(fileName);
	string imageName;

	double scaleFactor = 1.6;
	int step;
	cv::Size imgSz, procSz, cropSz;
	cv::Size outSz(objSize, objSize);

	while (num < numNeg)
	{
		if (!getline(file, imageName))
			return false;

		cv::Mat image = cv::imread(imageName.c_str(), 0);
		if (image.empty())
		{
			printf("%s not exist.", imageName.c_str());
			continue;
		}

		imgSz = image.size();
		int w = imgSz.width < imgSz.height ? imgSz.width : imgSz.height;
		cropSz = cv::Size(w, w);

		for (; cropSz.width > objSize; cropSz.width /= scaleFactor, cropSz.height /= scaleFactor) {

			procSz.width = imgSz.width - cropSz.width;
			procSz.height = imgSz.height - cropSz.height;

			step = cropSz.width;

			for (int y = 0; y < procSz.height; y += step) {
				for (int x = 0; x < procSz.width; x += step) {
					num++;
					cv::Mat cropImg = cv::Mat(image, cv::Rect(x, y, cropSz.width, cropSz.height));
					cv::resize(cropImg, cropImg, outSz);
					NonFaceDB.push_back(cropImg);

				}
			}

		}

	}

	NonFaceDB.resize(numNeg);

	printf("done\n");
	return true;
}

bool DataBase::getNonFaceImage(const char * fileName)
{

	if (negImageNames.empty())
	{

		ifstream file(fileName);
		string imageName;

		while (!file.eof())
		{
			getline(file, imageName);
			negImageNames.push_back(imageName);
		}

		random_shuffle(negImageNames.begin(), negImageNames.end());

	}

	int numNonFaceImgs;

	numNonFaceImgs = negImageNames.size();

	numGridFace.resize(numNonFaceImgs);
	numSlideFace.resize(numNonFaceImgs);

	for (int i = 0; i < numNonFaceImgs; i++) {
		numGridFace[i] = 1.2 * 1e6;
		numSlideFace[i] = 1.2 * 1e6;
	}

	return true;
}

void DataBase::resetNegNums()
{
	int numNonFaceImgs;

	numNonFaceImgs = NonFaceImages.size();

	numGridFace.resize(numNonFaceImgs);
	numSlideFace.resize(numNonFaceImgs);

	for (int i = 0; i < numNonFaceImgs; i++) {
		numGridFace[i] = 1.2 * 1e6;
		numSlideFace[i] = 1.2 * 1e6;
	}
}


