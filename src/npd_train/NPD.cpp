
#include "common.h"
#include "opencv2/core/core.hpp"
#include <omp.h>
#include <math.h>

using namespace std;

void NPD(vector<cv::Mat>& imageArray, cv::Mat& feaMat)
{

	// Set the number of threads
	int numProcs = omp_get_num_procs();
	int numThreads = (int)floor(numProcs * 0.8);
	omp_set_num_threads(numThreads);

	int height = imageArray[0].rows;
	int width = imageArray[0].cols;
	int numPixels = height * width;

	int numImgs = 1;
	numImgs = imageArray.size();

	int feaDims = numPixels * (numPixels - 1) / 2;

	feaMat =  cv::Mat(numImgs, feaDims, CV_8UC1);

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

	// process each image

	for (int k = 0; k < numImgs; k++) {
		const uchar* data = imageArray[k].data;
		int d = 0;

		uchar* ptr = feaMat.ptr(k);

		for (int i = 0; i < numPixels; i++) {
			for (int j = i+1 ; j < numPixels; j++) {
				ptr[d++] = npd::npdTable[data[offset[j]]][data[offset[i]]];
			}
		}
	}

}
