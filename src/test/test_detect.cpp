
/********************************************************************
 > File Name: test_detect.h
 > Author: cmxnono
 > Mail: 381880333@qq.com
 > Created Time: 14/01/2016 
 *********************************************************************/

#include <stdio.h>
#include <vector>

#include "npd/npddetect.h"

#include "opencv2/opencv.hpp"

using namespace std;

int main(int argc, char* argv[])
{

    printf("************* [TEST] Npd:detect test... *************\n");
    
    npd::npddetect npd;
    npd.load(argv[1]);
    cv::Mat img = cv::imread(argv[2], 0);
    string savepath = "1.jpg";
    float score = -1;
    if(argc >= 4)
	    savepath = argv[3];
    if(argc >= 5)
    {
	    score = atof(argv[4]);
    }

    int nt = 1; 
    int nc = nt; 
    int n;
    double t = (double)cvGetTickCount();
    while(nc-- > 0)
        n = npd.detect(img.data, img.cols, img.rows);
    t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;

    printf("Detect num: %d (%lf ms avg of %d test)\n", n, t/nt, nt);
    vector< int >& Xs = npd.getXs();
    vector< int >& Ys = npd.getYs();
    vector< int >& Ss = npd.getSs();
    vector< float >& Scores = npd.getScores();
    char buf[10];
    for(int i = 0; i < n; i++)
    {
	if(score > 0. && Scores[i] < score)
		continue;
	sprintf(buf, "%.3f", Scores[i]);
        cv::rectangle(img, cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]), 
                cv::Scalar(128,128,128));
	cv::putText(img, buf, cv::Point(Xs[i], Ys[i]), 1, 0.5, cv::Scalar(255,255,255)); 
    }
    cv::imwrite(savepath.c_str(), img);

    printf("************* [TEST] Npd:detect ok!!!!! *************\n");
    return 0;
}
