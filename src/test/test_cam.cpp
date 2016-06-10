
#include <ctime>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "npd/npddetect.h"

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("./main modelnpd");
        return -1;
    }
    std::string modelfilenpd(argv[1]);
    
    npd::npddetect npd;
    npd.load(modelfilenpd.c_str());

    cv::Mat shape;
    cv::Mat img;

    cv::VideoCapture cap;
    cap.open(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    cv::Mat frame;

    while(1)
    {
        cap >> frame;
        cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
        //TIMER_BEGIN
        double t = (double)cvGetTickCount();
        int n = npd.detect(img.data, img.cols, img.rows);
        t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;

        printf("Detect num: %d (%lf ms)\n", n, t);
        //LOG("Detect %d face, cost %lf(ms)", n, TIMER_NOW);
        //TIMER_END




        vector< int >& Xs = npd.getXs();
        vector< int >& Ys = npd.getYs();
        vector< int >& Ss = npd.getSs();
        //vector< float >& Scores = npd.getScores();

        for(int i = 0; i < n; i++)
        {
            //int deta = Ss[i]/10;
            //if(Scores[i] < 22.38)
            //    continue;

            cv::rectangle(img, cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]), cv::Scalar(255, 0, 0));

        }
        cv::imshow("hehe", img);
        cv::waitKey(10);

    }

    return 0;
}
