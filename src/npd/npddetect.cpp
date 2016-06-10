
/********************************************************************
  > File Name: npddetect.h
  > Author: cmxnono
  > Mail: 381880333@qq.com
  > Created Time: 14/01/2016
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "npd/npddetect.h"

string NumberToString(int val)
    {
        stringstream ss;
         ss << val;
         return ss.str();
    }

namespace npd{

    npddetect::npddetect()
    {
        init(m_model.m_objSize, 5000);
    }

    npddetect::npddetect(int minFace, int maxFace)
    {
        init(minFace, maxFace);
    }

    npddetect::~npddetect()
    {
        release();
    }

    //npddetect::npddetect()
    //{
     //   init();

        //load(modelpath);
    //}

    void npddetect::init(int minFace, int maxFace)
    {
        m_minFace = minFace;
        m_maxFace = maxFace;
        m_overlappingThreshold = 0.5;
        m_maxScanNum = 0;
        m_maxDetectNum = 0;

        m_Tpredicate    = NULL;
        m_Troot         = NULL;
        m_Tlogweight    = NULL;
        m_Tparent       = NULL;
        m_Trank         = NULL;
        mallocsacnspace(500);

        m_Tneighbors    = NULL;
        m_Tweight       = NULL;
        m_Txs           = NULL;
        m_Tys           = NULL;
        m_Tss           = NULL;
        mallocdetectspace(40);
    }

    void npddetect::mallocsacnspace(int s)
    {
        if(s > m_maxScanNum)
            freesacnspace();

        // Malloc.
        m_Tpredicate = (char*)malloc(sizeof(char) * s * s);
        if(m_Tpredicate == NULL)
            return;

        m_Troot = (int*)malloc(sizeof(int) * s);
        if(m_Troot == NULL)
            return;

        m_Tlogweight = (float*)malloc(sizeof(float) * s);
        if(m_Tlogweight == NULL)
            return;

        m_Tparent = (int*)malloc(sizeof(int) * s);
        if(m_Tparent == NULL)
            return;

        m_Trank = (int*)malloc(sizeof(int) * s);
        if(m_Trank == NULL)
            return;

        m_maxScanNum = s;
    }

    void npddetect::freesacnspace()
    {
        if(m_Tpredicate != NULL)
            free(m_Tpredicate);
        m_Tpredicate = NULL;

        if(m_Troot != NULL)
            free(m_Troot);
        m_Troot = NULL;

        if(m_Tlogweight != NULL)
            free(m_Tlogweight);
        m_Tlogweight = NULL;

        if(m_Tparent != NULL)
            free(m_Tparent);
        m_Tparent = NULL;

        if(m_Trank != NULL)
            free(m_Trank);
        m_Trank = NULL;
    }

    void npddetect::mallocdetectspace(int n)
    {
        if(n > m_maxDetectNum)
            freedetectspace();

        // Malloc.
        m_Tneighbors = (int*)malloc(sizeof(int) * n);
        if(m_Tneighbors == NULL)
            return;

        m_Tweight = (float*)malloc(sizeof(float) * n);
        if(m_Tweight == NULL)
            return;

        m_Txs = (float*)malloc(sizeof(float) * n);
        if(m_Txs == NULL)
            return;

        m_Tys = (float*)malloc(sizeof(float) * n);
        if(m_Tys == NULL)
            return;

        m_Tss = (float*)malloc(sizeof(float) * n);
        if(m_Tss == NULL)
            return;

        m_maxDetectNum = n;
    }

    void npddetect::freedetectspace()
    {
        if(m_Tneighbors != NULL)
            free(m_Tneighbors);
        m_Tneighbors = NULL;

        if(m_Tweight != NULL)
            free(m_Tweight);
        m_Tweight = NULL;

        if(m_Txs != NULL)
            free(m_Txs);
        m_Txs = NULL;

        if(m_Tys != NULL)
            free(m_Tys);
        m_Tys = NULL;

        if(m_Tss != NULL)
            free(m_Tss);
        m_Tss = NULL;
    }

    void npddetect::release()
    {
        freesacnspace();
        freedetectspace();
    }

    void npddetect::load(const char* modelpath)
    {
        m_model.load(modelpath);
    }

    int npddetect::detect(const unsigned char* I, int width, int height)
    {
        // Clear former data.
        reset();

        // Scan with model.
        m_numScan = scan(I, width, height);
        if(m_numScan > m_maxScanNum)
        {
            if(2*m_maxScanNum > m_numScan)
                mallocsacnspace(m_maxScanNum * 2);
            else
                mallocsacnspace(m_numScan + m_maxScanNum);
        }

        // Merge rect.
        m_numDetect = filter();

        return m_numDetect;
    }

    int npddetect::prescandetect(const unsigned char* I, int width, int height, double stepR, double thresR)
    {
        // Clear former data.
        reset();

        //double time = double(cvGetTickCount());
        //double frequency = double(cvGetTickFrequency());
        // Scan with model.
        m_numScan = gridScan(I, width, height, stepR, thresR);
        //1printf("gridScan time: %fms", (cvGetTickCount() - time)/frequency );

        if(m_numScan > m_maxScanNum)
        {
            if(2*m_maxScanNum > m_numScan)
                mallocsacnspace(m_maxScanNum * 2);
            else
                mallocsacnspace(m_numScan + m_maxScanNum);
        }

        // Merge rect.
        //time = cvGetTickCount();
        m_numDetect = filter();
        //printf("filter time: %fms", (cvGetTickCount() - time)/frequency );

        return m_numDetect;
    }

    void npddetect::reset()
    {
        m_xs.clear();
        m_ys.clear();
        m_sizes.clear();
        m_scores.clear();
        m_Xs.clear();
        m_Ys.clear();
        m_Ss.clear();
        m_Scores.clear();
        m_numScan = 0;
    }

    int npddetect::gridScan(const unsigned char* I, int width, int height, double stepR, double thresR)
    {
        //int minFace = m_model.m_objSize;
        int minFace = max(m_minFace, m_model.m_objSize);
        int maxFace = min(m_maxFace, min(height, width));

        if(min(height, width) < minFace)
            return 0;

        //printf("w:%d h:%d min:%d max:%d numScale:%d\n",
        //        width, height, minFace, maxFace, m_model.m_numScales);

        cv::Mat scoreMat(height, width, CV_32FC1, cv::Scalar::all(0));

        int k;
        //double tcnt = 0.0f;
        for(k = 0; k < m_model.m_numScales; k++) // process each scale
        {
            //printf("at scale: %d\n", m_model.m_winSize[k]);

            if(m_model.m_winSize[k] < minFace) continue;
            else if(m_model.m_winSize[k] > maxFace) break;

            // determine the step of the sliding subwindow
            int winStep = (int) floor(m_model.m_winSize[k] * stepR);
	    if(m_model.m_winSize[k] > 40) winStep = (int) floor(m_model.m_winSize[k] * 0.5 * stepR);
            //double t = (double)cvGetTickCount();
            // calculate the offset values of each pixel in a subwindow
            // pre-determined offset of pixels in a subwindow
            vector<int> offset(m_model.m_winSize[k] * m_model.m_winSize[k]);
            int p1 = 0, p2 = 0, gap = width;

            for(int j=0; j < m_model.m_winSize[k]; j++) // column coordinate
            {
                p2 = j;
                for(int i = 0; i < m_model.m_winSize[k]; i++) // row coordinate
                {
                    offset[p1++] = p2;
                    p2 += gap;
                }
            }
            //t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;
            //tcnt += t;

            int colMax = width - m_model.m_winSize[k] + 1;
            int rowMax = height - m_model.m_winSize[k] + 1;

            double minVal = 99999, maxVal = -99999;

            // process each subwindow
            for(int r = 0; r < rowMax; r += winStep) // slide in row
            {
                const unsigned char *pPixel = I + r * width;;
                for(int c = 0; c < colMax; c += winStep, pPixel += winStep) // slide in column

                {
                    int treeIndex = 0;
                    float _score = 0;
                    int s;

                    // test each tree classifier
                    for(s = 0; s < m_model.m_numStages; s++)
                    {
                        int node = m_model.m_treeRoot[treeIndex];

                        // test the current tree classifier
                        while(node > -1) // branch node
                        {
                            unsigned char p1 = pPixel[
                                offset[m_model.m_pixelx[k][node]]];
                            unsigned char p2 = pPixel[
                                offset[m_model.m_pixely[k][node]]];
                            unsigned char fea = npdTable[p2][p1];
                            //printf("w[0][0]=%d\n", pPixel[0]);
                            //printf("r = %d, c = %d, k = %d, node = %d, fea = %d, cutpoint = (%d, %d), p1off = %d, p2off = %d, p1x = %d, p2x = %d, p1 = %d, p2 = %d, winsize = %d\n",
                            //        r, c, k, node, int(fea), int(m_model.m_cutpoint[0][node]), int(m_model.m_cutpoint[1][node]),
                            //        offset[m_model.m_pixelx[k][node]], offset[m_model.m_pixely[k][node]],
                            //        m_model.m_pixelx[k][node], m_model.m_pixely[k][node], p1, p2, m_model.m_winSize[k]);

                            if(fea < m_model.m_cutpoint[0][node]
                                    || fea > m_model.m_cutpoint[1][node])
                                node = m_model.m_leftChild[node];
                            else
                                node = m_model.m_rightChild[node];
                        }

                        // leaf node
                        node = - node - 1;
                        _score = _score + m_model.m_fit[node];
                        treeIndex++;

                        //printf("stage = %d, score = %f\n", s, _score);
                        if(_score < m_model.m_stageThreshold[s])
                            break; // negative samples
                    }

                    scoreMat.at<float>(r, c) = _score;
                    if(_score < minVal)  minVal = _score ;
                    if(_score > maxVal)  maxVal = _score ;

                } // Cols.
            } // Row.


            floodScoreMat(scoreMat, rowMax, colMax, winStep);
            double GridThreshold = minVal +( maxVal - minVal) * thresR;
            //printf("GridThreshold: %f\n", GridThreshold);

            // determine the step of the sliding subwindow
            winStep = (int) floor(m_model.m_winSize[k] * 0.1);
            if(m_model.m_winSize[k] > 40) winStep = (int) floor(m_model.m_winSize[k] * 0.05);

            //double t = (double)cvGetTickCount();
            // calculate the offset values of each pixel in a subwindow
            // pre-determined offset of pixels in a subwindow

            //t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;
            //tcnt += t;

            // process each subwindow
            for(int r = 0; r < rowMax; r += winStep) // slide in row
            {
                const unsigned char *pPixel = I + r * width;;
                for(int c = 0; c < colMax; c += winStep, pPixel += winStep) // slide in column
                {
                    int treeIndex = 0;
                    float _score = 0;
                    int s = 0;

                    if(scoreMat.at<float>(r,c) < GridThreshold)
                        continue;
                    // test each tree classifier
                    for(s = 0; s < m_model.m_numStages; s++)
                    {
                        int node = m_model.m_treeRoot[treeIndex];

                        // test the current tree classifier
                        while(node > -1) // branch node
                        {
                            unsigned char p1 = pPixel[
                                offset[m_model.m_pixelx[k][node]]];
                            unsigned char p2 = pPixel[
                                offset[m_model.m_pixely[k][node]]];
                            unsigned char fea = npdTable[p2][p1];
                            //printf("w[0][0]=%d\n", pPixel[0]);
                            //printf("r = %d, c = %d, k = %d, node = %d, fea = %d, cutpoint = (%d, %d), p1off = %d, p2off = %d, p1x = %d, p2x = %d, p1 = %d, p2 = %d, winsize = %d\n",
                            //        r, c, k, node, int(fea), int(m_model.m_cutpoint[0][node]), int(m_model.m_cutpoint[1][node]),
                            //        offset[m_model.m_pixelx[k][node]], offset[m_model.m_pixely[k][node]],
                            //        m_model.m_pixelx[k][node], m_model.m_pixely[k][node], p1, p2, m_model.m_winSize[k]);

                            if(fea < m_model.m_cutpoint[0][node]
                                    || fea > m_model.m_cutpoint[1][node])
                                node = m_model.m_leftChild[node];
                            else
                                node = m_model.m_rightChild[node];
                        }

                        // leaf node
                        node = - node - 1;
                        _score = _score + m_model.m_fit[node];
                        treeIndex++;

                        //printf("stage = %d, score = %f\n", s, _score);
                        if(_score < m_model.m_stageThreshold[s])
                            break; // negative samples
                    }

                    if(s == m_model.m_numStages) // a face detected
                    {
                        m_ys.push_back(r + 1);
                        m_xs.push_back(c + 1);
                        m_sizes.push_back(m_model.m_winSize[k]);
                        m_scores.push_back(_score);
                    }
                } // Cols.
            } // Row.

        } // Scale.

        //for(int i = 0; i < scoreMats.size(); i++)
        //{
        //    double minVal, maxVal;
        //    cv::minMaxLoc(scoreMats[i], &minVal, &maxVal);
        //    scoreMats[i] -= minVal;
        //    cv::Mat B;
        //    scoreMats[i].convertTo(B, CV_8UC1, 255.0/(maxVal - minVal));
        //    cv::namedWindow("result" + NumberToString(i));
        //    cv::imshow("result" + NumberToString(i), B);
        //}

        //cv::waitKey(-1);

        return m_ys.size();

    }



    int npddetect::floodScoreMat(cv::Mat& mat, int rowMax, int colMax, int winStep)
    {
        int rows = mat.rows;
        int cols = mat.cols;

        int yGridNum = rowMax / winStep;
        int xGridNum = colMax / winStep;


         //for cols
        for(int c = 0; c < colMax; c+=winStep)
        {
             for(int g = 0; g < yGridNum; g++)
             {
                 int begin = g*winStep;
                 int end = begin+winStep;
                 float beginVal = mat.at<float>(begin, c);
                 float endVal = mat.at<float>(end, c);
                 //if(beginVal == 0 || endVal == 0)
                 //{
                     //printf("Warning: grid scores have 0 val ! begin: %d end: %d col: %d beginVal: %f endVal: %f\n", begin, end, c, beginVal, endVal );
                 //}

                 for(int r = begin + 1; r < end; r ++)
                 {
                      mat.at<float>(r, c) = ((r - begin)*endVal + (end - r)*beginVal)/winStep;
                 }
             }

             int begin = yGridNum *winStep;
             int end = rows;
             int step = rows - begin;
             float beginVal = mat.at<float>(begin, c);
             float endVal = mat.at<float>(0, c);
             for(int r = begin + 1; r < end ;r++)
             {
                  mat.at<float>(r, c) = ((r - begin)*endVal + (end - r)*beginVal)/step;
             }

        }

        //for rows
        for(int r = 0; r < rows ; r++)
        {

             float* rowHead = mat.ptr<float>(r);
             for(int g = 0; g < xGridNum; g++)
             {
                int begin = g*winStep;
                int end = begin + winStep;
                float beginVal = rowHead[begin];
                float endVal = rowHead[end];
                //if(beginVal == 0 || endVal == 0)
                //{
                    //printf("Warning: grid scores have 0 val ! begin: %d end: %d row: %d beginVal: %f endVal: %f\n", begin, end, r, beginVal, endVal );
                //}

                for(int c = begin + 1; c < end; c++)
                {
                     rowHead[c] = ((c - begin) * endVal + (end - c) * beginVal)/winStep;
                }
             }

             int begin = xGridNum *winStep;
             int end = cols;
             int step = end - begin;
             float beginVal = rowHead[begin];
             float endVal = rowHead[0];
             for(int c = begin + 1; c < end ;c++)
             {
                 rowHead[c] = ((c - begin) * endVal + (end - c) * beginVal)/step;
             }
        }

        return 0;
    }

    int npddetect::scan(const unsigned char* I, int width, int height)
    {
        //int minFace = m_model.m_objSize;
        int minFace = max(m_minFace, m_model.m_objSize);
        int maxFace = min(m_maxFace, min(height, width));

        if(min(height, width) < minFace)
            return 0;

        //printf("w:%d h:%d min:%d max:%d numScale:%d\n",
        //        width, height, minFace, maxFace, m_model.m_numScales);

        int k;
        //double tcnt = 0.0f;
        for(k = 0; k < m_model.m_numScales; k++) // process each scale
        {
            if(m_model.m_winSize[k] < minFace) continue;
            else if(m_model.m_winSize[k] > maxFace) break;

            // determine the step of the sliding subwindow
            int winStep = (int) floor(m_model.m_winSize[k] * 0.1);
            if(m_model.m_winSize[k] > 40) winStep = (int) floor(m_model.m_winSize[k] * 0.05);

            //double t = (double)cvGetTickCount();
            // calculate the offset values of each pixel in a subwindow
            // pre-determined offset of pixels in a subwindow
            vector<int> offset(m_model.m_winSize[k] * m_model.m_winSize[k]);
            int p1 = 0, p2 = 0, gap = width;

            for(int j=0; j < m_model.m_winSize[k]; j++) // column coordinate
            {
                p2 = j;
                for(int i = 0; i < m_model.m_winSize[k]; i++) // row coordinate
                {
                    offset[p1++] = p2;
                    p2 += gap;
                }
            }
            //t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;
            //tcnt += t;

            int colMax = width - m_model.m_winSize[k] + 1;
            int rowMax = height - m_model.m_winSize[k] + 1;

            // process each subwindow
            for(int r = 0; r < rowMax; r += winStep) // slide in row
            {
                const unsigned char *pPixel = I + r * width;;
                for(int c = 0; c < colMax; c += winStep, pPixel += winStep) // slide in column

                {
                    int treeIndex = 0;
                    float _score = 0;
                    int s;

                    // test each tree classifier
                    for(s = 0; s < m_model.m_numStages; s++)
                    {
                        int node = m_model.m_treeRoot[treeIndex];

                        // test the current tree classifier
                        while(node > -1) // branch node
                        {
                            unsigned char p1 = pPixel[
                                offset[m_model.m_pixelx[k][node]]];
                            unsigned char p2 = pPixel[
                                offset[m_model.m_pixely[k][node]]];
                            unsigned char fea = npdTable[p2][p1];
                            //printf("w[0][0]=%d\n", pPixel[0]);
                            //printf("r = %d, c = %d, k = %d, node = %d, fea = %d, cutpoint = (%d, %d), p1off = %d, p2off = %d, p1x = %d, p2x = %d, p1 = %d, p2 = %d, winsize = %d\n",
                            //        r, c, k, node, int(fea), int(m_model.m_cutpoint[0][node]), int(m_model.m_cutpoint[1][node]),
                            //        offset[m_model.m_pixelx[k][node]], offset[m_model.m_pixely[k][node]],
                            //        m_model.m_pixelx[k][node], m_model.m_pixely[k][node], p1, p2, m_model.m_winSize[k]);

                            if(fea < m_model.m_cutpoint[0][node]
                                    || fea > m_model.m_cutpoint[1][node])
                                node = m_model.m_leftChild[node];
                            else
                                node = m_model.m_rightChild[node];
                        }

                        // leaf node
                        node = - node - 1;
                        _score = _score + m_model.m_fit[node];
                        treeIndex++;

                        //printf("stage = %d, score = %f\n", s, _score);
                        if(_score < m_model.m_stageThreshold[s])
                            break; // negative samples
                    }

                    if(s == m_model.m_numStages) // a face detected
                    {
                        m_ys.push_back(r + 1);
                        m_xs.push_back(c + 1);
                        m_sizes.push_back(m_model.m_winSize[k]);
                        m_scores.push_back(_score);
                    }
                } // Cols.
            } // Row.
        } // Scale.

        //printf("Userd %lf(ms) for scan...\n", 0);

        return (int) m_ys.size();
    }

    float logistic(float s)
    {
        return log(1 + exp(double(s)));
    }


    int npddetect::filter()
    {
        if(m_numScan <= 0)
            return 0;

        int i, j, ni, nj;
        float h, w, s, si, sj;
        //char* predicate = (char*)malloc(
        //        sizeof(char) * m_numScan * m_numScan);

        //printf("maxScan:%d numScan:%d mPredict:%d\n",
        //        m_maxScanNum, m_numScan, m_Tpredicate);

        memset(m_Tpredicate, 0,
                sizeof(char) * m_numScan * m_numScan);

        // mark nearby detections
        for(i = 0; i < m_numScan; i++)
        {
            ni = i * m_numScan;
            for(j = 0; j < m_numScan; j++)
            {
                nj = j * m_numScan;
                h = min(m_ys[i] + m_sizes[i],
                        m_ys[j] + m_sizes[j]) -
                    max(m_ys[i], m_ys[j]);
                w = min(m_xs[i] + m_sizes[i],
                        m_xs[j] + m_sizes[j]) -
                    max(m_xs[i], m_xs[j]);
                s = max(double(h),0.0) * max(double(w),0.0);
                si = m_sizes[i]*m_sizes[i];
                sj = m_sizes[j]*m_sizes[j];

                // 1. Overlap 50%
                if((s / (si + sj - s)) >=
                        m_overlappingThreshold)
                {
                    m_Tpredicate[ni + j] = 1;
                    m_Tpredicate[nj + i] = 1;
                }

                // 2. Overlap 80% of small one.
                //if(s / si >= 0.8 || s / sj >= 0.8)
                //{
                //    m_Tpredicate[ni + j] = 1;
                //    m_Tpredicate[nj + i] = 1;
                //}
            }
        }

        //int* root = (int*)malloc(sizeof(int) * m_numScan);
        for(i = 0; i < m_numScan; i++)
            m_Troot[i] = -1;
        int n = partition(m_Tpredicate, m_Troot);
        if(n > m_maxDetectNum)
            mallocdetectspace(n + 40);

        //float* logweight = (float*)malloc(sizeof(float) * m_numScan);
        for(i = 0; i < m_numScan; i++)
        {
            m_Tlogweight[i] = logistic(m_scores[i]);
        }

        //int* neighbors = (int*)malloc(sizeof(int) * n);
        //float* weight = (float*)malloc(sizeof(float) * n);
        //float* xs = (float*)malloc(sizeof(float) * n);
        //float* ys = (float*)malloc(sizeof(float) * n);
        //float* ss = (float*)malloc(sizeof(float) * n);
        memset(m_Tweight, 0, sizeof(float) * n);
        memset(m_Tneighbors, 0, sizeof(int) * n);
        memset(m_Txs, 0, sizeof(float) * n);
        memset(m_Tys, 0, sizeof(float) * n);
        memset(m_Tss, 0, sizeof(float) * n);
        for(i = 0; i < m_numScan; i++)
        {
            m_Tweight[m_Troot[i]] += m_Tlogweight[i];
            m_Tneighbors[m_Troot[i]] += 1;
        }

        for(i = 0; i < m_numScan; i++)
        {
            if(m_Tweight[m_Troot[i]] != 0)
                m_Tlogweight[i] /= m_Tweight[m_Troot[i]];
            else
                m_Tlogweight[i] = 1 / m_Tneighbors[m_Troot[i]];
            m_Txs[m_Troot[i]] += m_xs[i] * m_Tlogweight[i];
            m_Tys[m_Troot[i]] += m_ys[i] * m_Tlogweight[i];
            m_Tss[m_Troot[i]] += m_sizes[i] * m_Tlogweight[i];
        }

        //printf("Detect %d faces:\n", n);
        for(i = 0; i < n; i++)
        {
            //printf("%fx%fx%fx%f %f\n", xs[i], ys[i], ss[i], ss[i], weight[i]);
            m_Xs.push_back(int(m_Txs[i]));
            m_Ys.push_back(int(m_Tys[i]));
            m_Ss.push_back(int(m_Tss[i]));
            m_Scores.push_back((m_Tweight[i]));
        }

//        free(predicate);
//        free(root);
//        free(logweight);
//        free(neighbors);
//        free(weight);
//        free(xs);
//        free(ys);
//        free(ss);

        return n;
    }

    int findRoot(int* parent, int i)
    {
        if(parent[i] != i)
            return findRoot(parent, parent[i]);
        else
            return i;
    }

    int npddetect::partition(char* predicate, int* root)
    {
        //int* parent = (int*)malloc(sizeof(int) * m_numScan);
        //int* rank = (int*)malloc(sizeof(int) * m_numScan);
        int i, j, ni;
        int root_i, root_j;
        for(i = 0; i < m_numScan; i++)
            m_Tparent[i] = i;
        m_Trank[i] = 0;

        ni = 0;
        for(i = 0; i < m_numScan; i++)
        {
            for(j = 0; j < m_numScan; j++, ni++)
            {
                if(predicate[ni] == 0)
                    continue;

                root_i = findRoot(m_Tparent, i);
                root_j = findRoot(m_Tparent, j);

                if(root_i != root_j)
                {
                    if(m_Trank[i] > m_Trank[j])
                        m_Tparent[root_j] = root_i;
                    else if(m_Trank[i] < m_Trank[j])
                        m_Tparent[root_i] = root_j;
                    else
                    {
                        m_Tparent[root_j] = root_i;
                        m_Trank[root_i] ++;
                    }
                }
            }
        }

        int n = 0;
        for(i = 0; i < m_numScan; i++)
        {
            if(m_Tparent[i] == i)
            {
                if(root[i] == -1)
                    root[i] = n++;
                continue;
            }

            root_i = findRoot(m_Tparent, i);
            if(root[root_i] == -1)
                root[root_i] = n++;
            root[i] = root[root_i];
        }

        //free(rank);
        //free(parent);

        return n;
    }

}
