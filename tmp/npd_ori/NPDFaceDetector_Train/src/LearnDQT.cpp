/* Mex function for deep quadratic tree (DQT) learning
 *
 * [feaId, cutpoint, leftChild, rightChild, fit, minCost] = 
 *      LearnDQT(posX, negX, posW, negW, posFx, negFx, posIndex, negIndex [, treeLevel, minLeaf, numThreads])
 * 
 * Input:
 * 
 *   <posX>: features of the positive samples. Size: [nPos, d].
 *   <negX>: features of the negative samples. Size: [nNeg, d].
 *   <posW>: weights of the positive samples. Size: [nPos, 1].
 *   <negW>: weights of the negative samples. Size: [nNeg, 1].
 *   <posFx>: current AdaBoost scores of the positive samples. Size: [nPos, 1].
 *   <negFx>: current AdaBoost scores of the negative samples. Size: [nNeg, 1].
 *   <posIndex>: indexes of the current positive samples after rejection, 
 *      random selection, and weight trimming. Size: [1, nCurrPos].
 *   <negIndex>: indexes of the current negative samples after rejection, 
 *      random selection, and weight trimming. Size: [1, nCurrNeg].
 *      These two vectors are used to avoid frequently copying a subset of the positive/negative samples for training. 
 *   [treeLevel]: the maximal depth of the DQT trees to be learned. Default: 4.
 *   [minLeaf]: minimal samples required in each leaf node. This is used to avoid overfitting. Default: 100.
 *   [numThreads]: the number of computing threads in tree learning. Positive double scalar. Default: the number of all available processors.
 *
 * Output:
 *
 *   feaId: the learned/selected optimal feature indexes in constructing the DQT tree.
 *   cutpoint: thresholds in splitting the tree nodes. Each branch has two thresholds.
 *   leftChild: index of the left child in each branch.
 *   rightChild: index of the right child in each branch.
 *   fit: fitting values of each leaf node.
 *   minCost: the corresponding minimal cost in the optimization.
 *
 * Compile:
 *     Windows: mex -largeArrayDims -DWINDOWS 'COMPFLAGS=/openmp $COMPFLAGS' LearnDQT.cpp -v
 *     Linux:  mex -largeArrayDims -DLINUX 'CFLAGS=\$CFLAGS -fopenmp' 'LDFLAGS=\$LDFLAGS -fopenmp' LearnDQT.cpp -v
 * 
 * Reference:
 *     Shengcai Liao, Anil K. Jain, and Stan Z. Li, "A Fast and Accurate Unconstrained Face Detector," 
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015 (Accepted).
 *
 * Version: 1.0
 * Date: 2015-10-04
 *
 * Author: Shengcai Liao
 * Institute: National Laboratory of Pattern Recognition,
 *   Institute of Automation, Chinese Academy of Sciences
 * Email: scliao@nlpr.ia.ac.cn
 *
 * ----------------------------------
 * Copyright (c) 2015 Shengcai Liao
 * ----------------------------------
*/

#include "mex.h"
#include <omp.h>
#include <math.h>
#include <string.h>
#include <vector>

using namespace std;

void WeightHist(unsigned char *X, float *W, int *index, int n, int count[256], float wHist[256]);

float LearnQuadStump(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, float *posFx, float *negFx, 
        int *posIndex, int *negIndex, int nPos, int nNeg, int minLeaf, int numThreads, float parentFit,
        int &feaId, unsigned char (&cutpoint)[2], float (&fit)[2]);

float LearnDQT(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, float *posFx, float *negFx, 
        int *posIndex, int *negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, float parentFit,
        vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<float> &fit);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{    
    if(nlhs != 6 || nrhs < 8)
    {
        mexErrMsgTxt("Usage: [feaId, cutpoint, leftChild, rightChild, fit, minCost] = LearnDQT(posX, negX, posW, negW, posFx, negFx, posIndex, negIndex, treeLevel, minLeaf, numThreads)");
    }
    
    int treeLevel = 4;
    int minLeaf = 100;
    
    if(nrhs >= 9 && mxGetScalar(prhs[8]) > 0) treeLevel = (int) mxGetScalar(prhs[8]);
    if(nrhs >= 10 && mxGetScalar(prhs[9]) > 0) minLeaf = (int) mxGetScalar(prhs[9]);
    
    // Set the number of threads
    int numProcs = omp_get_num_procs();
    int numThreads = numProcs;
    
    if(nrhs >= 11 && mxGetScalar(prhs[10]) > 0) numThreads = (int) mxGetScalar(prhs[10]);
    
    if(numThreads > numProcs) numThreads = numProcs;
    
    /* Validate the input data type */
    if(!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]) ||
            !mxIsSingle(prhs[2]) || !mxIsSingle(prhs[3]) ||
            !mxIsSingle(prhs[4]) || !mxIsSingle(prhs[5]) ||
            !mxIsInt32(prhs[6]) || !mxIsInt32(prhs[7]))
    {
        mexErrMsgTxt("The data types of the input arrays are not acceptable.");
    }
    
    int nTotalPos = (int) mxGetM(prhs[0]);
    int nTotalNeg = (int) mxGetM(prhs[1]);
    int feaDims = (int) mxGetN(prhs[0]);
    int nPos = (int) mxGetNumberOfElements(prhs[6]);
    int nNeg = (int) mxGetNumberOfElements(prhs[7]);
    
    // get input pointers
    vector<unsigned char *> ppPosX(feaDims);
    ppPosX[0] = (unsigned char *) mxGetData(prhs[0]);
    for(int i = 1; i < feaDims; i++) ppPosX[i] = ppPosX[i-1] + nTotalPos;
    
    vector<unsigned char *> ppNegX(feaDims);
    ppNegX[0] = (unsigned char *) mxGetData(prhs[1]);
    for(int i = 1; i < feaDims; i++) ppNegX[i] = ppNegX[i-1] + nTotalNeg;
    
    float *pPosW = (float *) mxGetData(prhs[2]);
    float *pNegW = (float *) mxGetData(prhs[3]);
    float *pPosFx = (float *) mxGetData(prhs[4]);
    float *pNegFx = (float *) mxGetData(prhs[5]);
    int *pPosIndex = (int *) mxGetData(prhs[6]);
    int *pNegIndex = (int *) mxGetData(prhs[7]);
    
    vector<int> feaId, leftChild, rightChild;
    vector< vector<unsigned char> > cutpoint;
    vector<float> fit;

    float minCost = LearnDQT(ppPosX, ppNegX, pPosW, pNegW, pPosFx, pNegFx, pPosIndex, pNegIndex, nPos, nNeg, treeLevel, minLeaf, numThreads, 0, 
        feaId, cutpoint, leftChild, rightChild, fit);
    
    int numBranchNodes = (int) feaId.size();
    int numLeafNodes = (int) fit.size();
    
    /* Create mxArrays for the output data and initialize to 0 */
    plhs[0] = mxCreateDoubleMatrix(numBranchNodes, 1, mxREAL);
    plhs[1] = mxCreateNumericMatrix(2, numBranchNodes, mxUINT8_CLASS, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(numBranchNodes, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(numBranchNodes, 1, mxREAL);
    plhs[4] = mxCreateNumericMatrix(numLeafNodes, 1, mxSINGLE_CLASS, mxREAL);
    plhs[5] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    *(float*) mxGetData(plhs[5]) = minCost;
    
    /* Create pointers to the output data */
    double *pFeaId = (double*) mxGetData(plhs[0]);
    unsigned char *pCutpoint = (unsigned char*) mxGetData(plhs[1]);
    double *pLeftChild = (double*) mxGetData(plhs[2]);
    double *pRightChild = (double*) mxGetData(plhs[3]);
    float *pFit = (float*) mxGetData(plhs[4]);
    
    for(int i = 0; i < numBranchNodes; i++)
    {
        pFeaId[i] = feaId[i] + 1;
        pCutpoint[i*2] = cutpoint[i][0];
        pCutpoint[i*2+1] = cutpoint[i][1];
        pLeftChild[i] = leftChild[i];
        pRightChild[i] = rightChild[i];        
    }
    
    for(int i = 0; i < numLeafNodes; i++) pFit[i] = fit[i];
}


float LearnDQT(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, float *posFx, float *negFx, 
        int *posIndex, int *negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, float parentFit,
        vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<float> &fit)
{
    int _feaId;
    unsigned char _cutpoint[2];
    float _fit[2];
    
    float minCost = LearnQuadStump(posX, negX, posW, negW, posFx, negFx, posIndex, negIndex, nPos, nNeg, minLeaf, numThreads, parentFit, 
        _feaId, _cutpoint, _fit);

    if(_feaId < 0) return minCost;
    
    feaId.push_back(_feaId);
    cutpoint.push_back(vector<unsigned char>(_cutpoint, _cutpoint+2));
    leftChild.push_back(-1);
    rightChild.push_back(-2);

    if(treeLevel <= 1)
    {
        fit.push_back(_fit[0]);
        fit.push_back(_fit[1]);        
        return minCost;
    }

    int nPos1 = 0, nNeg1 = 0, nPos2 = 0, nNeg2 = 0;
    vector<int> posIndex1(nPos), posIndex2(nPos), negIndex1(nNeg), negIndex2(nNeg);

    for(int j = 0; j < nPos; j++)
    {
        if(posX[size_t(_feaId)][size_t(posIndex[j])] < _cutpoint[0] || posX[size_t(_feaId)][size_t(posIndex[j])] > _cutpoint[1])
        {
            posIndex1[nPos1++] = posIndex[j];
        }
        else
        {
            posIndex2[nPos2++] = posIndex[j];
        }
    }

    for(int j = 0; j < nNeg; j++)
    {
        if(negX[size_t(_feaId)][size_t(negIndex[j])] < _cutpoint[0] || negX[size_t(_feaId)][size_t(negIndex[j])] > _cutpoint[1])
        {
            negIndex1[nNeg1++] = negIndex[j];
        }
        else
        {
            negIndex2[nNeg2++] = negIndex[j];
        }
    }

    vector<int> feaId1, feaId2, leftChild1, leftChild2, rightChild1, rightChild2;
    vector< vector<unsigned char> > cutpoint1, cutpoint2;
    vector<float> fit1, fit2;

    float minCost1 = LearnDQT(posX, negX, posW, negW, posFx, negFx, &posIndex1[0], &negIndex1[0], nPos1, nNeg1, treeLevel - 1, minLeaf, numThreads, _fit[0],
        feaId1, cutpoint1, leftChild1, rightChild1, fit1);

    float minCost2 = LearnDQT(posX, negX, posW, negW, posFx, negFx, &posIndex2[0], &negIndex2[0], nPos2, nNeg2, treeLevel - 1, minLeaf, numThreads, _fit[1],
        feaId2, cutpoint2, leftChild2, rightChild2, fit2);

    if(feaId1.empty() && feaId2.empty())
    {
        fit.push_back(_fit[0]);
        fit.push_back(_fit[1]);        
        return minCost;
    }

    if(minCost1 + minCost2 >= minCost)
    {
        fit.push_back(_fit[0]);
        fit.push_back(_fit[1]);
        return minCost;
    }

    minCost = minCost1 + minCost2;

    if(feaId1.empty())
    {
        fit.push_back(_fit[0]);
    }
    else
    {
        feaId.insert(feaId.end(), feaId1.begin(), feaId1.end());
        cutpoint.insert(cutpoint.end(), cutpoint1.begin(), cutpoint1.end());
        fit = fit1;
        
        for(int i = 0; i < leftChild1.size(); i++)
        {
            if(leftChild1[i] >= 0) leftChild1[i]++;
            if(rightChild1[i] >= 0) rightChild1[i]++;
        }
        
        leftChild[0] = 1;
        leftChild.insert(leftChild.end(), leftChild1.begin(), leftChild1.end());
        rightChild.insert(rightChild.end(), rightChild1.begin(), rightChild1.end());
    }

    int numBranchNodes = (int) feaId.size();
    int numLeafNodes = (int) fit.size();

    if(feaId2.empty())
    {
        fit.push_back(_fit[1]);
        rightChild[0] = -(numLeafNodes + 1);
    }
    else
    {
        feaId.insert(feaId.end(), feaId2.begin(), feaId2.end());
        cutpoint.insert(cutpoint.end(), cutpoint2.begin(), cutpoint2.end());
        fit.insert(fit.end(), fit2.begin(), fit2.end());
        
        for(int i = 0; i < leftChild2.size(); i++)
        {
            if(leftChild2[i] >= 0) leftChild2[i] += numBranchNodes;
            else leftChild2[i] -= numLeafNodes;
            
            if(rightChild2[i] >= 0) rightChild2[i] += numBranchNodes;
            else rightChild2[i] -= numLeafNodes;
        }
        
        leftChild.insert(leftChild.end(), leftChild2.begin(), leftChild2.end());
        rightChild[0] = numBranchNodes;
        rightChild.insert(rightChild.end(), rightChild2.begin(), rightChild2.end());
    }
    
    return minCost;
}


float LearnQuadStump(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, float *posFx, float *negFx, 
        int *posIndex, int *negIndex, int nPos, int nNeg, int minLeaf, int numThreads, float parentFit,
        int &feaId, unsigned char (&cutpoint)[2], float (&fit)[2])
{
    float w = 0;
    for(int i = 0; i < nPos; i++) w += posW[ posIndex[i] ];
    float minCost = w * (parentFit - 1) * (parentFit - 1);
    
    w = 0;
    for(int i = 0; i < nNeg; i++) w += negW[ negIndex[i] ];
    minCost += w * (parentFit + 1) * (parentFit + 1);
    
    feaId = -1;
    if(nPos == 0 || nNeg == 0 || nPos + nNeg < 2 * minLeaf) return minCost;

    int feaDims = (int) posX.size();
    minCost = 1e16f;

    omp_set_num_threads(numThreads);
    #pragma omp parallel for

    // process each dimension
    for(int i = 0; i < feaDims; i++)
    {
        int count[256];
        float posWHist[256];
        float negWHist[256];
        
        memset(count, 0, 256 * sizeof(int));
        
        WeightHist(posX[i], posW, posIndex, nPos, count, posWHist);
        WeightHist(negX[i], negW, negIndex, nNeg, count, negWHist);
        
        float posWSum = 0;
        float negWSum = 0;
        
        for(int bin = 0; bin < 256; bin++)
        {
            posWSum += posWHist[bin];
            negWSum += negWHist[bin];
        }        
        
        int totalCount = nPos + nNeg;
        float wSum = posWSum + negWSum;
        
        float minMSE = 3.4e38f;
        int thr0 = -1, thr1;
        float fit0, fit1;
        
        for(int v = 0; v < 256; v++) // lower threshold
        {
            int rightCount = 0;
            float rightPosW = 0;
            float rightNegW = 0;
            
            for(int u = v; u < 256; u++) // upper threshold
            {
                rightCount += count[u];
                rightPosW += posWHist[u];
                rightNegW += negWHist[u];
                
                if(rightCount < minLeaf) continue;
                
                int leftCount = totalCount - rightCount;
                if(leftCount < minLeaf) break;                
                
                float leftPosW = posWSum - rightPosW;
                float leftNegW = negWSum - rightNegW;
                
                float leftFit, rightFit;
                
                if(leftPosW + leftNegW <= 0) leftFit = 0.0f;
                else leftFit = (leftPosW - leftNegW) / (leftPosW + leftNegW);
                
                if(rightPosW + rightNegW <= 0) rightFit = 0.0f;
                else rightFit = (rightPosW - rightNegW) / (rightPosW + rightNegW);
                
                float leftMSE = leftPosW * (leftFit - 1) * (leftFit - 1) + leftNegW * (leftFit + 1) * (leftFit + 1);
                float rightMSE = rightPosW * (rightFit - 1) * (rightFit - 1) + rightNegW * (rightFit + 1) * (rightFit + 1);
                
                float mse = leftMSE + rightMSE;
                
                if(mse < minMSE)
                {
                    minMSE = mse;
                    thr0 = v;
                    thr1 = u;
                    fit0 = leftFit;
                    fit1 = rightFit;
                }
            }
        }
        
        if(thr0 == -1) continue;
        
        if(minMSE < minCost)
        {
            #pragma omp critical // modify the record by a single thread
            {
                minCost = minMSE;
                feaId = i;
                cutpoint[0] = (unsigned char) thr0;
                cutpoint[1] = (unsigned char) thr1;
                fit[0] = fit0;
                fit[1] = fit1;
            }
        }
    } // end omp parallel for feaDims
    
    return minCost;
}


void WeightHist(unsigned char *X, float *W, int *index, int n, int count[256], float wHist[256])
{
    memset(wHist, 0, 256 * sizeof(float));
    
    for(int j = 0; j < n; j++)
    {
        unsigned char bin = X[ index[j] ];
        count[bin]++; 
        wHist[bin] += W[ index[j] ];
    }
}
