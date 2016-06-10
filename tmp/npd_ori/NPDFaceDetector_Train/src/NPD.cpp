/* Mex function for computing the normalized pixel difference (NPD) feature. The NPD look up table is used.
 *
 * features = NPD(patches)
 * 
 * Input:
 * 
 *   patches: the original input images/patches. Size: [height, width, numImgs].
 *
 * Output:
 *
 *   features: the computed NPD features. Size: [numImgs, feaDims].
 *
 * Compile:
 *     Windows: mex -largeArrayDims -DWINDOWS 'COMPFLAGS=/openmp $COMPFLAGS' NPD.cpp -v
 *     Linux:  mex -largeArrayDims -DLINUX 'CFLAGS=\$CFLAGS -fopenmp' 'LDFLAGS=\$LDFLAGS -fopenmp' NPD.cpp -v
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

#include <omp.h>
#include <math.h>
#include <vector>

using namespace std;

#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{    
    if(nlhs > 1 || nrhs != 1)
    {
        mexErrMsgTxt("Usage: features = NPD(patches)");
    }
    
    // Set the number of threads
    int numProcs = omp_get_num_procs();
    int numThreads = (int) floor(numProcs * 0.8);    
    omp_set_num_threads(numThreads);
    
    unsigned char ppNpdTable[256][256];
    
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
            double fea = 0.5;            
            if(i > 0 || j > 0) fea = double(i) / (double(i) + double(j));            
            fea = floor(256 * fea);
            if(fea > 255) fea = 255;
            
            ppNpdTable[i][j] = (unsigned char) fea;
        }
    }
    
    
    /* Validate the input data type */
    if(!mxIsUint8(prhs[0]))
    {
        mexErrMsgTxt("The data type of the input array must be uint8.");
    }    
    
    if(mxGetNumberOfDimensions(prhs[0]) < 2 || mxGetNumberOfDimensions(prhs[0]) > 3)
    {
        mexErrMsgTxt("The input array must be a matrix or a 3-dim array.");
    }
    
    size_t height = mxGetDimensions(prhs[0])[0];
    size_t width = mxGetDimensions(prhs[0])[1];
    size_t numPixels = height * width;
    
    size_t numImgs = 1;
    if(mxGetNumberOfDimensions(prhs[0]) == 3) numImgs = mxGetDimensions(prhs[0])[2];
    
    size_t feaDims = numPixels * (numPixels - 1) / 2;
    
    const unsigned char *pPatches = (unsigned char *) mxGetData(prhs[0]);
    
    /* Create mxArrays for the output data and initialize to 0 */
    plhs[0] = mxCreateNumericMatrix(numImgs, feaDims, mxUINT8_CLASS, mxREAL);
    
    /* Create pointers to the output data */
    vector<unsigned char *> ppFea(feaDims);
    ppFea[0] = (unsigned char*) mxGetData(plhs[0]);
    for(int i = 1; i < feaDims; i++) ppFea[i] = ppFea[i-1] + numImgs;
        
    #pragma omp parallel for

    // process each image
    for(int k = 0; k < numImgs; k++)
    {
        const unsigned char *pPixel = pPatches + numPixels * size_t(k);
        int d = 0;
        
        for(int i = 0; i < numPixels; i++)
        {
            for(int j = i + 1; j < numPixels; j++)
            {
                ppFea[d++][k] = ppNpdTable[pPixel[i]][pPixel[j]];
            }
        }
    }
}
