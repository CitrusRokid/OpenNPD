
/********************************************************************
 > File Name: npdmodel.h
 > Author: cmxnono
 > Mail: 381880333@qq.com
 > Created Time: 17/02/2016 
 *********************************************************************/

#ifndef _NPD_BASE_NPDMODEL_H_
#define _NPD_BASE_NPDMODEL_H_

#include "npdtable.h"

namespace npd{

    class npdmodel{

        public:

            npdmodel();
            
            ~npdmodel();

            npdmodel(const char* modelpath);

            void init();

            void release();
    
            void prepare(int os, int nst, int nb, int nl, float sf, int nsa);

            void load(const char* modelpath);

            int m_objSize;
            int m_numStages;
            int m_numBranchNodes;
            int m_numLeafNodes;
            int m_numScales;
            float m_scaleFactor;
            
            unsigned char** m_cutpoint;

            int* m_winSize;
            int* m_treeRoot;
            int* m_leftChild;
            int* m_rightChild;
            int** m_pixelx;
            int** m_pixely;

            float* m_stageThreshold;
            float* m_fit;
            
    };
}

#endif
