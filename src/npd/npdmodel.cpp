
/********************************************************************
 > File Name: npdmodel.cpp
 > Author: cmxnono
 > Mail: 381880333@qq.com
 > Created Time: 17/02/2016 
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "npd/npdmodel.h"


namespace npd{

    npdmodel::npdmodel()
    {
        init();
    }

    npdmodel::npdmodel(const char* modelpath)
    {
        init();
        load(modelpath);
    }

    npdmodel::~npdmodel()
    {
        release();
    }

    void npdmodel::init()
    {
        m_objSize = 0;
        m_numStages = 0;
        m_numScales = 0;
        m_numLeafNodes = 0;
        m_numBranchNodes = 0;
        m_scaleFactor = 0.0;

        m_cutpoint = NULL;
        m_winSize = NULL;
        m_treeRoot = NULL;
        m_leftChild = NULL;
        m_rightChild = NULL;
        m_pixelx = NULL;
        m_pixely = NULL;
        m_stageThreshold = NULL;
        m_fit = NULL;

        // ******************** TEMP *********************
        //prepare();
    }

    void npdmodel::release()
    {
        int i;
        if(m_cutpoint != NULL)
        {
            for(i = 0; i < 2; i++)
            {
                free(m_cutpoint[i]);
                m_cutpoint[i] = NULL;
            }
            free(m_cutpoint);
        }
        m_cutpoint = NULL;

        if(m_winSize != NULL)
        {
            free(m_winSize);
        }
        m_winSize = NULL;

        if(m_treeRoot != NULL)
        {
            free(m_treeRoot);
        }
        m_treeRoot = NULL;

        if(m_leftChild != NULL)
        {
            free(m_leftChild);
        }
        m_leftChild = NULL;

        if(m_rightChild != NULL)
        {
            free(m_rightChild);
        }
        m_rightChild = NULL;

        if(m_pixelx != NULL)
        {
            for(i = 0; i < m_numScales; i++)
            {
                free(m_pixelx[i]);
                m_pixelx[i] = NULL;
            }
            free(m_pixelx);
        }
        m_pixelx = NULL;

        if(m_pixely != NULL)
        {
            for(i = 0; i < m_numScales; i++)
            {
                free(m_pixely[i]);
                m_pixely[i] = NULL;
            }
            free(m_pixely);
        }
        m_pixely = NULL;
        
        if(m_stageThreshold != NULL)
        {
            free(m_stageThreshold);
        }
        m_stageThreshold = NULL;

        if(m_fit != NULL)
        {
            free(m_fit);
        }
        m_fit = NULL;

        m_objSize = 0;
        m_numStages = 0;
        m_numScales = 0;
        m_numLeafNodes = 0;
        m_numBranchNodes = 0;
        m_scaleFactor = 0.0;

    }

    void npdmodel::prepare(int os, int nst, int nb, int nl, 
            float sf, int nsa)
    {
//        m_objSize = 20;
//        m_numStages = 72;
//        m_numBranchNodes = 1018;
//        m_numLeafNodes = 1090;
//        m_scaleFactor = 1.2;
//        m_numScales = 30;

        m_objSize = os;
        m_numStages = nst;
        m_numBranchNodes = nb;
        m_numLeafNodes = nl;
        m_scaleFactor = sf;
        m_numScales = nsa;

        int i;


        m_stageThreshold = (float*)malloc(m_numStages * sizeof(float));
        m_treeRoot = (int*)malloc(m_numStages * sizeof(int));

        m_pixelx = (int**)malloc(m_numScales * sizeof(int*));
        m_pixely = (int**)malloc(m_numScales * sizeof(int*));
        for(i = 0; i < m_numScales; i++)
        {
            m_pixelx[i] = (int*)malloc(m_numBranchNodes * sizeof(int));
            m_pixely[i] = (int*)malloc(m_numBranchNodes * sizeof(int));
        }

        m_cutpoint = (unsigned char**)malloc(2 * sizeof(unsigned char*));
        for(i = 0; i < 2; i++)
        {
            m_cutpoint[i] = (unsigned char*)malloc(m_numBranchNodes * sizeof(unsigned char));
        }
        
        m_leftChild = (int*)malloc(m_numBranchNodes * sizeof(int));
        m_rightChild = (int*)malloc(m_numBranchNodes * sizeof(int));

        m_fit = (float*)malloc(m_numLeafNodes * sizeof(float));

        m_winSize = (int*)malloc(m_numScales * sizeof(int));
    }

    void npdmodel::load(const char* modelpath)
    {
	    //printf("Load models:%s ...\n", modelpath);

	    int n = 0;
	    int i;
	    int os, nst, nb, nl, nsa; 
	    float sf;
	    FILE* fp = fopen(modelpath, "rb");
	    size_t rs;

	    rs = fread(&os, sizeof(int), 1, fp);
	    rs = fread(&nst, sizeof(int), 1, fp);
	    rs = fread(&nb, sizeof(int), 1, fp);
	    rs = fread(&nl, sizeof(int), 1, fp);
	    rs = fread(&sf, sizeof(float), 1, fp);
	    rs = fread(&nsa, sizeof(int), 1, fp);

	    // Malloc space.
	    prepare(os, nst, nb, nl, sf, nsa);

	    rs = fread(m_stageThreshold, sizeof(float), m_numStages, fp);
	    n += rs;
	    rs = fread(m_treeRoot, sizeof(int), m_numStages, fp);
	    n += rs;
	    for(i = 0; i < m_numScales; i++)
	    {
		    rs = fread(m_pixelx[i], sizeof(int), m_numBranchNodes, fp);
		    n += rs;
	    }
	    for(i = 0; i < m_numScales; i++)
	    {
		    rs = fread(m_pixely[i], sizeof(int), m_numBranchNodes, fp);
		    n += rs;
	    }
	    for(i = 0; i < 2; i++)
	    {
		    rs = fread(m_cutpoint[i], sizeof(unsigned char), m_numBranchNodes, fp);
		    n += rs;
	    }
	    rs = fread(m_leftChild, sizeof(int), m_numBranchNodes, fp);
	    n += rs;
	    rs = fread(m_rightChild, sizeof(int), m_numBranchNodes, fp);
	    n += rs;
	    rs = fread(m_fit, sizeof(float), m_numLeafNodes, fp);
	    n += rs;
	    rs = fread(m_winSize, sizeof(int), m_numScales, fp);
	    n += rs;

	    fclose(fp);
        //printf("Models loaded(load %d bytes)!!!\n", n);
    }
}
