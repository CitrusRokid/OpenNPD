#include "common.h"
#include "opencv2/core/core.hpp"
#include <iostream>

using namespace std;

double TestSubTree(Stage& tree,const cv::Mat& x, int node) {
	double score;

	if (tree.feaId.empty()) {
		return 0;
	}

	if (node < 0) {
		score = tree.fit[-node-1];
	} 
	else{
		int mNode = node;

		unsigned char * ptr = x.data;

		bool isLeft = (  ptr[tree.feaId[mNode]] < tree.cutpoint[0][mNode]  || ptr[tree.feaId[mNode]] > tree.cutpoint[1][mNode]);
		
		if (isLeft) {
			score = TestSubTree(tree, x, tree.leftChild[mNode]);
		}
		else {
			score = TestSubTree(tree, x, tree.rightChild[mNode]);
		}
	}

	return score;

}


double TestDQT(Stage& tree, const cv::Mat& x) {
	
	double score;

	if (tree.feaId.empty()) {
			score = tree.fit[0];
	}
	else {
		score = TestSubTree(tree, x, 0);
	}

	return score;
}
