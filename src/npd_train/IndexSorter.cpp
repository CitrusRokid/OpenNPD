#include "common.h"
using namespace std;

vector<int> *s_index;
vector<double> *s_content;

bool IndexSorterCompareUp(int i, int j) {
	return ((s_content->at(i)) < (s_content->at(j)));
}

bool IndexSorterCompareDown(int i, int j) {
	return ((s_content->at(i)) > (s_content->at(j)));
}

void IndexSort(vector<int>& _index, vector<double>& _content, int direction) {
	s_index = &_index;
	s_content = &_content;
	if (direction > 0)
	{
		sort(s_index->begin(), s_index->end(), IndexSorterCompareUp);
	}
	else
	{
		sort(s_index->begin(), s_index->end(), IndexSorterCompareDown);
	}
};
