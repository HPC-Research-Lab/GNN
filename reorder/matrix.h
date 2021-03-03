#ifndef GCONV_MATRIX_H
#define GCONV_MATRIX_H

#include <iostream>
#include <iomanip>      
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <sys/time.h>
#include <set>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <memory>
#include "config.h"
#include "util.h"


template <class T>
struct SparseMatrixCSR_CPU {
	int nrows, ncols, nnz;
	vector<int> rowptr;
	vector<int> colidx;
	vector<T> values;

	SparseMatrixCSR_CPU():nrows(0), ncols(0), nnz(0) {}

	void loadCOO(string filename, bool start_one=false) {

		ifstream fin(filename.c_str());
		string line;
		getline(fin, line);
		stringstream sin_meta(line);
		sin_meta >> nrows >> ncols;

		rowptr.resize(nrows + 1);

		map<int, vector<pair<int, T>>> dict;
		while (getline(fin, line)) {
			int r, c; 
			T v;
			stringstream sin(line);
			sin >> r >> c >> v;
			if (start_one) {
				r--; c--;
			}
			if (dict.find(r) == dict.end()) {
				dict[r] = vector<pair<int, T>>();
			}
			dict[r].push_back(make_pair(c, v));
		}

		rowptr[0] = 0;
		nnz = 0;
		for (int i=1; i<nrows+1; i++) {
			rowptr[i] = rowptr[i-1] + dict[i-1].size();
			for (int j=0; j<dict[i-1].size(); j++) {
				colidx.push_back(dict[i-1][j].first);
				values.push_back(dict[i-1][j].second);	
				nnz++;
			}
		}
	}


	vector<pair<int, map<int, T>>> get_rowmap() {

		vector<pair<int, map<int, T>>> rowlist;
		for (int i=0; i<nrows; i++) {
			rowlist.push_back(make_pair(i, map<int, T>()));
			for (int j=rowptr[i]; j<rowptr[i+1]; j++) {
				rowlist[i].second[colidx[j]] = values[j];
			}
		}
		return rowlist;
	}

};

#endif
