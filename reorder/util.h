//
// Created by Peng Jiang on 2019-03-03.
//

#ifndef GCONV_UTIL_H
#define GCONV_UTIL_H

#include "config.h"
#include <iostream>
using namespace std;


inline int DIV(int x, int tile_size) {
	if (x % tile_size) return x / tile_size + 1; else return x / tile_size; } 

inline int CLAMP(int x, int thd=MAX_TPB) {return thd<x? thd : x; }

template <class T>
T MIN(T x, T y) { return x<y?x:y; }

template <class T>
T MAX(T x, T y) { return x>y?x:y; }


#define cudaCheckError() {                                          \
	cudaError_t e=cudaGetLastError();                                 \
	if(e!=cudaSuccess) {                                              \
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
		exit(0); \
	}                                                                 \
}

template <class T>
void print(vector<T> &v) {
	for (int i=0; i<MIN(50, (int)(v.size())); i++) cout << v[i] << " ";
	cout << endl;
}

struct blist {
	void *left = NULL;
	void *right = NULL;

	blist(int i) {
		left = new int(i);
	};

	blist(blist &l1, blist &l2) {
		left = &l1;
		right = &l2;
	}
	
};

#endif
