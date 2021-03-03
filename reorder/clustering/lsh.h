#ifndef _LSH_H
#define _LSH_H


#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <set>
#include <queue>
#include <omp.h>

using namespace std;

class LSH {

	/* This function calculates (ab)%c */
	static int modulo(int a,int b,int c){
	    long long x=1,y=a; // long long is taken to avoid overflow of intermediate results
	    while(b > 0){
		        if(b%2 == 1){
				            x=(x*y)%c;
				        }
		        y = (y*y)%c; // squaring the base
		        b /= 2;
		    }
	    return x%c;
	}

	/* this function calculates (a*b)%c taking into account that a*b might overflow */
	static long long mulmod(long long a,long long b,long long c){
	    long long x = 0,y=a%c;
	    while(b > 0){
		        if(b%2 == 1){
				            x = (x+y)%c;
				        }
		        y = (y*2)%c;
		        b /= 2;
		    }
	    return x%c;
	}

	/* Miller-Rabin primality test, iteration signifies the accuracy of the test */
	static bool Miller(long long p,int iteration){
	    if(p<2){
		        return false;
		    }
	    if(p!=2 && p%2==0){
		        return false;
		    }
	    long long s=p-1;
	    while(s%2==0){
		        s/=2;
		    }
	    for(int i=0;i<iteration;i++){
		        long long a=rand()%(p-1)+1,temp=s;
		        long long mod=modulo(a,temp,p);
		        while(temp!=p-1 && mod!=1 && mod!=p-1){
				            mod=mulmod(mod,mod,p);
				            temp *= 2;
				        }
		        if(mod!=p-1 && temp%2==0){
				            return false;
				        }
		    }
	    return true;
	}

	static int first_prime_greater_than(int input) {
	    int i = 0;
	
	    if(input%2==0)
	        i = input+1;
	    else i = input;
	
	    for(;i<2*input;i+=2) // from Rajendra's answer
	        if(Miller(i,20)) // 18-20 iterations are enough for most of the applications.
	            break;
	    return i;
	}

	class PermHash {
		long long a, b, c, mv;

		public:
		PermHash(int max_values) {
			a = rand() % max_values;
			b = rand() % max_values;
			c = LSH::first_prime_greater_than(max_values);
			mv = max_values;
		}

		uint32_t operator()(int x) {
			uint32_t r = (a * x + b) % c;
			if (r >= mv) r %= mv;
			return r;
		}

	};

	class MinHash {
		vector<PermHash> hash_funcs;
		public:
		MinHash(int original_length, int signature_length) {
			for (int i=0; i<signature_length; i++) {
				hash_funcs.push_back(PermHash(original_length));
			}
		}

		vector<uint32_t> operator()(const vector<int> &s) {
			vector<uint32_t> res;
			res.reserve(hash_funcs.size());
			for (auto &h: hash_funcs) {
				uint32_t m = INT_MAX;
				for (int e: s) {
					int t = h(e);
					if (t < m) m = t;
				}
				res.push_back(m);	
			}
			return res;
		}
	};

	static inline float compute_lsh_threshold(int r, int signature_length) {
		return pow((1.0 * r / signature_length), 1.0 / r);
	}

	static inline float jaccard_similarity(const vector<int> &a, const vector<int> &b) {
		set<int> sb(b.begin(), b.end());
		int c = 0;
		for (int v: a) {
			if (sb.find(v) != sb.end()) c++;
		}
		int u = a.size() + b.size() - c;
		return 1.0 * c / u;
	}

	static inline float jaccard_similarity_v1(const vector<int> &a, const vector<int> &b) {
		int c = 0;
		int i = 0, j = 0;

		while (i < a.size() && j < b.size()) {
			if (a[i] == b[j]) {
				c++;
				i++;
				j++;
			} else if (a[i] > b[j]) {
				j++;
			} else {
				i++;
			}
		}

		int u = a.size() + b.size() - c;
		return 1.0 * c / u;
	}


template <class T>
	static vector<int> getkeys(map<int, T> &m) {
		vector<int> ks;
		for (auto &t: m) ks.push_back(t.first);
		return ks;
	}

	static size_t vector_hash(const vector<uint32_t> &vec) {
		size_t seed = vec.size();
		for (auto &v: vec) {
			seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}

	template <class T>
	static void print_vec(const vector<T> &vec) {
			for (auto &v: vec) cout << v << " ";
			cout << endl;
		}

	static inline int get_nelem(map<int, vector<int>> &clusters) {
		int cluster_nelem = 0;
		for (auto &c: clusters) {
			cluster_nelem += c.second.size();
		}
		return cluster_nelem;
	}

	static bool similiar_size(int a, int b) {
		if (a < b) {
			int tmp = a;
			a = b;
			b = tmp;
		}
		return a <= 1.5 * b;
	}


	friend class Clustering;
	friend class MYH;

	public:
	LSH() {

		//cout << "first_prime_greater_than 100: " << first_prime_greater_than(100) << endl; 

		//PermHash h1(100000);
		//PermHash h2(100000);
		//cout << h1(20) << " " << h1(30) << endl;
		//cout << h2(20) << " " << h2(30) << endl;

		//MinHash mh(100000, 64);
		//vector<int> t(1000);
		//generate(t.begin(), t.end(), [](){return rand() % 100000; });
		//vector<int> res = mh(t);
		//for (int x: res) {
		//	cout << x << " ";	
		//}
		//cout << endl;

	}

	static map<pair<int, int>, float> get_close_pairs_v2(vector<pair<int, vector<int>>> &rows, size_t start, size_t end, int original_length, int signature_length = 128, int band_size = 8) {
		assert (signature_length % band_size == 0 && original_length >= signature_length);
		MinHash mh(original_length, signature_length);
		float th = compute_lsh_threshold(band_size, signature_length);
		int b = signature_length / band_size;

	//	cout << "lsh threshold: " << th << ", bandsize: " << band_size << ", num_bands: " << b << endl;

		vector<map<size_t, vector<int>>> buckets(b);
		end = end > rows.size() ? rows.size() : end;
		size_t length = end - start;

		for (int p=0; p<length; p++) {
			auto &elem = rows[p+start];
			vector<uint32_t> sr = mh(elem.second);
#pragma omp parallel for
			for (int i=0; i<signature_length; i+=band_size) {
				size_t bucket_id = vector_hash(vector<uint32_t>(sr.begin()+i, sr.begin()+i+band_size));
				int band_id = i / band_size;
				if (buckets[band_id].find(bucket_id) == buckets[band_id].end()) buckets[band_id][bucket_id] = vector<int>();
				buckets[band_id][bucket_id].push_back(p);
			}
		}

	//	cout << "finish hashing...start to create pairs..." << endl;


		vector<map<pair<int, int>, float>> close_pairs(buckets.size());
#pragma omp parallel for
		for (int i=0; i<buckets.size(); i++) {
			auto &band = buckets[i];
			for (auto &buck: band) {
				vector<int> &bu = buck.second;
				if (bu.size() > 1) {
					for (int k1=0; k1<bu.size()-1; k1++) {
						for (int k2=k1+1; k2<bu.size(); k2++) {
							int a = bu[k1];
							int b = bu[k2];	
							assert (a != b);
							if (close_pairs[i].find(make_pair(a, b)) == close_pairs[i].end()) {
								float sim = jaccard_similarity_v1(rows[a+start].second, rows[b+start].second);
								if (sim >= th) {
									close_pairs[i][make_pair(a, b)] = sim; 
								}
							}
						}
					}
				}
			}
		}
		map<pair<int, int>, float> res;
		for (auto &cp: close_pairs) {
			for (auto &c: cp) {
				res.insert(c);
			}
			cp.clear();
		}
	//	cout << "num of close pairs: " << res.size() << endl;
		return res;
	}




template <class T>
	static map<pair<int, int>, float> get_close_pairs_v1(vector<pair<int, map<int, T>>> &rows, int original_length, int signature_length = 128, int band_size = 8) {
		assert (signature_length % band_size == 0 && original_length >= signature_length);
		MinHash mh(original_length, signature_length);
		float th = compute_lsh_threshold(band_size, signature_length);
		int b = signature_length / band_size;

	//	cout << "lsh threshold: " << th << ", bandsize: " << band_size << ", num_bands: " << b << endl;

		vector<map<size_t, vector<int>>> buckets(b);

		for (int p=0; p<rows.size(); p++) {
			auto &elem = rows[p];
			vector<uint32_t> sr = mh(getkeys(elem.second));
#pragma omp parallel for
			for (int i=0; i<signature_length; i+=band_size) {
				size_t bucket_id = vector_hash(vector<uint32_t>(sr.begin()+i, sr.begin()+i+band_size));
				int band_id = i / band_size;
				if (buckets[band_id].find(bucket_id) == buckets[band_id].end()) buckets[band_id][bucket_id] = vector<int>();
				buckets[band_id][bucket_id].push_back(p);
			}
		}

	//	cout << "finish hashing...start to create pairs..." << endl;


		vector<map<pair<int, int>, float>> close_pairs(buckets.size());
#pragma omp parallel for
		for (int i=0; i<buckets.size(); i++) {
			auto &band = buckets[i];
			for (auto &buck: band) {
				vector<int> &bu = buck.second;
				if (bu.size() > 1) {
					for (int k1=0; k1<bu.size()-1; k1++) {
						for (int k2=k1+1; k2<bu.size(); k2++) {
							int a = bu[k1];
							int b = bu[k2];	
							assert (a != b);
							if (close_pairs[i].find(make_pair(a, b)) == close_pairs[i].end()) {
								float sim = jaccard_similarity_v1(getkeys(rows[a].second), getkeys(rows[b].second));
								if (sim >= th) {
									close_pairs[i][make_pair(a, b)] = sim; 
								}
							}
						}
					}
				}
			}
		}
		map<pair<int, int>, float> res;
		for (auto &cp: close_pairs) {
			for (auto &c: cp) {
				res.insert(c);
			}
			cp.clear();
		}
	//	cout << "num of close pairs: " << res.size() << endl;
		return res;
	}

	static float compute_average_similarity(map<int, map<int, float>> &rows) {
		float th = 0.0;
		if (rows.size() > 1) {
			for (auto it=rows.begin(); next(it) != rows.end(); it++) {
				th += jaccard_similarity(getkeys(it->second), getkeys(next(it)->second));
			}
			th /= (rows.size() - 1);
		}
		return th;
	}

	template <class T>
	static map<pair<int, int>, float> get_close_pairs(map<int, map<int, T>> &rows, int original_length, int signature_length = 128, int band_size = 8) {
		assert (signature_length % band_size == 0 && original_length >= signature_length);
		MinHash mh(original_length, signature_length);
		float th = compute_lsh_threshold(band_size, signature_length);
		int b = signature_length / band_size;

	//	cout << "lsh threshold: " << th << ", bandsize: " << band_size << ", num_bands: " << b << endl;

		vector<map<size_t, vector<int>>> buckets(b);

		for (auto &elem: rows) {
			vector<uint32_t> sr = mh(getkeys(elem.second));
#pragma omp parallel for
			for (int i=0; i<signature_length; i+=band_size) {
				size_t bucket_id = vector_hash(vector<uint32_t>(sr.begin()+i, sr.begin()+i+band_size));
				int band_id = i / band_size;
				if (buckets[band_id].find(bucket_id) == buckets[band_id].end()) buckets[band_id][bucket_id] = vector<int>();
				buckets[band_id][bucket_id].push_back(elem.first);
			}
		}

//		cout << "finish hashing...start to create pairs..." << endl;

	
		vector<map<pair<int, int>, float>> close_pairs(buckets.size());
#pragma omp parallel for
		for (int i=0; i<buckets.size(); i++) {
			auto &band = buckets[i];
			for (auto &buck: band) {
				vector<int> &bu = buck.second;
				if (bu.size() > 1) {
					for (int k1=0; k1<bu.size()-1; k1++) {
						for (int k2=k1+1; k2<bu.size(); k2++) {
							int a = bu[k1];
							int b = bu[k2];	
							if (close_pairs[i].find(make_pair(a, b)) == close_pairs[i].end()) {
								float sim = jaccard_similarity_v1(getkeys(rows[a]), getkeys(rows[b]));
								if (sim >= th) {
									close_pairs[i][make_pair(a, b)] = sim; 
								}
							}
						}
					}
				}
			}
		}
		map<pair<int, int>, float> res;
		for (auto &cp: close_pairs) {
			for (auto &c: cp) {
				res.insert(c);
			}
			cp.clear();
		}
	//	cout << "num of close pairs: " << res.size() << endl;
		return res;
	}
	
};

#endif
