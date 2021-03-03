#ifndef _MYH_H
#define _MYH_H
#include "lsh.h"
using namespace std;

class MYH {
	public:
	static map<pair<int, int>, float> get_close_pairs(map<int, map<int, float>> &rows, int original_length, int signature_length = 128, int band_size = 8) {
		assert (signature_length % band_size == 0 && original_length >= signature_length);
		map<int, vector<int>> cols;
		for (auto &rr: rows) {
			int r = rr.first;
			for (auto &cc: rr.second) {
				int c = cc.first;
				if (cols.find(c) == cols.end()) cols[c] = vector<int>();
				cols[c].push_back(r);			
			}
		}
		
		vector<pair<int, vector<int>>> cols_vec;	
		for (auto &c: cols) cols_vec.push_back(c);
		sort(cols_vec.begin(), cols_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b){ return a.second.size() > b.second.size(); });

		int b = signature_length / band_size;
		map<int, vector<vector<uint32_t>>> important_cols_sig; 
		for (int i=0; i<(cols_vec.size()<signature_length?cols_vec.size():signature_length); i+=band_size) {
			for (int j=i; j<i+band_size; j++) {
				int c = cols_vec[j].first;
				for (int r: cols_vec[i].second) {

					if (important_cols_sig.find(r) == important_cols_sig.end()) important_cols_sig[r] = vector<vector<uint32_t>>(b);
					important_cols_sig[r][i/band_size].push_back(c);
				}
			}
		}

		vector<map<size_t, vector<int>>> buckets(b);
		for (auto &rr: important_cols_sig) {
			int r = rr.first;
			for (int i=0; i<b; i++) {
				auto &sig = rr.second[i];
				size_t bucket_id = LSH::vector_hash(sig);
				if (buckets[i].find(bucket_id) == buckets[i].end()) buckets[i][bucket_id] = vector<int>();
				buckets[i][bucket_id].push_back(r);
			}
		}

		cout << "finish hashing...start to create pairs..." << endl;

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
								float sim = LSH::jaccard_similarity(LSH::getkeys(rows[a]), LSH::getkeys(rows[b]));
								close_pairs[i][make_pair(a, b)] = sim; 
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
		cout << "num of close pairs: " << res.size() << endl;
		return res;


	}
};


#endif
