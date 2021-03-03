#ifndef _CLST_H
#define _CLST_H

#include "lsh.h"
#include "myh.h"
using namespace std;

class Clustering {
	public:
		static vector<int> hierachical_clustering_v2(vector<pair<int, vector<int>>> &sp, size_t start, size_t end, map<pair<int, int>, float> &close_pairs, int cluster_size) {
		using item_t = pair<float, pair<int, int>>;
		auto cmp = [](const item_t &a, const item_t &b){ return a.first < b.first; };
		priority_queue<item_t, vector<item_t>, decltype(cmp)> sims(cmp);
		for (auto &p: close_pairs) {
			sims.push(make_pair(p.second, p.first));
		}
		end = end > sp.size() ? sp.size() : end;
		size_t length = end - start;
		vector<int> clusters(length);
		vector<int> sz(length);
		vector<int> valid(length);
		int nclusters = length;
		map<int, int> row_to_cluster;
		for (int i=0; i<nclusters; i++) {
			clusters[i] = i;
			sz[i] = 1;
			valid[i] = 1;
		}

	//	cout << "start clustering" << endl;

		while (!sims.empty() && nclusters != 0) {
			//cout << nclusters << " " << sims.size() << endl;
			item_t s = sims.top();
			sims.pop();
			int i = s.second.first;
			int j = s.second.second;
			if (clusters[i] == i && clusters[j] == j) {
				if (!valid[i] || !valid[j]) continue;
				nclusters--;
				if (sz[i] < sz[j]) {
					clusters[i] = j;
					sz[j] += sz[i];
					if (sz[j] >= cluster_size) {
						valid[j] = 0;
						nclusters--;
					}
				} else {
					clusters[j] = i;
					sz[i] += sz[j];
					if (sz[i] >= cluster_size) {
						valid[i] = 0;
						nclusters--;
					}
				}
			} else {
				while (i != clusters[i]) {
					clusters[i] = clusters[clusters[i]];
					i = clusters[i];	
				}
				while (j != clusters[j]) {
					clusters[j] = clusters[clusters[j]];
					j = clusters[j];	
				}
				if (!valid[i] || !valid[j]) continue;
				if (i != j) {
					auto p = make_pair(i, j);
					if (close_pairs.find(p) == close_pairs.end()) {
						float s = LSH::jaccard_similarity_v1(sp[i+start].second, sp[j+start].second);
						sims.push(make_pair(s, p));
						close_pairs[p] = s;
					}
				}
			}
		}

		vector<int> reordered;
		map<int, vector<int>> reordered_dict;

		for (int i=0; i<clusters.size(); i++) {
			int j = i;
			while (j != clusters[j]) j = clusters[j];
			if (reordered_dict.find(j) == reordered_dict.end()) reordered_dict[j] = vector<int>();
			reordered_dict[j].push_back(sp[i+start].first);
		}

		for (auto &t: reordered_dict) {
			reordered.insert(reordered.end(), t.second.begin(), t.second.end());
		}

		assert(length == reordered.size());

		return reordered; 
	}


	static vector<int> hierachical_clustering_v1(vector<pair<int, map<int, float>>> &sp, map<pair<int, int>, float> &close_pairs, int cluster_size) {
		using item_t = pair<float, pair<int, int>>;
		auto cmp = [](const item_t &a, const item_t &b){ return a.first < b.first; };
		priority_queue<item_t, vector<item_t>, decltype(cmp)> sims(cmp);
		for (auto &p: close_pairs) {
			sims.push(make_pair(p.second, p.first));
		}
		vector<int> clusters(sp.size());
		vector<int> sz(sp.size());
		vector<int> valid(sp.size());
		int nclusters = sp.size();
		map<int, int> row_to_cluster;
		for (int i=0; i<sp.size(); i++) {
			clusters[i] = i;
			sz[i] = 1;
			valid[i] = 1;
		}

	//	cout << "start clustering" << endl;

		while (!sims.empty() && nclusters != 0) {
			//cout << nclusters << " " << sims.size() << endl;
			item_t s = sims.top();
			sims.pop();
			int i = s.second.first;
			int j = s.second.second;
			if (clusters[i] == i && clusters[j] == j) {
				if (!valid[i] || !valid[j]) continue;
				nclusters--;
				if (sz[i] < sz[j]) {
					clusters[i] = j;
					sz[j] += sz[i];
					if (sz[j] >= cluster_size) {
						valid[j] = 0;
						nclusters--;
					}
				} else {
					clusters[j] = i;
					sz[i] += sz[j];
					if (sz[i] >= cluster_size) {
						valid[i] = 0;
						nclusters--;
					}
				}
			} else {
				while (i != clusters[i]) {
					clusters[i] = clusters[clusters[i]];
					i = clusters[i];	
				}
				while (j != clusters[j]) {
					clusters[j] = clusters[clusters[j]];
					j = clusters[j];	
				}
				if (!valid[i] || !valid[j]) continue;
				if (i != j) {
					auto p = make_pair(i, j);
					if (close_pairs.find(p) == close_pairs.end()) {
						float s = LSH::jaccard_similarity_v1(LSH::getkeys(sp[i].second), LSH::getkeys(sp[j].second));
						sims.push(make_pair(s, p));
						close_pairs[p] = s;
					}
				}
			}
		}

		vector<int> reordered;
		map<int, vector<int>> reordered_dict;

		for (int i=0; i<clusters.size(); i++) {
			int j = i;
			while (j != clusters[j]) j = clusters[j];
			if (reordered_dict.find(j) == reordered_dict.end()) reordered_dict[j] = vector<int>();
			reordered_dict[j].push_back(i);
		}

		for (auto &t: reordered_dict) {
			reordered.insert(reordered.end(), t.second.begin(), t.second.end());
		}

		assert(sp.size() == reordered.size());

		return reordered; 
	}

template <class T>
	static vector<int> hierachical_clustering_v0(vector<pair<int, map<int, T>>> &sp, map<pair<int, int>, float> &close_pairs, int cluster_size) {
		using item_t = pair<float, pair<int, int>>;
		auto cmp = [](const item_t &a, const item_t &b){ return a.first < b.first; };
		priority_queue<item_t, vector<item_t>, decltype(cmp)> sims(cmp);
		for (auto &p: close_pairs) {
			sims.push(make_pair(p.second, p.first));
		}
		vector<int> clusters(sp.size());
		vector<int> sz(sp.size());
		vector<int> valid(sp.size());
		int nclusters = sp.size();
		map<int, int> row_to_cluster;
		for (int i=0; i<sp.size(); i++) {
			clusters[i] = i;
			sz[i] = 1;
			valid[i] = 1;
		}

	//	cout << "start clustering" << endl;

		while (!sims.empty() && nclusters != 0) {
			//cout << nclusters << " " << sims.size() << endl;
			item_t s = sims.top();
			sims.pop();
			int i = s.second.first;
			int j = s.second.second;
			if (clusters[i] == i && clusters[j] == j) {
				if (!valid[i] || !valid[j]) continue;
				nclusters--;
				if (sz[i] < sz[j]) {
					clusters[i] = j;
					sz[j] += sz[i];
					if (sz[j] >= cluster_size) {
						valid[j] = 0;
						nclusters--;
					}
				} else {
					clusters[j] = i;
					sz[i] += sz[j];
					if (sz[i] >= cluster_size) {
						valid[i] = 0;
						nclusters--;
					}
				}
			} else {
				while (i != clusters[i]) {
					clusters[i] = clusters[clusters[i]];
					i = clusters[i];	
				}
				while (j != clusters[j]) {
					clusters[j] = clusters[clusters[j]];
					j = clusters[j];	
				}
				if (!valid[i] || !valid[j]) continue;
				if (i != j) {
					auto p = make_pair(i, j);
					if (close_pairs.find(p) == close_pairs.end()) {
						float s = LSH::jaccard_similarity_v1(LSH::getkeys(sp[i].second), LSH::getkeys(sp[j].second));
						sims.push(make_pair(s, p));
						close_pairs[p] = s;
					}
				}
			}
		}

		vector<int> reordered;
		map<int, vector<int>> reordered_dict;

		for (int i=0; i<clusters.size(); i++) {
			int j = i;
			while (j != clusters[j]) j = clusters[j];
			if (reordered_dict.find(j) == reordered_dict.end()) reordered_dict[j] = vector<int>();
			reordered_dict[j].push_back(sp[i].first);
		}

		for (auto &t: reordered_dict) {
			reordered.insert(reordered.end(), t.second.begin(), t.second.end());
		}

		assert(sp.size() == reordered.size());

		return reordered; 
	}


};

#endif
