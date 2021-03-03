#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "clustering/clustering.h"
#include "matrix.h"

int main(int argc, char *argv[]) {
  SparseMatrixCSR_CPU<int> sm;
  sm.loadCOO(argv[1]);
  auto sp = sm.get_rowmap();
  auto close_pairs = LSH::get_close_pairs_v1(sp, sm.ncols, 128, 2);

  std::cout << "num of candidated pairs: " << close_pairs.size() << std::endl;
  auto reordered_rows = Clustering::hierachical_clustering_v0(sp, close_pairs, 512);

  std::ofstream fout(argv[2]);
  for (int r: reordered_rows) {
    fout << r << std::endl;
  }
  fout.close();
  return 0;

}