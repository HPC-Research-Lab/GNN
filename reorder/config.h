// Created by Peng Jiang on 2019-03-03.
//

#ifndef GCONV_CONFIG_H
#define GCONV_CONFIG_H

#define MAX_TPB 512
#define WARP_SIZE 32
#define SROW_PER_TILE 2      // needs to be smaller than 32 
#define SROW_PER_TILE1 2      // needs to be smaller than 32 
#define SROW_PER_TILE2 4      // needs to be smaller than 32 
#define DROW_PER_TILE 128 
#define DCOL_PER_TILE 128
#define DENSE_COL_THRD 24
#define DENSE_ROW_THRD 24
#define SPARSE3_THRESHOLD 64




#endif //GCONV_CONFIG_
