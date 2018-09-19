#pragma OPENCL EXTENSION cl_altera_channels : enable
channel uint manager2prod  __attribute__((depth(1024)));
channel uint prod2consu  __attribute__((depth(1024)));
channel uint consu2manager  __attribute__((depth(1024)));
typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define COL_SIZE  30 * 32
#define HASH_CU_NUM 128
#define UPDATETABLE_CU_NUM 128
#define WORK_AHEAD_SIZE 1
#define HASHBKNUM 1024*32 //rHashTableBucketNum/UPDATETABLE_CU_NUM
// worksize for every CU. total Size =  update_channels CUs * WORK_AHEAD_SIZE * hash-div CUs
// WASTable = HASH_CU_NUM *UPDATETABLE_CU_NUM* WORK_AHEAD_SIZE * sizeof(bucket_cell_t)
// Manager contains the information about the concrete memory offset and num of the bucket_cell need to processed.
// buffer contains the bucket_cell need to be processed of every CU and every bucket region
uint sim_hash(uint key, uint mod){
	return key % mod;
}
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_compute_units(1)))
__kernel void manager (){
}
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void hash_div_func(__global uint * restrict rTable, __global bucket_cell_t * restrict WASTable, const uint size,  const uint rHashTableBucketNum, __global uint * restrict midHashCount)
{
  uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
  uint iteration = tid;
  uint l_cnt[UPDATETABLE_CU_NUM] = {0};
  uint l_rHashTableBucketNum = rHashTableBucketNum;
  uint offset_factor = rHashTableBucketNum/UPDATETABLE_CU_NUM;
  uint D3 = COL_SIZE * UPDATETABLE_CU_NUM;
  while (tid < size){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[tid *2]);
		  uint key  = rtable_uint2.x;
      uint val  = rtable_uint2.y;
      uint hash_index = key & (l_rHashTableBucketNum-1);
      bucket_cell_t bucket_cell;
      bucket_cell.hash_index = hash_index;
      bucket_cell.key = key;
      bucket_cell.value = val;
      uint offset = (uint) hash_index / offset_factor;
      WASTable[ get_global_id(0) * D3 + offset * COL_SIZE  + l_cnt[offset] * WORK_AHEAD_SIZE]  = bucket_cell;
      l_cnt [offset] ++;
      midHashCount[ get_global_id(0)*UPDATETABLE_CU_NUM + offset] = l_cnt[offset];
      tid += numWorkItems;
  }
 // for (int i = 0; i < UPDATETABLE_CU_NUM; i ++){
 //   midHashCount[ get_global_id(0)*UPDATETABLE_CU_NUM + i] = l_cnt[i];
 // }//assign part hash count to global
}
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void update_table( __global uint * restrict rHashTable, __global bucket_cell_t * restrict WASTable, __global uint * restrict rHashCount, const uint hashBucketSize,__global uint * restrict l_cnt, const uint size)
{
  uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
  uint iteration = tid;
  uint l_hashBucketSize = hashBucketSize;
  uint l_process_cnt [UPDATETABLE_CU_NUM] = {0};

while(1){
  for (uint i = 0; i < HASH_CU_NUM; i ++){
    uint hash_idx_cnt = rHashCount[i * UPDATETABLE_CU_NUM + get_global_id(0)];
    //  printf("hash idx cnt is %d \n",hash_idx_cnt);
    for(int j = l_process_cnt[i]; j < hash_idx_cnt; j ++){
      bucket_cell_t bucket_cell = WASTable[ i * COL_SIZE * UPDATETABLE_CU_NUM + get_global_id(0) * COL_SIZE + j * WORK_AHEAD_SIZE];
      uint l_cnt_offset = bucket_cell.hash_index;
     // printf("l_cnt_offset is %d idx is %d  haha is %d HASH_CU_NUM is %d \n ",l_cnt_offset,bucket_cell.hash_index,get_global_id(0)*HASHBKNUM/UPDATETABLE_CU_NUM,i);
      rHashTable[bucket_cell.hash_index * l_hashBucketSize * 2 + l_cnt [l_cnt_offset] * 2 + 0] = bucket_cell.key;
      rHashTable[bucket_cell.hash_index * l_hashBucketSize * 2 + l_cnt [l_cnt_offset] * 2 + 1] = bucket_cell.value;

      l_cnt [l_cnt_offset] ++;
    }
      l_process_cnt[i] = hash_idx_cnt;
  }
  uint hash_idx_sum = 0;
  uint l_cnt_sum = 0;
  for(int i = 0; i < UPDATETABLE_CU_NUM; i ++){
    // hash_idx_sum += rHashCount[i * UPDATETABLE_CU_NUM + get_global_id(0)];
     l_cnt_sum += l_process_cnt[i];
  }
 // printf("hash_idx_sum %d,l_cnt_sum %d", hash_idx_sum,l_cnt_sum);
  if((size >> 7) == l_cnt_sum)
 // if (hash_idx_sum == l_cnt_sum) 
  break;
}
}
