typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define COL_SIZE  30
#define HASH_CU_NUM 128
#define UPDATETABLE_CU_NUM 128
#define WORK_AHEAD_SIZE 3
#define HASHBKNUM 1024*4//rHashTableBucketNum/UPDATETABLE_CU_NUM
// worksize for every CU. total Size =  update_channels CUs * WORK_AHEAD_SIZE * hash-div CUs
// WASTable = HASH_CU_NUM *UPDATETABLE_CU_NUM* WORK_AHEAD_SIZE * sizeof(bucket_cell_t)
// Manager contains the information about the concrete memory offset and num of the bucket_cell need to processed.
// buffer contains the bucket_cell need to be processed of every CU and every bucket region
uint sim_hash(uint key, uint mod){
	return key % mod;
}
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void hash_div_func(__global uint * restrict rTable, __global bucket_cell_t * restrict WASTable, __global uint * restrict WASReadyTable, const uint size,  const uint rHashTableBucketNum, __global uint * restrict midHashCount)
{
  uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
  uint iteration = tid;
  uint l_cnt[UPDATETABLE_CU_NUM] = {0};
 // printf("\ntid is %d \n",tid);
  for (int i = 0; i < UPDATETABLE_CU_NUM; i ++) l_cnt[i] = 0;
  while (tid < size){
    /*if(WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset ]  = 0xffff)
        l_cnt [offset] = 0;
      if ( l_cnt [offset] == WORK_AHEAD_SIZE) {
        WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset ]  = 1;
        WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset + 1]  = WORK_AHEAD_SIZE;
      }else {}*/
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[tid *2]);
		  uint key  = rtable_uint2.x;
      uint val  = rtable_uint2.y;
      uint hash_index = key & (rHashTableBucketNum-1);
      bucket_cell_t bucket_cell;
      bucket_cell.hash_index = hash_index;
      bucket_cell.key = key;
      bucket_cell.value = val;
      uint offset = (uint) hash_index / (rHashTableBucketNum/UPDATETABLE_CU_NUM);
     // printf(" off set is %d l_cnt[offset] is %d\n",offset,l_cnt[offset]);
      //if(bucket_cell.hash_index == 190) printf("tid is %d offset is %d\n",get_global_id(0),offset);
  //    printf("wastable[%lu]\n",get_global_id(0) * COL_SIZE * UPDATETABLE_CU_NUM + offset*COL_SIZE + l_cnt[offset]* WORK_AHEAD_SIZE);
      WASTable [get_global_id(0) * COL_SIZE * UPDATETABLE_CU_NUM + offset * COL_SIZE  + l_cnt[offset] * WORK_AHEAD_SIZE]  = bucket_cell;
     // printf("!");
      l_cnt [offset] ++;
      tid += numWorkItems;
  }
  for (int i = 0; i < UPDATETABLE_CU_NUM; i ++){
     midHashCount[ get_global_id(0)*UPDATETABLE_CU_NUM + i] = l_cnt[i];
  }//assign part hash count to global
}
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void update_table( __global uint * restrict rHashTable, __global bucket_cell_t * restrict WASTable, __global uint * restrict WASReadyTable,__global uint * restrict rHashCount, const uint rHashTableBucketNum, const uint hashBucketSize)
{
  uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
  uint iteration = tid;
  uint l_cnt[HASHBKNUM/UPDATETABLE_CU_NUM] = {0};
  for (int i = 0; i < HASHBKNUM/UPDATETABLE_CU_NUM; i ++) l_cnt[i] = 0;
 // printf("tid is %d \n",tid);
for (uint i = 0; i < HASH_CU_NUM; i ++){
    uint hash_idx_cnt = rHashCount[i * UPDATETABLE_CU_NUM + get_global_id(0)];
  //  printf("hash idx cnt is %d \n",hash_idx_cnt);
    for(int j = 0; j < hash_idx_cnt; j ++){
      bucket_cell_t bucket_cell = WASTable[ i * COL_SIZE * UPDATETABLE_CU_NUM + get_global_id(0) * COL_SIZE + j * WORK_AHEAD_SIZE];
      uint l_cnt_offset = bucket_cell.hash_index - get_global_id(0)*HASHBKNUM/UPDATETABLE_CU_NUM;
  //   printf("l_cnt_offset is %d idx is %d  hash_index is  %d HASH_CU_NUM is %d \n ",l_cnt[l_cnt_offset],bucket_cell.hash_index,get_global_id(0)*HASHBKNUM/UPDATETABLE_CU_NUM,i);
//      printf("rHashTable[%lu]\n",bucket_cell.hash_index * hashBucketSize * 2 + l_cnt [l_cnt_offset] * 2 + 0);
      rHashTable[bucket_cell.hash_index * hashBucketSize * 2 + l_cnt [l_cnt_offset] * 2 + 0] = bucket_cell.key;
      rHashTable[bucket_cell.hash_index * hashBucketSize * 2 + l_cnt [l_cnt_offset] * 2 + 1] = bucket_cell.value;

      l_cnt [l_cnt_offset] ++;
    }
  }
}

