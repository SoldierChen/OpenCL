#pragma OPENCL EXTENSION cl_altera_channels : enable
typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define CHANNEL_NUM 8
#define HASH_CU_NUM 128
#define UPDATETABLE_CU_NUM 128
#define WORK_AHEAD_SIZE 64 // worksize for every CU. total Size =  update_channels CUs * WORK_AHEAD_SIZE * hash-div CUs
// WASTable = HASH_CU_NUM *UPDATETABLE_CU_NUM* WORK_AHEAD_SIZE * sizeof(bucket_cell_t)
// Manager contains the information about the concrete memory offset and num of the bucket_cell need to processed.
// buffer contains the bucket_cell need to be processed of every CU and every bucket region
channel bucket_cell_t update_channel[CHANNEL_NUM] __attribute__ ((depth(1024)));
uint sim_hash(uint key, uint mod){
	return key % mod;
}
__kernel void __attribute__((task))
  hash_div_func(__global uint * restrict rTable, __global bucket_cell_t * restrict WASTable, __global uint * restrict WASReadyTable,const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
  int cnt = 0;
  uint l_cnt[UPDATETABLE_CU_NUM] = {0};
  for (int i = 0; i < size; i += HASH_CU_NUM){
#pragma unroll HASH_CU_NUM
    for (int j = 0; j < HASH_CU_NUM; j ++){
      if(WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset ]  = 0xffff)
        l_cnt [offset] = 0;
      if ( l_cnt [offset] == WORK_AHEAD_SIZE) {
        WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset ]  = 1;
        WASReadyTable[j * 2 * UPDATETABLE_CU_NUM  + offset + 1]  = WORK_AHEAD_SIZE;
      }else {

  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+ (i+j) *2]);
		  uint key  = rtable_uint2.x;
      uint val  = rtable_uint2.y;
      uint hash_index = key & (rHashTableBucketNum-1);

      bucket_cell_t bucket_cell;
      bucket_cell.hash_index = hash_index;
      bucket_cell.key = key;
      bucket_cell.value = val;
      uint offset = (uint) hash_idex / (rHashTableBucketNum/UPDATETABLE_CU_NUM);

      WASTable[ j * UPDATETABLE_CU_NUM  + offset * WORK_AHEAD_SIZE + l_cnt[offset]  = bucket_cell; 
      l_cnt [offset] ++;
     //      cnt ++;
     //   printf("current tuple is %x \n", cnt);
      mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
    }
  }
}

__kernel void __attribute__((task))
update_table( __global uint * restrict rHashTable, __global bucket_cell_t * restrict WASTable, __global uint * restrict WASReadyTable,const uint hashBucketSize)
{
#pragma unroll UPDATETABLE_CU_NUM
  for (int i  = 0; i < UPDATETABLE_CU_NUM; i ++){
    for (int j = 0;)
  }
while(true){
    bool ret;
    //printf("liangliang\n");
    bucket_cell_t bucket_cell[8]; // may not support dynamic read channel, use the switch case to do it
   {
     }
     uint j[CHANNEL_NUM];
    for(j[0] = 0;j[0] < hashBucketSize; j[0] ++){
      if (rHashTable[bucket_cell[0].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] == 0)
        break;
    rHashTable[bucket_cell[0].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] = bucket_cell[0].key;
    rHashTable[bucket_cell[0].hash_index * hashBucketSize * 2 + j[0] * 2 + 1] = bucket_cell[0].value;
  }
  }
}
