uint djb2_hash(uint  key){
  unsigned long hash = 5381;
  uint tempKey[4];
  tempKey[0] = (0x000000ff & key) >> 0;
  tempKey[1] = (0x0000ff00 & key) >> 8;
  tempKey[2] = (0x00ff0000 & key) >> 16;
  tempKey[3] = (0xff000000 & key) >> 24;
    for(uint i = 0; i < 4; i++)
        hash = ((hash << 5) + hash) + tempKey[i];
    return hash;
}
uint sim_hash(uint key, uint mod){
	return key % mod;
}
#define LOCAL_MEMORY_NUM_BITS 15
#define LOCAL_MEMORY_NUM      (1<<LOCAL_MEMORY_NUM_BITS)  //32<<10)
#define BUCKETNUM 1024 * 4
#define SW
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))

__kernel void buildHashTable(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict tHist)
{
	uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
	uint lid          = get_local_id(0);
	uint lsize        = get_local_size(0);
  #ifdef SW
  //int * tHist = malloc(sizeof(int) * 128 * BUCKETNUM);
  #else
  //int tHist[128 * BUCKETNUM];
  #endif
  //for(int i = 0; i < 128 * BUCKETNUM; i ++) tHist[i] = 0;
  // printf("ck 1 \n");
  {
    int iteration = tid;
    while (iteration < size)
  	{ 
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + iteration * 2 ]);
  		int key = rtable_uint2.x; int val = rtable_uint2.y;
  		int hash_index = sim_hash(key, rHashTableBucketNum);//key & (rHashTableBucketNum - 1);//djb2_hash(key) & (rHashTableBucketNum-1); // real bucket to populate
  	  //printf("hash_index %d \n",hash_index);
      tHist[tid * rHashTableBucketNum + hash_index] ++ ;
  		iteration += numWorkItems;
  	}
  }

   barrier(CLK_GLOBAL_MEM_FENCE);

// build the prefix-sum table 
  for(int now_item = 1; now_item < (numWorkItems-1); now_item ++){
    int iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[now_item * rHashTableBucketNum + iteration]  += tHist[(now_item-1) * rHashTableBucketNum + iteration];
        iteration += numWorkItems;
    }
  }


   barrier(CLK_GLOBAL_MEM_FENCE);

  // the prefix sum for the thread ID 0; 
  {
    int iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[127 * rHashTableBucketNum + iteration]= 0;
        iteration += numWorkItems;
    }
  }
   barrier(CLK_GLOBAL_MEM_FENCE);

  {
    int iteration = tid;
    while (iteration < size){
          uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + iteration * 2 ]);
          int key = rtable_uint2.x; int val = rtable_uint2.y;
          int hash_index = sim_hash(key, rHashTableBucketNum);      
          if(tid != 0){
          *(__global uint2 *)(&rHashTable[hash_index * hashBucketSize * 2 + tHist[(tid-1) * rHashTableBucketNum + hash_index] * 2] ) = rtable_uint2;
          tHist[(tid-1) * rHashTableBucketNum + hash_index] ++;
          }
          else {
          *(__global uint2 *)(&rHashTable[hash_index * hashBucketSize * 2 + tHist[127 * rHashTableBucketNum + hash_index] * 2] ) = rtable_uint2;
            tHist[127 * rHashTableBucketNum + hash_index] ++;
          }
          iteration += numWorkItems;
    }
  }
}

