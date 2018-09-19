/* simple hash join on BRAM */
//----channel define----//
#define ENDFLAG 0xffff
channel uint2 relR[16] __attribute__((depth(256)));
channel uint2 relS[16] __attribute__((depth(256)));
channel uint relRendFlagCh __attribute__((depth(128)));
channel uint relSendFlagCh __attribute__((depth(128)));


#define HASH(K, MASK, SKIP) (((K) & MASK) >> SKIP)
#define RELR_L_NUM 1024*256*2
#define HASHTABLE_L_SIZE 1024*256*2
#define HASHTABLE_BUCKET_SIZE 4 // 2 byte per 
#define HASHTABLE_BUCKET_NUM HASHTABLE_L_SIZE/HASHTABLE_BUCKET_SIZE
//#define SW
__attribute__((task))
__kernel void relRead (
                        __global uint2 * restrict rTable, 
                        __global uint2 * restrict rTableReadRange, 
                        __global uint2 * restrict sTable, 
                        __global uint2 * restrict sTableReadRange
                      )
{
    uint rTableOffset = rTableReadRange[0].x;
    uint rTableReadNum = rTableReadRange[0].y;
    uint sTableOffset = sTableReadRange[0].x;
    uint sTableReadNum = sTableReadRange[0].y;
    for(int i = rTableOffset; i < rTableOffset + rTableReadNum; i += 16){
    #pragma unroll 16
      for(int k = 0; k < 16; k++){
          uint2 rtable_uint2 = rTable[i + k];
          write_channel_altera(relR[k], rtable_uint2);
      }
    }

    for(int i = sTableOffset; i < (sTableOffset + sTableReadNum); i += 16){
    #pragma unroll 16
      for(int k = 0; k < 16; k++){
          uint2 stable_uint2 = sTable[i + k];
          write_channel_altera(relS[k], stable_uint2);
      }
    }

}

__attribute__((task))
__kernel void hashjoin (
                        __global uint * restrict matchedTable, 
                        __global uint2 * restrict rTableReadRange,  
                        __global uint2 * restrict sTableReadRange
                      )
{
   // build phrase 
   //__local uint relR_l [RELR_L_SIZE];
    uint2 hashtable_l [HASHTABLE_L_SIZE >> 4][16];
    uint hashtable_bucket_cnt [HASHTABLE_BUCKET_NUM >> 4][16];
/*
    for(int i = 0; i < HASHTABLE_BUCKET_NUM; i ++){
      hashtable_bucket_cnt[i] = 0;
    }
    uint2 init_value;
    init_value.x = 0;
    init_value.y = 0;
    for(int i = 0; i < HASHTABLE_L_SIZE; i ++){
      hashtable_l[i] = init_value;
    }
*/
    bool valid_r = false;
    uint rTableReadNum = rTableReadRange[0].y;

    for (int k = 0; k < rTableReadNum; k += 16){
      #pragma unroll 16
        for(int i = 0; i < 16; i ++){ 
              uint2 rtable_uint2 = read_channel_altera(relR[i]);
              {
                uint key  = rtable_uint2.x; 
                uint val  = rtable_uint2.y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= rtable_uint2;
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }
        }
    }
    //  probe phrase
    uint matchedCnt = 0; 
    uint iter = (sTableReadRange[0].y);

    
    bool valid_s = false;
    for (int k = 0; k < iter; k +=16){
      #pragma unroll 16
        for(int i = 0; i < 16; i ++){
            uint2 stable_uint2 = read_channel_altera(relS[i]);
            {
              uint key = stable_uint2.x;
              uint hash_idx = HASH (key, (HASHTABLE_BUCKET_NUM - 1),4);
              //printf("probe key %d \n", key);
              for(int j = 0; j < HASHTABLE_BUCKET_SIZE; j ++){
                  //uint hashTable_val = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + j].y;
                  uint hashTable_key = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + j][i].x;
                  if(key == hashTable_key){
                    //matchedTable[matchedCnt] = key;
                    //matchedTable[matchedCnt + 1] = rtable_uint2.y;
                    //matchedTable[matchedCnt + 2] = hashTable_val;
                    matchedCnt ++;
                  }
              }
            }
        }
    }
        
    matchedTable[0] = matchedCnt;


  #ifdef SW
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", sTable[i].x);
      printf ("\n");
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", rTable[i].x);
      printf ("\n");
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", hashtable_l[i].x);
      printf ("\n");

  #endif
}
/*
#pragma unroll 8
    for ( int i = 0; i < RELR_L_SIZE; i ++){
        relR_l [i] = rTable [i];
    }
*/
#if 0

#define LOCAL_MEMORY_NUM_BITS 15
#define LOCAL_MEMORY_NUM      (1<<LOCAL_MEMORY_NUM_BITS)  //32<<10)
#define BucketNum 1024 * 1024 * 4
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))

__kernel void buildHashTable(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize,__global uint * restrict tHist, __global uint * restrict rHashCount)
{
	uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
	uint lid          = get_local_id(0);
	uint lsize        = get_local_size(0);
	uint key, val, hash_lock, hash_index, count;
  uint iteration = 0;
  iteration = tid;
  while (tid < size)
	{
		//key = rTableOnDevice[offset * 2 + tid * 2 + 0];
		//val = rTableOnDevice[offset * 2 + tid * 2 + 1];
		uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + tid * 2 ]);
		key  = rtable_uint2.x; val  = rtable_uint2.y;
		      //2, update the corresponding bucket with index: hash_index.
		      hash_index = djb2_hash(key) & (rHashTableBucketNum-1); // real bucket to populate
	        //tHist[hash_index] ++ ;
	        tHist[get_global_id(0) * rHashTableBucketNum + hash_index] ++ ;
          //count = l_rHashCount[hash_index]++; //count = rHashCount[hash_index]; rHashCount[hash_index] = count + 1;
		      //rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 0] = key;	  //hash：
          //rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 1] = val;      //hash
		tid += numWorkItems;
	}

    barrier(CLK_GLOBAL_MEM_FENCE);
// to caculate the elements nums in each buckets 
    tid = get_global_id(0);
  while (tid < rHashTableBucketNum){

    count = 0;
    for(int i = 0; i < numWorkItems; i ++){
       count += tHist[i * rHashTableBucketNum + tid];
    }
    rHashCount[tid] = count;

     tid += numWorkItems;
   }
// build the prefix-sum table 
  barrier(CLK_GLOBAL_MEM_FENCE);
  tid = get_global_id(0);
  for (int now_item = 1; now_item < (numWorkItems-1); now_item ++){
    iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[now_item * rHashTableBucketNum + iteration]  += tHist[(now_item-1) * rHashTableBucketNum + iteration];
      iteration += numWorkItems;
    }
  }
  // the prefix sum for the thread ID 0; 
    tid = get_global_id(0);
    iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[127 * rHashTableBucketNum + iteration]= 0;
      iteration += numWorkItems;
    }

  barrier(CLK_GLOBAL_MEM_FENCE);
// write result to hash table 
  tid = get_global_id(0);
  iteration = tid;
  while (iteration < size){
        uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + iteration * 2 ]);
        key  = rtable_uint2.x; val  = rtable_uint2.y;
        hash_index = djb2_hash(key) & (rHashTableBucketNum-1);
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

__attribute__((num_compute_units(4)))
__kernel void probeHashTable(__global uint * restrict rHashTable, __global uint * restrict  sTableOnDevice, __global  uint * restrict matchedTable, 
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
	  uint numWorkItems = get_global_size(0);
    uint tid          = get_global_id(0);
    uint lid          = get_local_id(0);
	  int block_id      = get_group_id(0);
 // int block_size    = get_num_groups(0);
	  uint key, val, hash, count, matchedNum;
	  __local int local_counter[128];
//	if (lid == 0)
//	 local_counter = 0;
	  local_counter[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
	while(tid < size)
	{
		key = sTableOnDevice[offset * 2 + tid * 2 + 0];
		val = sTableOnDevice[offset * 2 + tid * 2 + 1];

			//since hash value calculation consumes only tens ms, so GPU will finish it first
//		hash = djb2_hash(key,rHashTableBucketNum);

			hash = djb2_hash(key) &(rHashTableBucketNum-1);//sim_hash(key,rHashTableBucketNum);

			//find out matched tuples in hash table for R table
			count = 0;
			//int hashBucketRealSize = rHashCount[hash];
			while(count < hashBucketSize) //before optimization:  hashBucketRealSize
			{
//if(atomic_cmpxchg(&rHashTable[hash * hashBucketSize * 2 + count * 2 + 0],0,0) != 0  && atomic_cmpxchg(&rHashTable[hash * hashBucketSize * 2 + count * 2 + 0],0,0) == key)
				if(rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] != 0 && rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] == key)
				{
					local_counter[lid]++;//matchedNum = atomic_inc(&local_counter); //0 matchedTable[block_id]
//					matchedTable[3 + matchedNum * 3 + 0] = key;
//					matchedTable[3 + matchedNum * 3 + 1] = val;
//					matchedTable[3 + matchedNum * 3 + 2] = rHashTable[hash * hashBucketSize * 2 + count * 2 + 1];
					count++;
				}
				else if (rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] == 0)
					break;
				else
					count++;
			}

		 tid += numWorkItems;
	}
     barrier(CLK_LOCAL_MEM_FENCE);
	 if ( (lid&3) == 0)
	   local_counter[lid] = local_counter[lid] + local_counter[lid+1] + local_counter[lid+2] + local_counter[lid+3];
     barrier(CLK_LOCAL_MEM_FENCE);
	 if ((lid&15) == 0)
	   local_counter[lid] = local_counter[lid] + local_counter[lid+4] + local_counter[lid+8] + local_counter[lid+12];
     barrier(CLK_LOCAL_MEM_FENCE);

    int counter_s = 0;

	 if (lid == 0)
	  {
	    counter_s = local_counter[0]  + local_counter[16] + local_counter[32] + local_counter[48] +
	               local_counter[64] + local_counter[80] + local_counter[96] + local_counter[112];
	    matchedTable[block_id] = counter_s;
	  }

//sum the local_counter[local_size];
//     if (lid == 0)
//	   matchedTable[block_id] = local_counter;
}
#endif
