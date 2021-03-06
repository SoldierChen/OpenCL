uint sim_hash(uint key, uint mod)
{
	return key % mod;
}
// this is the test for remote edit the file ...//
#define LOCAL_MEMORY_NUM_BITS 15
#define LOCAL_MEMORY_NUM      (1<<LOCAL_MEMORY_NUM_BITS)  //32<<10)
#define BucketNum 1024 * 1024 * 4
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))

__kernel void buildHist_p1(__global uint2 * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize,__global uint * restrict tHist, __global uint * restrict rHashCount)
{
	uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
	uint key, val, hash_lock, hash_index, count;
  uint iteration = 0;
  iteration = tid;
  while (tid < size)
	{
	  uint2 uint2_rtable = rTableOnDevice[tid];
		hash_index = uint2_rtable.x & (rHashTableBucketNum-1); // real bucket to populate
	    tHist[get_global_id(0) * rHashTableBucketNum + hash_index] ++ ;
		tid += numWorkItems;
	}
}
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void buildHist_p2(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize,__global uint * restrict tHist, __global uint * restrict rHashCount)
{
	uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
	uint key, val, hash_lock, hash_index, count;
  uint iteration = 0;
  tid = get_global_id(0);
  uint sum = 0;
  uint tHist_buf[128];
  while (tid < rHashTableBucketNum){
    sum = 0;
    for(int i =0; i < numWorkItems; i ++){
      sum += tHist[i * rHashTableBucketNum + tid];
       tHist[i * rHashTableBucketNum + tid]  = sum;
    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
/*    for(int i =0; i < numWorkItems; i ++){
      tHist[i * rHashTableBucketNum + tid] = tHist_buf[i];
    }
*/    tid += numWorkItems;
  }
}

__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void buildHashTable(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize,__global uint * restrict tHist, __global uint * restrict rHashCount){
  uint  tid = get_global_id(0);
	uint numWorkItems = get_global_size(0);
  uint  iteration = tid;
  while (iteration < size){
        uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + iteration * 2 ]);
        uint key  = rtable_uint2.x; uint val  = rtable_uint2.y;
        uint hash_index = key & (rHashTableBucketNum-1);
          tHist[(tid) * rHashTableBucketNum + hash_index] --;
          *(__global uint2 *)(&rHashTable[hash_index * hashBucketSize * 2 + tHist[(tid) * rHashTableBucketNum + hash_index] * 2] ) = rtable_uint2;
        iteration += numWorkItems;
  }
}
#if 0
__attribute__((num_compute_units(4)))
__kernel void probeHashTable(__global uint * restrict rHashTable, __global uint * restrict sTableOnDevice, __global uint * restrict matchedTable, 
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
			hash = key &(rHashTableBucketNum-1);//sim_hash(key,rHashTableBucketNum);
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
