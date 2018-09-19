/* simple hash join on BRAM */
//----channel define----//
#define ENDFLAG 0xffff
channel uint2 relR[8][16] __attribute__((depth(8)));
channel uint2 relS[8][16] __attribute__((depth(8)));
channel uint relRendFlagCh __attribute__((depth(128)));
channel uint relSendFlagCh __attribute__((depth(128)));


#define HASH(K, MASK, SKIP) (((K) & MASK) >> SKIP)
#define RELR_L_NUM 1024*256*1
#define HASHTABLE_L_SIZE 1024*256*1
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
    //printf("sTableReadNum %d, \n", rTableReadNum);
  
 //   for(int i = rTableOffset; i < rTableOffset + rTableReadNum; i ++){
    for(int i = 0; i < rTableReadNum; i += 8){
      #pragma unroll 8
        for(int k = 0; k < 8; k ++){
          uint2 rtable_uint2 = rTable[i+k];
          switch((rtable_uint2.x) & 0xf){
            case 0 :  write_channel_altera(relR[k][0], rtable_uint2); break;
            case 1 :  write_channel_altera(relR[k][1], rtable_uint2); break;
            case 2 :  write_channel_altera(relR[k][2], rtable_uint2); break;
            case 3 :  write_channel_altera(relR[k][3], rtable_uint2); break;
            case 4 :  write_channel_altera(relR[k][4], rtable_uint2); break;
            case 5 :  write_channel_altera(relR[k][5], rtable_uint2); break;
            case 6 :  write_channel_altera(relR[k][6], rtable_uint2); break;
            case 7 :  write_channel_altera(relR[k][7], rtable_uint2); break;
            case 8 :  write_channel_altera(relR[k][8], rtable_uint2); break;
            case 9 :  write_channel_altera(relR[k][9], rtable_uint2); break;
            case 10 : write_channel_altera(relR[k][10], rtable_uint2); break;
            case 11 : write_channel_altera(relR[k][11], rtable_uint2); break;
            case 12 : write_channel_altera(relR[k][12], rtable_uint2); break;
            case 13 : write_channel_altera(relR[k][13], rtable_uint2); break;
            case 14 : write_channel_altera(relR[k][14], rtable_uint2); break;
            case 15 : write_channel_altera(relR[k][15], rtable_uint2); break;
          }
        }
      }
 

  write_channel_altera(relRendFlagCh, ENDFLAG);

    //for(int i = sTableOffset; i < (sTableOffset + sTableReadNum); i ++){
    for(int i = 0; i < (sTableReadNum); i += 8){
      #pragma unroll 8
        for(int k = 0; k < 8; k ++){
          uint2 stable_uint2 = sTable[i + k];
          switch((stable_uint2.x)& 0xf){
            case 0 :  write_channel_altera(relS[k][0], stable_uint2); break;
            case 1 :  write_channel_altera(relS[k][1], stable_uint2); break;
            case 2 :  write_channel_altera(relS[k][2], stable_uint2); break;
            case 3 :  write_channel_altera(relS[k][3], stable_uint2); break;
            case 4 :  write_channel_altera(relS[k][4], stable_uint2); break;
            case 5 :  write_channel_altera(relS[k][5], stable_uint2); break;
            case 6 :  write_channel_altera(relS[k][6], stable_uint2); break;
            case 7 :  write_channel_altera(relS[k][7], stable_uint2); break;
            case 8 :  write_channel_altera(relS[k][8], stable_uint2); break;
            case 9 :  write_channel_altera(relS[k][9], stable_uint2); break;
            case 10 : write_channel_altera(relS[k][10], stable_uint2); break;
            case 11 : write_channel_altera(relS[k][11], stable_uint2); break;
            case 12 : write_channel_altera(relS[k][12], stable_uint2); break;
            case 13 : write_channel_altera(relS[k][13], stable_uint2); break;
            case 14 : write_channel_altera(relS[k][14], stable_uint2); break;
            case 15 : write_channel_altera(relS[k][15], stable_uint2); break;
        }
      }
    }

  write_channel_altera(relSendFlagCh, ENDFLAG);

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
  
    uint rTableReadNum = rTableReadRange[0].y;
    bool engine_finish[16] = {false};  

    uint2 data_r[8][16];
    bool valid_r[8][16];
    #pragma unroll 16
      for(int i = 0; i < 16; i ++){
      #pragma unroll 16
        for(int j = 0; j < 8; j ++)
          valid_r[j][i] = false;
      } 

    while(true){
      #pragma unroll 16
        for(int i = 0; i < 16; i ++){ 
        // each collect engine do their work 
          #pragma unroll 8
            for(int j = 0; j < 8; j ++){  
              data_r[j][i] = read_channel_nb_altera(relR[j][i], &valid_r[j][i]);
            }
            // low is active
            engine_finish[i] = valid_r[0][i] | valid_r[1][i] | valid_r[2][i] | valid_r[3][i] | 
                               valid_r[4][i] | valid_r[5][i] | valid_r[6][i] | valid_r[7][i] ;

              if(valid_r[0][i]){
                uint key  = data_r[0][i].x; 
                uint val  = data_r[0][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[0][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[1][i]){
                uint key  = data_r[1][i].x; 
                uint val  = data_r[1][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[1][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[2][i]){
                uint key  = data_r[2][i].x; 
                uint val  = data_r[2][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[2][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[3][i]){
                uint key  = data_r[3][i].x; 
                uint val  = data_r[3][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[3][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[4][i]){
                uint key  = data_r[4][i].x; 
                uint val  = data_r[4][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[4][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[5][i]){
                uint key  = data_r[5][i].x; 
                uint val  = data_r[5][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[5][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[6][i]){
                uint key  = data_r[6][i].x; 
                uint val  = data_r[6][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[6][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }

              else if(valid_r[7][i]){
                uint key  = data_r[7][i].x; 
                uint val  = data_r[7][i].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[7][i];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
              }
          /* 
            for(int j = 0; j < 8; j ++){  
                if(valid_r[j] == 0) continue;
                uint key  = data_r[j].x; 
                uint val  = data_r[j].y;
                uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
                hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= data_r[j];
                hashtable_bucket_cnt[hash_idx][i] ++; 
               // printf("key %d \n", key);
            }
          */
        //--------------------------------//

        }

      // low is active
      bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
                        engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7] |
                        engine_finish[8] | engine_finish[9] | engine_finish[10]| engine_finish[11]| 
                        engine_finish[12]| engine_finish[13]| engine_finish[14]| engine_finish[15];

      bool valid_endflag = false;
      uint endFlagData;
      uint endFlag = read_channel_nb_altera (relRendFlagCh, &valid_endflag);
      if(valid_endflag ) endFlagData =  endFlag;
      if(endFlagData == ENDFLAG && !all_finish) break; 
    }
    //  probe phrase
    uint matchedCnt[16] = {0}; 
    uint iter = (sTableReadRange[0].y);

    
    while(true){

       bool engine_finish[16] = {false};

    #pragma unroll 16
      for(int i = 0; i < 16; i ++){
      // each collect engine do their work 
        bool valid_s[8] = {false};
        uint2 data_s[8];
        #pragma unroll 8
          for(int j = 0; j < 8; j ++){  
              data_s[j] = read_channel_nb_altera(relS[j][i], &valid_s[j]);
          }

          engine_finish[i] = valid_s[0] | valid_s[1] | valid_s[2]  | valid_s[3] | 
                             valid_s[4] | valid_s[5] | valid_s[6]  | valid_s[7] ;


          for(int j = 0; j < 8; j ++){  
            if(valid_s[j]){
              uint key = data_s[j].x;
              uint hash_idx = HASH (key, (HASHTABLE_BUCKET_NUM - 1),4);
              //printf("probe key %d \n", key);
              for(int k = 0; k < HASHTABLE_BUCKET_SIZE; k ++){
                    //uint hashTable_val = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + j].y;
                  uint hashTable_key = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + k][i].x;
                  if(key == hashTable_key){
                //   matchedTable[matchedCnt] = key;
                //   matchedTable[matchedCnt + 1] = rtable_uint2.y;
                //   matchedTable[matchedCnt + 2] = hashTable_val;
                      matchedCnt[i] ++;
                  }
              }
            }
          }
      //--------------------------------------//
      }
      // low level is active
      bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
                  engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7] |
                  engine_finish[8] | engine_finish[9] | engine_finish[10]| engine_finish[11]| 
                  engine_finish[12]| engine_finish[13]| engine_finish[14]| engine_finish[15];

      bool valid_endflag = false;
      uint endFlagData;
      uint endFlag = read_channel_nb_altera (relSendFlagCh, &valid_endflag);
      if(valid_endflag ) endFlagData =  endFlag;
      if(endFlagData == ENDFLAG && !all_finish) break; 
    }
    
    uint total_cnt = 0;
  #pragma unroll 16
    for(int i = 0; i < 16; i ++){
        total_cnt += matchedCnt[i];
    }
    matchedTable[0] = total_cnt;


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
