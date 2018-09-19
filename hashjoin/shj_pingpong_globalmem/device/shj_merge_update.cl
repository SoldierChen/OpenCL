#pragma OPENCL EXTENSION cl_altera_channels : enable
typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define CHANNEL_NUM 4
channel bucket_cell_t update_channel[16] __attribute__ ((depth(1024)));
uint sim_hash(uint key, uint mod)
{
	return key % mod;
}
__kernel void  __attribute__((task)) hash_div_func(__global uint * restrict rTable, __global uint * restrict rHashTable,const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
  int cnt = 0;
 // uint tid = get_global_id(0);
 // uint numWorkItems = get_global_size(0);
#pragma unroll 4
  for (uint i = 0; i < size; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x; 
      uint val  = rtable_uint2.y;
      uint hash_index = key & (rHashTableBucketNum-1);
      bucket_cell_t bucket_cell;
      bucket_cell.hash_index = hash_index;
      bucket_cell.key = key;
      bucket_cell.value = val;
      uint tid =  i % 4;
      uint channel_group = (int)hash_index / (rHashTableBucketNum/CHANNEL_NUM);
      cnt ++;

      printf("current tuple is %d \n", cnt);
      switch (tid)
      {
           case 0: {{ 
                  switch (channel_group+4)
                    case 4: write_channel_altera(update_channel[0],bucket_cell);break;
                    case 5: write_channel_altera(update_channel[1],bucket_cell);break;
                    case 6: write_channel_altera(update_channel[2],bucket_cell);break;
                    case 7: write_channel_altera(update_channel[3],bucket_cell);break;
                } break;}

          case 1: {{ 
                  switch (channel_group+8)
                    case 8: write_channel_altera(update_channel[4],bucket_cell);break;
                    case 9: write_channel_altera(update_channel[5],bucket_cell);break;
                    case 10: write_channel_altera(update_channel[6],bucket_cell);break;
                    case 11: write_channel_altera(update_channel[7],bucket_cell);break;
                } break;}

          case 2: {{ 
                  switch (channel_group+12)
                    case 12: write_channel_altera(update_channel[8],bucket_cell);break;
                    case 13: write_channel_altera(update_channel[9],bucket_cell);break;
                    case 14: write_channel_altera(update_channel[10],bucket_cell);break;
                    case 15: write_channel_altera(update_channel[11],bucket_cell);break;
                } break;}

          case 3: {{ 
                  switch (channel_group+16)
                    case 16: write_channel_altera(update_channel[12],bucket_cell);break;
                    case 17: write_channel_altera(update_channel[13],bucket_cell);break;
                    case 18: write_channel_altera(update_channel[14],bucket_cell);break;
                    case 19: write_channel_altera(update_channel[15],bucket_cell);break;
                } break;}
     }
     // mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
     // tid += numWorkItems;
  }
}
__kernel void __attribute__((task))
  update_table( __global uint * restrict rHashTable,const uint hashBucketSize)
{
#if 1
  for(int i = 0; i < 100; i ++){
    bool ret;
    //printf("liangliang\n");
    bucket_cell_t bucket_cell[8]; // may not support dynamic read channel, use the switch case to do it
   {
     bucket_cell[0] = read_channel_altera(update_channel[0]);
     bucket_cell[1] = read_channel_altera(update_channel[1]);
     bucket_cell[2] = read_channel_altera(update_channel[2]);
     bucket_cell[3] = read_channel_altera(update_channel[3]);
   }
     uint j[CHANNEL_NUM];
   for(int i = 0; i < 4; i++){
    for(j[0] = 0;j[0] < hashBucketSize; j[0] ++){
        if (rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] == 0)
           break;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] = bucket_cell[i].key;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 1] = bucket_cell[i].value;
    }
   }
   {
     bucket_cell[0] = read_channel_altera(update_channel[4]);
     bucket_cell[1] = read_channel_altera(update_channel[5]);
     bucket_cell[2] = read_channel_altera(update_channel[6]);
     bucket_cell[3] = read_channel_altera(update_channel[7]);
   }
   for(int i = 0; i < 4; i++){
    for(j[0] = 0;j[0] < hashBucketSize; j[0] ++){
        if (rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] == 0)
           break;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] = bucket_cell[i].key;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 1] = bucket_cell[i].value;
    }
   }
 {
     bucket_cell[0] = read_channel_altera(update_channel[8]);
     bucket_cell[1] = read_channel_altera(update_channel[9]);
     bucket_cell[2] = read_channel_altera(update_channel[10]);
     bucket_cell[3] = read_channel_altera(update_channel[11]);
   }
   for(int i = 0; i < 4; i++){
    for(j[0] = 0;j[0] < hashBucketSize; j[0] ++){
        if (rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] == 0)
           break;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] = bucket_cell[i].key;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 1] = bucket_cell[i].value;
    }
   }
 {
     bucket_cell[0] = read_channel_altera(update_channel[12]);
     bucket_cell[1] = read_channel_altera(update_channel[13]);
     bucket_cell[2] = read_channel_altera(update_channel[14]);
     bucket_cell[3] = read_channel_altera(update_channel[15]);
   }
   for(int i = 0; i < 4; i++){
    for(j[0] = 0;j[0] < hashBucketSize; j[0] ++){
        if (rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] == 0)
           break;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 0] = bucket_cell[i].key;
    rHashTable[bucket_cell[i].hash_index * hashBucketSize * 2 + j[0] * 2 + 1] = bucket_cell[i].value;
    }
   }

}
#endif
}
