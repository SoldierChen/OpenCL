#pragma OPENCL EXTENSION cl_altera_channels : enable
typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define CHANNEL_NUM 4
channel uint update_channel[4] __attribute__ ((depth(16)));
uint sim_hash(uint key, uint mod)
{
	return key % mod;
}
__kernel void  __attribute__((task)) hash_div_func(__global uint * restrict rTable, __global uint * restrict rHashTable,const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
  int cnt = 0;
#pragma unroll 1
  for (uint i = 0; i < size; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x; 
      uint channel_group = (uint)key % 4;
      cnt ++;
      printf("current tuple is %d \n", cnt);

  //    write_channel_altera(update_channel[0],key);
                  switch (channel_group){
                    case 0: write_channel_altera(update_channel[0],key);break;
                    case 1: write_channel_altera(update_channel[1],key);break;
                    case 2: write_channel_altera(update_channel[2],key);break;
                    case 3: write_channel_altera(update_channel[3],key);break;
                   // default: write_channel_altera(update_channel[3],key);break;
                  }
     // mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}
__kernel void __attribute__((task))
  update_table( __global uint * restrict rHashTable,const uint hashBucketSize)
{
#if 1

  uint cnt = 0;
  for(int i = 0; i < 100; i ++){
    uint ret;
    printf("liangliang\n");
    uint bucket[4]= {0,0,0,0}; // may not support dynamic read channel, use the switch case to do it
   {
     bucket[0] = read_channel_altera(update_channel[0]);
     bucket[1] = read_channel_altera(update_channel[1]);
     bucket[2] = read_channel_altera(update_channel[2]);
     bucket[3] = read_channel_altera(update_channel[3]);
   }
    cnt ++;
    printf("upadate cnt %d \n",cnt);
   rHashTable[i] = bucket[0]  + bucket[1] +bucket[2] +bucket[3] ;
  }
#endif
}
