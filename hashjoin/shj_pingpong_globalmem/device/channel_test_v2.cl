#pragma OPENCL EXTENSION cl_altera_channels : enable
typedef struct {
  uint key;
  uint hash_index;
  uint value;
} bucket_cell_t;
#define CHANNEL_NUM 4
#define END_FLAG 0x7ffffff0
channel uint update_channel0 __attribute__ ((depth(16)));
channel uint update_channel1 __attribute__ ((depth(16)));
channel uint update_channel2 __attribute__ ((depth(16)));
channel uint update_channel3 __attribute__ ((depth(16)));
uint sim_hash(uint key, uint mod)
{
	return key % mod;
}
__kernel void  __attribute__((task)) hash_div_func(__global uint * restrict rTable, __global uint * restrict rHashTable,const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
 // int cnt = 0;
  for (uint i = 0; i < size ; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x;
      uint value = rtable_uint2.y;
      uint channel_group = (uint)key % 4;     
      if(channel_group == 0 )
        write_channel_altera(update_channel0,value);
}

  for (uint i = 0; i < size ; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x;
      uint value = rtable_uint2.y;
      uint channel_group = (uint)key % 4;     
      if(channel_group == 1 )
        write_channel_altera(update_channel1,value);
}

  for (uint i = 0; i < size ; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x;
      uint value = rtable_uint2.y;
      uint channel_group = (uint)key % 4;     
      if(channel_group == 2 )
        write_channel_altera(update_channel2,value);
}

  for (uint i = 0; i < size ; i ++){
  		uint2 rtable_uint2 = *(__global uint2 *)(&rTable[offset*2+i*2]);
		  uint key  = rtable_uint2.x;
      uint value = rtable_uint2.y;
      uint channel_group = (uint)key % 4;     
      if(channel_group == 3 )
        write_channel_altera(update_channel3,value);
}

      // cnt ++;
     // if( value == END_FLAG)
     // printf("current tuple is %d \n", cnt);

     // write_channel_altera(update_channel,value);
          /*        switch (channel_group){
                    case 0: write_channel_altera(update_channel[0],value);break;
                    case 1: write_channel_altera(update_channel[1],value);break;
                    case 2: write_channel_altera(update_channel[2],value);break;
                    case 3: write_channel_altera(update_channel[3],value);break;
                   // default: write_channel_altera(update_channel[3],key);break;
                  }*/
    // mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
__kernel void __attribute__((task))
  update_table( __global uint * restrict rHashTable,const uint hashBucketSize)
{
#if 1

 // uint cnt = 0;
  while(1){
    //printf("liangliang\n");
   // uint bucket[4]= {0,0,0,0}; // may not support dynamic read channel, use the switch case to do it
       uint  data0,data1,data2,data3; bool ret0,ret1,ret2,ret3;
       data0 = read_channel_nb_altera(update_channel0,&ret0);
       if(ret0)
       rHashTable[0] = data0;
       
       data1 = read_channel_nb_altera(update_channel1,&ret1);
       if(ret1)
       rHashTable[1] = data1;

       data2 = read_channel_nb_altera(update_channel2,&ret2);
        if(ret2)
        rHashTable[2] = data2;
       data3 = read_channel_nb_altera(update_channel3,&ret3);
        if(ret3)
       rHashTable[3] = data3;


//mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//   cnt ++;
    if(rHashTable[0] == END_FLAG)// && rHashTable[1] == END_FLAG && rHashTable[2] == END_FLAG && rHashTable[3] == END_FLAG ){ 
   // printf("END_FLAG FOUND \n");
    break;
    }
#endif
}
