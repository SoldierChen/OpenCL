/* simple hash join on BRAM */
//----channel define----//
channel uint2 relR __attribute__((depth(2048)));
channel uint16 relS __attribute__((depth(2048)));
#define HASH(K, MASK, SKIP) (((K) & MASK) >> SKIP)
#define RELR_L_NUM 1024*256
#define HASHTABLE_L_SIZE 1024*256
#define HASHTABLE_BUCKET_SIZE 4 // 2 byte per 
#define HASHTABLE_BUCKET_NUM HASHTABLE_L_SIZE/HASHTABLE_BUCKET_SIZE
//#define SW
__attribute__((task))
__kernel void relRead (
                        __global uint2 * restrict rTable, 
                        __global uint2 * restrict rTableReadRange, 
                        __global uint16 * restrict sTable, 
                        __global uint2 * restrict sTableReadRange
                      )
{
    uint rTableOffset = rTableReadRange[0].x;
    uint rTableReadNum = rTableReadRange[0].y;
    uint sTableOffset = sTableReadRange[0].x;
    uint sTableReadNum = sTableReadRange[0].y;

    for(int i = rTableOffset; i < (rTableOffset + rTableReadNum); i ++){
      uint2 rtable_uint2 = rTable[i];
      write_channel_altera(relR, rtable_uint2);
    }
    for(int i = sTableOffset / 8; i < (sTableOffset + sTableReadNum) / 8; i ++){
      uint16 stable_uint2 = sTable[i];
      write_channel_altera(relS, stable_uint2);
    }
}

__kernel void hashjoin(
                        __global uint * restrict matchedTable, 
                        __global uint2 * restrict rTableReadRange,  
                        __global uint2 * restrict sTableReadRange
                      )
{
    //int * next, * bucket;
    int next[HASHTABLE_L_SIZE];
    int key[HASHTABLE_L_SIZE];
    int bucket[HASHTABLE_BUCKET_NUM];
    uint idx[8];
    uint keyS[8];

    for(int i = 0; i < HASHTABLE_L_SIZE; i ++) {
        next[i] = 0;
        key[i] = 0;
    }
    for(int i = 0; i < HASHTABLE_BUCKET_NUM; i ++) {
        bucket[i] = 0;
    }


    const uint numR = rTableReadRange[0].y;

    uint matches = 0;

    const uint MASK = (HASHTABLE_BUCKET_NUM-1);

    for(uint i=0; i < numR; ){
        uint2 rtable_uint2 = read_channel_altera(relR);
        uint idx = HASH(rtable_uint2.x, MASK, 0);
        next[i] = bucket[idx];
        key[i] = rtable_uint2.x;
        bucket[idx]  = ++i;     /* we start pos's from 1 instead of 0 */
        /* Enable the following tO avoid the code elimination
           when running probe only for the time break-down experiment */
        /* matches += idx; */
    }

    const uint  numS = sTableReadRange[0].y;

    /* Disable the following loop for no-probe for the break-down experiments */
    /* PROBE- LOOP */
    #pragma ii 4
    for(uint i=0; i < numS / 8; i++ ){
        uint16 stable_uint16 = read_channel_altera(relS);


        keyS[0] = stable_uint16.s0;
        keyS[1] = stable_uint16.s2;
        keyS[2] = stable_uint16.s4;
        keyS[3] = stable_uint16.s6;
        keyS[4] = stable_uint16.s8;
        keyS[5] = stable_uint16.sa;
        keyS[6] = stable_uint16.sc;
        keyS[7] = stable_uint16.se;

        #pragma unroll 8
        for(int i = 0; i < 8; i ++){
            idx[i] = HASH(keyS[i], MASK, 0);
        }


        #pragma unroll 8
        for(int i = 0; i < 8; i ++){
            for(int hit = bucket[idx[i]]; hit > 0; hit = next[hit-1]){

                if(key[hit-1] == keyS[i]){
                    /* TODO: copy to the result buffer, we skip it */
                    matches ++;
                }
            }
        }
    }
    /* PROBE-LOOP END  */
    matchedTable[0] = matches;
}
