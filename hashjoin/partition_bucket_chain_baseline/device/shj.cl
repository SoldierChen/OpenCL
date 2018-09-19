/* simple hash join on BRAM */
//----channel define----//
channel uint2 relR __attribute__((depth(2048)));
channel uint2 relS __attribute__((depth(2048)));
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
                        __global uint2 * restrict sTable, 
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
    for(int i = sTableOffset; i < (sTableOffset + sTableReadNum); i ++){
      uint2 stable_uint2 = sTable[i];
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

    for(int i = 0; i < HASHTABLE_L_SIZE; i ++) {
        next[i] = 0;
        key[i] = 0;
    }
    for(int i = 0; i < HASHTABLE_BUCKET_NUM; i ++) {
        bucket[i] = 0;
    }


    const uint numR = rTableReadRange[0].y;

    //uint32_t N = numR;
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
    for(uint i=0; i < numS; i++ ){
        uint2 stable_uint2 = read_channel_altera(relS);

        uint idx = HASH(stable_uint2.x, MASK, 0);

        for(int hit = bucket[idx]; hit > 0; hit = next[hit-1]){

            if(key[hit-1] == stable_uint2.x){
                /* TODO: copy to the result buffer, we skip it */
                matches ++;
            }
        }
    }
    /* PROBE-LOOP END  */
    matchedTable[0] = matches;
}
