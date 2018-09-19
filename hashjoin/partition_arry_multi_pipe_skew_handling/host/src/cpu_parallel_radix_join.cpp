/** \copydoc RJ */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>              /* CPU_ZERO, CPU_SET */
#include <pthread.h>            /* pthread_* */
#include <stdlib.h>             /* malloc, posix_memalign */
#include <sys/time.h>           /* gettimeofday */
#include <stdio.h>              /* printf */
//#include <smmintrin.h>          /* simd only for 32-bit keys – SSE4.1 */

#include "cpu_parallel_radix_join.h"
#include "prj_params.h"  

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)
#ifndef NEXT_POW_2

#define NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

int64_t 
bucket_chaining_join(const relation_t * const R, 
                     const relation_t * const S,
                     relation_t * const tmpR)
{
    int * next, * bucket;
    const uint32_t numR = R->num_tuples;
    uint32_t N = numR;
    int64_t matches = 0;

    NEXT_POW_2(N);
    /* N <<= 1; */
    const uint32_t MASK = (N-1) << (NUM_RADIX_BITS);

    next   = (int*) malloc(sizeof(int) * numR);
    /* posix_memalign((void**)&next, CACHE_LINE_SIZE, numR * sizeof(int)); */
    bucket = (int*) calloc(N, sizeof(int));

    const tuple_t * const Rtuples = R->tuples;
    for(uint32_t i=0; i < numR; ){
        uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, NUM_RADIX_BITS);
        next[i]      = bucket[idx];
        bucket[idx]  = ++i;     /* we start pos's from 1 instead of 0 */

        /* Enable the following tO avoid the code elimination
           when running probe only for the time break-down experiment */
        /* matches += idx; */
    }

    const tuple_t * const Stuples = S->tuples;
    const uint32_t        numS    = S->num_tuples;

    /* Disable the following loop for no-probe for the break-down experiments */
    /* PROBE- LOOP */
    for(uint32_t i=0; i < numS; i++ ){

        uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, NUM_RADIX_BITS);

        for(int hit = bucket[idx]; hit > 0; hit = next[hit-1]){

            if(Stuples[i].key == Rtuples[hit-1].key){
                /* TODO: copy to the result buffer, we skip it */
                matches ++;
            }
        }
    }
    /* PROBE-LOOP END  */
    
    /* clean up temp */
    free(bucket);
    free(next);

    return matches;
}


void
radix_cluster_nopadding(relation_t * outRel, relation_t * inRel, int R, int D)
{
    tuple_t ** dst;
    tuple_t * input;
    /* tuple_t ** dst_end; */
    uint32_t * tuples_per_cluster;
    uint32_t i;
    uint32_t offset;
    const uint32_t M = ((1 << D) - 1) << R;
    const uint32_t fanOut = 1 << D;
    const uint32_t ntuples = inRel->num_tuples;

    tuples_per_cluster = (uint32_t*)calloc(fanOut, sizeof(uint32_t));
    dst     = (tuple_t**)malloc(sizeof(tuple_t*)*fanOut);
    input = inRel->tuples;
    for( i=0; i < ntuples; i++ ){
            uint32_t idx = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
            tuples_per_cluster[idx]++;
            input++;
        }

        offset = 0;
            /* determine the start and end of each cluster depending on the counts. */
            for ( i=0; i < fanOut; i++ ) {
                    dst[i]      = outRel->tuples + offset;
                    offset     += tuples_per_cluster[i];
                    /* dst_end[i]  = outRel->tuples + offset; */
                }

    input = inRel->tuples;
    /* copy tuples to their corresponding clusters at appropriate offsets */
    for( i=0; i < ntuples; i++ ){
            uint32_t idx   = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
            *dst[idx] = *input;
            ++dst[idx];
            input++;
        }

    free(dst);
    free(tuples_per_cluster);
}

int64_t
RJ(relation_t * relR, relation_t * relS, int nthreads)
{
    int64_t result = 0;
    uint32_t i;

    relation_t *outRelR, *outRelS;

    outRelR = (relation_t*) malloc(sizeof(relation_t));
    outRelS = (relation_t*) malloc(sizeof(relation_t));

    /* allocate temporary space for partitioning */
    /* TODO: padding problem */
    size_t sz = relR->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
    outRelR->tuples     = (tuple_t*) malloc(sz);
    outRelR->num_tuples = relR->num_tuples;

    sz = relS->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
    outRelS->tuples     = (tuple_t*) malloc(sz);
    outRelS->num_tuples = relS->num_tuples;

    /***** do the multi-pass partitioning *****/
#if NUM_PASSES==1
    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, 0, NUM_RADIX_BITS);
    relR = outRelR;

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, 0, NUM_RADIX_BITS);
    relS = outRelS;

#elif NUM_PASSES==2
    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, 0, NUM_RADIX_BITS/NUM_PASSES);

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, 0, NUM_RADIX_BITS/NUM_PASSES);

    /* apply radix-clustering on relation R for pass-2 */
    radix_cluster_nopadding(relR, outRelR,
                                    NUM_RADIX_BITS/NUM_PASSES, 
                                                                NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES));

    /* apply radix-clustering on relation S for pass-2 */
    radix_cluster_nopadding(relS, outRelS,
                                    NUM_RADIX_BITS/NUM_PASSES, 
                                                                NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES));

    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);

#else
#error Only 1 or 2 pass partitioning is implemented, change NUM_PASSES!
#endif

    int * R_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));
    int * S_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));

    /* compute number of tuples per cluster */
    for( i=0; i < relR->num_tuples; i++ ){
            uint32_t idx = (relR->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
            R_count_per_cluster[idx] ++;
        }
    for( i=0; i < relS->num_tuples; i++ ){
            uint32_t idx = (relS->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
            S_count_per_cluster[idx] ++;
        }

    /* build hashtable on inner */
    int r, s; /* start index of next clusters */
    r = s = 0;
    for( i=0; i < (1<<NUM_RADIX_BITS); i++ ){
            relation_t tmpR, tmpS;
 
            if(R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0){
 
                        tmpR.num_tuples = R_count_per_cluster[i];
                        tmpR.tuples = relR->tuples + r;
                        r += R_count_per_cluster[i];

                        tmpS.num_tuples = S_count_per_cluster[i];
                        tmpS.tuples = relS->tuples + s;
                        s += S_count_per_cluster[i];
 
                        result += bucket_chaining_join(&tmpR, &tmpS, NULL);
                    }
            else {
                        r += R_count_per_cluster[i];
                        s += S_count_per_cluster[i];
                    }
        }

    /* clean-up temporary buffers */
    free(S_count_per_cluster);
    free(R_count_per_cluster);

#if NUM_PASSES == 1
    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);
#endif

    return result;
}

