#ifndef PARALLEL_RADIX_JOIN_H
#define PARALLEL_RADIX_JOIN_H

#include "types.h" /* relation_t */

int64_t
RJ(relation_t * relR, relation_t * relS, int nthreads);

#endif
