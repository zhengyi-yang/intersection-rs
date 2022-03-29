#ifndef _INTER_ALGOS_H
#define _INTER_ALGOS_H

#include "intersection/include/util.hpp"

// ScalarMerge:
// int intersect_scalarmerge_uint(const int *set_a, int size_a,
//            const int *set_b, int size_b, int *set_c);
// ScalarMerge+BSR:
// int intersect_scalarmerge_bsr(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);

// ScalarGalloping:
// int intersect_scalargalloping_uint(const int *set_a, int size_a,
//            const int *set_b, int size_b, int *set_c);
// ScalarGalloping+BSR:
// int intersect_scalargalloping_bsr(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);

// SIMDGalloping:
int intersect_simdgalloping_uint(const unsigned int *set_a, int size_a,
                                 const unsigned int *set_b, int size_b,
                                 unsigned int *set_c, bool count_only);
// SIMDGalloping+BSR:
// int intersect_simdgalloping_bsr(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);

// QFilter:
int intersect_qfilter_uint_b4(const unsigned int *set_a, int size_a,
                              const unsigned int *set_b, int size_b,
                              unsigned int *set_c, bool count_only);
int intersect_qfilter_uint_b4_v2(const int *set_a, int size_a, const int *set_b,
                                 int size_b, int *set_c);

// QFilter+BSR:
// int intersect_qfilter_bsr_b4(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);
// int intersect_qfilter_bsr_b4_v2(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);

// Shuffling:
int intersect_shuffle_uint_b4(const int *set_a, int size_a, const int *set_b,
                              int size_b, int *set_c);
int intersect_shuffle_uint_b8(const int *set_a, int size_a, const int *set_b,
                              int size_b, int *set_c);
int intersect_shuffle_uint_vec256(const int *set_a, int size_a,
                                  const int *set_b, int size_b, int *set_c);
// Shuffling+BSR:
// int intersect_shuffle_bsr_b4(int* bases_a, int* states_a, int size_a,
//            int* bases_b, int* states_b, int size_b,
//            int* bases_c, int* states_c);
#endif
