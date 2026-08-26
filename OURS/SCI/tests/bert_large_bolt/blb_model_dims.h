#pragma once

// The original BLB artifact is compiled for BERT-base.  Keep those values as
// the default so the donor binary remains byte-for-byte compatible at the
// layout boundary, while allowing a separately compiled attention target to
// select another head count/hidden width.  INPUT_DIM is the *physical* token
// tile; a workload may expose fewer logical tokens through a public mask.
#ifndef PACKING_NUM
#define PACKING_NUM 12
#endif

#ifndef INPUT_DIM
#define INPUT_DIM 128
#endif

#ifndef COMMON_DIM
#define COMMON_DIM 768
#endif

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 64
#endif

#ifndef INTER_DIM
#define INTER_DIM 3072
#endif

#ifndef ATTENTION_LAYERS
#define ATTENTION_LAYERS 1
#endif

#ifndef BLB_QKV_WEIGHT_FRACTIONAL_BITS
#define BLB_QKV_WEIGHT_FRACTIONAL_BITS 6
#endif

#ifndef BLB_QKV_BIAS_FRACTIONAL_BITS
#define BLB_QKV_BIAS_FRACTIONAL_BITS 11
#endif

static_assert(PACKING_NUM > 0 && PACKING_NUM % 2 == 0,
              "BLB cross packing requires an even number of heads");
static_assert(COMMON_DIM == PACKING_NUM * OUTPUT_DIM,
              "hidden width must equal heads times head width");
