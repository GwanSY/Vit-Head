/*
Authors: Mayank Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "BuildingBlocks/value-extension.h"

using namespace std;
using namespace sci;

XTProtocol::XTProtocol(int party, IOPack *iopack, OTPack *otpack) {
  this->party = party;
  this->iopack = iopack;
  this->otpack = otpack;
  this->aux = new AuxProtocols(party, iopack, otpack);
  this->millionaire = this->aux->mill;
  this->triple_gen = this->millionaire->triple_gen;
}

XTProtocol::~XTProtocol() { delete aux; };

void XTProtocol::z_extend(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  uint64_t mask_bwA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
  uint64_t mask_bwB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
  uint8_t *wrap = new uint8_t[dim];

  if (msbA != nullptr) {
    this->aux->MSB_to_Wrap(inA, msbA, wrap, dim, bwA);
  } else {
    this->aux->wrap_computation(inA, wrap, dim, bwA);
  }

  uint64_t *arith_wrap = new uint64_t[dim];
  this->aux->B2A(wrap, arith_wrap, dim, (bwB - bwA));

  for (int i = 0; i < dim; i++) {
    outB[i] = ((inA[i] & mask_bwA) - (1ULL << bwA) * arith_wrap[i]) & mask_bwB;
  }

  delete[] wrap;
  delete[] arith_wrap;
}

void XTProtocol::s_extend(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  uint64_t mask_bwA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
  uint64_t mask_bwB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));

  uint64_t *mapped_inA = new uint64_t[dim];
  uint64_t *mapped_outB = new uint64_t[dim];
  if (party ==sci::ALICE) {
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = (inA[i] + (1ULL << (bwA - 1))) & mask_bwA;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = inA[i];
    }
  }

  uint8_t *tmp_msbA = nullptr;
  if (msbA != nullptr) {
    tmp_msbA = new uint8_t[dim];
    for (int i = 0; i < dim; i++) {
      tmp_msbA[i] = (party ==sci::ALICE ? msbA[i] ^ 1 : msbA[i]);
    }
  }
  this->z_extend(dim, mapped_inA, mapped_outB, bwA, bwB, tmp_msbA);
  if (msbA != nullptr) {
    delete[] tmp_msbA;
  }

  if (party ==sci::ALICE) {
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i] - (1ULL << (bwA - 1))) & mask_bwB;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i]) & mask_bwB;
    }
  }
  delete[] mapped_inA;
  delete[] mapped_outB;
}

void XTProtocol::s_extend_wide(int32_t dim, const uint64_t *inA,
                               uint64_t *out_low, uint64_t *out_high,
                               int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (dim <= 0 || bwA <= 1 || bwA >= 64 || bwB <= 64 || bwB > 127 ||
      bwB <= bwA || out_low == nullptr || out_high == nullptr) {
    throw std::invalid_argument("invalid wide signed-extension metadata");
  }

  const uint64_t mask_a = (uint64_t{1} << bwA) - 1;
  const uint64_t sign_a = uint64_t{1} << (bwA - 1);
  const int32_t delta = bwB - bwA;
  const __uint128_t modulus_b = __uint128_t{1} << bwB;
  const __uint128_t mask_b = modulus_b - 1;

  vector<uint64_t> mapped_in(static_cast<size_t>(dim));
  for (int32_t i = 0; i < dim; ++i) {
    const uint64_t value = inA[i] & mask_a;
    mapped_in[i] = party == sci::ALICE ? (value + sign_a) & mask_a : value;
  }

  vector<uint8_t> mapped_msb;
  uint8_t *mapped_msb_ptr = nullptr;
  if (msbA != nullptr) {
    mapped_msb.resize(static_cast<size_t>(dim));
    for (int32_t i = 0; i < dim; ++i) {
      mapped_msb[i] = party == sci::ALICE
                          ? static_cast<uint8_t>(msbA[i] ^ 1)
                          : msbA[i];
    }
    mapped_msb_ptr = mapped_msb.data();
  }

  vector<uint8_t> wrap(static_cast<size_t>(dim));
  if (mapped_msb_ptr != nullptr) {
    aux->MSB_to_Wrap(mapped_in.data(), mapped_msb_ptr, wrap.data(), dim,
                     bwA);
  } else {
    aux->wrap_computation(mapped_in.data(), wrap.data(), dim, bwA);
  }
  vector<uint64_t> arithmetic_wrap(static_cast<size_t>(dim));
  aux->B2A(wrap.data(), arithmetic_wrap.data(), dim, delta);

  const __uint128_t low_mask = static_cast<__uint128_t>(~uint64_t{0});
  for (int32_t i = 0; i < dim; ++i) {
    __uint128_t extended =
        (__uint128_t(mapped_in[i]) -
         (__uint128_t{1} << bwA) * arithmetic_wrap[i]) & mask_b;
    if (party == sci::ALICE) {
      extended = (extended - sign_a) & mask_b;
    }
    out_low[i] = static_cast<uint64_t>(extended & low_mask);
    out_high[i] = static_cast<uint64_t>(extended >> 64);
  }
}

void XTProtocol::z_extend_wide(int32_t dim, uint64_t *inA,
                               uint64_t *out_low, uint64_t *out_high,
                               int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (dim <= 0 || bwA <= 1 || bwA >= 64 || bwB <= 64 || bwB > 127 ||
      bwB <= bwA || out_low == nullptr || out_high == nullptr) {
    throw std::invalid_argument("invalid wide zero-extension metadata");
  }

  const uint64_t mask_a = (uint64_t{1} << bwA) - 1;
  const __uint128_t modulus_b = __uint128_t{1} << bwB;
  const __uint128_t mask_b = modulus_b - 1;
  const int32_t delta = bwB - bwA;

  vector<uint8_t> wrap(static_cast<size_t>(dim));
  if (msbA != nullptr) {
    aux->MSB_to_Wrap(inA, msbA, wrap.data(), dim, bwA);
  } else {
    aux->wrap_computation(inA, wrap.data(), dim, bwA);
  }
  vector<uint64_t> arithmetic_wrap(static_cast<size_t>(dim));
  aux->B2A(wrap.data(), arithmetic_wrap.data(), dim, delta);

  for (int32_t i = 0; i < dim; ++i) {
    const __uint128_t extended =
        (__uint128_t(inA[i] & mask_a) -
         (__uint128_t{1} << bwA) * arithmetic_wrap[i]) & mask_b;
    out_low[i] = static_cast<uint64_t>(extended);
    out_high[i] = static_cast<uint64_t>(extended >> 64);
  }
}

// prime to ring, use optim. from bumblebee, assume inA is positive
void XTProtocol::z_prime2ring(int32_t dim, uint64_t *inA, uint64_t *outB,
                          uint64_t Q, int32_t bwB) {
  // cout << "log2Q, bwB:" << std::log2(Q) << "," << bwB << endl;
  assert(bwB>=(int32_t)ceil(std::log2(Q)));
  
  uint64_t mask_bwB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
  uint8_t *wrap = new uint8_t[dim];
  uint8_t *msbA = new uint8_t[dim];
  memset(msbA, 0, dim);
  // this->aux->wrap_computation_prime(inA, wrap, dim, bwA, Q);
  this->aux->MSB_to_Wrap(inA, msbA, wrap, dim, bwB);
  for (int i = 0; i < dim; i++) {
    outB[i] = ((inA[i] % Q) - Q * wrap[i]) & mask_bwB;
  }

  delete[] wrap;
}
