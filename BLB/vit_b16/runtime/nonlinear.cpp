#include "nonlinear.h"


NonLinear::NonLinear(int party, string address, int port){
    this->party = party;
    this->address = address;
    this->port = port;

    for(int i = 0; i < MAX_THREADS; ++i){
        this->iopackArr[i] = new IOPack(party, port + i, address);
        if (i & 1) {
            this->otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
            this->fpmath[i] = new FPMath(3 - party, iopackArr[i], otpackArr[i]);
        } else {
            this->otpackArr[i] = new OTPack(iopackArr[i], party);
            this->fpmath[i] = new FPMath(party, iopackArr[i], otpackArr[i]);
        }
    }
}

NonLinear::NonLinear(){}

NonLinear::~NonLinear(){
  for (int i = 0; i < MAX_THREADS; i++) {
    // delete this->iopackArr[i];
    // delete this->otpackArr[i];
    // delete this->fpmath[i];
  }
}

void softmax_thread(int tid, int party, uint64_t *x, uint64_t *y, uint64_t* l, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }

  vector<FixArray> output_array;
  FixArray l_short;
  tie(output_array, l_short) = fpmath->softmax_fix(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }

  memcpy(l, l_short.data, num_ops*array_size * sizeof(uint64_t));
}

void NonLinear::softmax(int nthreads, uint64_t* input, uint64_t* output, uint64_t* l, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                &l[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void softmax_irons_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array = fpmath->softmax_fix_iron_1(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::softmax_iron(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_irons_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void layer_norm_thread(int tid, int party, uint64_t *x, uint64_t *y, uint64_t *w, uint64_t *b, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    FixArray input = fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s);
    // input = fpmath->fix->right_shift(input, 4);
    // input = fpmath->fix->extend(input, ell);
    input_array.push_back(input);
  }
  FixArray w_array = fpmath->fix->input(this_party, num_ops*array_size, w, true, ell, s);
  FixArray b_array = fpmath->fix->input(this_party, num_ops*array_size, b, true, ell, s);
  vector<FixArray> output_array = fpmath->layer_norm_fix(input_array, w_array, b_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::layer_norm(int nthreads, uint64_t* input, uint64_t* output, uint64_t* weight, uint64_t* bias, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                layer_norm_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                &weight[offset*array_size], 
                &bias[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_approx_2(input);
  // output = fpmath->fix->extend(output, 64);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_approx(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

// he to ss
void gt_p_sub_thread(int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int ell, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  // if input > p, then sub p
  // sub p/2 anyway
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s_in);
  FixArray p_array = fpmath->fix->input(sci::PUBLIC, num_ops, p, true, ell, s_in);
  FixArray p_2_array = fpmath->fix->input(sci::PUBLIC, num_ops, (p-1)/2, true, ell, s_in);
  FixArray output = fpmath->gt_p_sub(input, p_array);
  output = fpmath->fix->sub(output, p_2_array);

  // FixArray input = fpmath->fix->input(this_party, num_ops, x, false, 29, s_in);
  // FixArray output = fpmath->fix->extend(input, ell);
  // output.signed_ = true;
  // FixArray p_2_array = fpmath->fix->input(sci::PUBLIC, num_ops, (p-1)/2, true, ell, s_in);
  // output = fpmath->fix->sub(output, p_2_array);

  if(s_in > s_out){
    output = fpmath->fix->right_shift(output, s_in - s_out);
  } else if(s_in < s_out){
    output = fpmath->fix->mul(output, 1<<(s_out - s_in));
  }
  
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

// he to ss
void gt_p_sub_part1_thread(int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int ell, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  // if input > p, then sub p
  // sub p/2 anyway
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s_in);
  FixArray p_array = fpmath->fix->input(sci::PUBLIC, num_ops, p, true, ell, s_in);
  FixArray p_2_array = fpmath->fix->input(sci::PUBLIC, num_ops, (p-1)/2, true, ell, s_in);
  FixArray output = fpmath->gt_p_sub(input, p_array);
  output = fpmath->fix->sub(output, p_2_array);

  // FixArray input = fpmath->fix->input(this_party, num_ops, x, false, 29, s_in);
  // FixArray output = fpmath->fix->extend(input, ell);
  // output.signed_ = true;
  // FixArray p_2_array = fpmath->fix->input(sci::PUBLIC, num_ops, (p-1)/2, true, ell, s_in);
  // output = fpmath->fix->sub(output, p_2_array);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

// he to ss
void gt_p_sub_part2_thread(int tid, int party, uint64_t* x, uint64_t *y, int num_ops, int ell, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray output = fpmath->fix->input(this_party, num_ops, x, true, ell, s_in);
  if(s_in > s_out){
    output = fpmath->fix->right_shift(output, s_in - s_out);
  } else if(s_in < s_out){
    output = fpmath->fix->mul(output, 1<<(s_out - s_in));
  }
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void field_to_ring_thread(int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int b_out, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  // if input > p, then sub p
  // sub p/2 anyway
  uint64_t b_in = (uint64_t)ceil(log2(p));
  // cout << "in field_to_ring, b_out, s_in, s_out:" << b_out << "," << s_in << "," << s_out << endl;
  FixArray input = fpmath->fix->input(this_party, num_ops, x, false, b_in, s_in);

  // Convert the field share at its original scale first.  The public BLB
  // source accidentally overwrote s_in with s_out below, so the CKKS path
  // bypassed the l=43/s=13 field-to-ring boundary and fed source-scale
  // coefficients into the FFT.  Keep the signed fixed-point metadata here:
  // after the centered field adapter, the ring value is a two's-complement
  // signed value even though the field input itself is unsigned.
  FixArray output = fpmath->fix->field2ring(input, p, b_in);
  assert(b_in>=b_out);
  if(s_in > s_out){
    output.signed_ = true;
    output = fpmath->fix->truncate_reduce(output, s_in - s_out);
  }
  else if(s_in < s_out){
    output = fpmath->fix->mul(output, 1<<(s_out - s_in));
  }
  output.signed_ = true;
  if(output.ell < b_out){
    output = fpmath->fix->extend(output, b_out);
  }
  else if(output.ell>b_out){
    output = fpmath->fix->reduce(output, b_out);
  }
  // reduce bitwidth
  for(int i=0;i<num_ops;i++){
    output.data[i] = output.data[i] & output.ell_mask();
  }
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}


void ring_to_field_thread(int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int bitwidth, int s_in, int s_out, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }

  
  // bit_in:40, bit_out:64
  // s_in:20, s_out:44

  FixArray input = fpmath->fix->input(this_party, num_ops, x, false, bitwidth, s_in);
  FixArray output;
  // trunc first, then mod transfer
  if(s_in > s_out){
    output = fpmath->fix->right_shift(input, s_in - s_out);
  } else if(s_in < s_out){
    for (int i = 0; i < input.size; i++) {
        input.data[i] = (input.data[i]<<(s_out-s_in));
    }
    input.ell = bitwidth + s_out-s_in;
    input.s = s_out;
    output = input;
  }
  else{
    output = input;
  }
  output.s = s_out;
  uint64_t bit_out = input.ell + 30;
  bit_out = bit_out > 64 ? 64 : bit_out;
  // std::cout <<"bit_in:" << bitwidth << ", bit_out:" << bit_out << std::endl;
  // std::cout << "s_in:" << s_in << ", s_out:" << s_out << std::endl;
  uint8_t* msb_x = new uint8_t[output.size];
  memset(msb_x, 0, output.size);
  if(output.ell >= 64){
    output.ell = 34;
  }
  // cout << "here" << endl;
  if(output.ell < bit_out){
    output = fpmath->fix->extend(output, bit_out, msb_x);
  }
  // cout << "here2" << endl;
  // std::cout << mask << std::endl;
  for(int i=0;i<num_ops;i++){
    output.data[i] = output.data[i] % p;
  }
  /*
    Let sci::ALICE do the adjustments
    h0 = (x0 - 2^k) mod Q
       = (x0 mod Q) + (-2^k mod Q)
    h1 = x1 mod Q
    then h0+h1 = (x0 mod Q) + (-2^k mod Q) + (x1 mod Q) = (x0+x1-2^k) mod Q = x mod Q with probability > 1 - |x|/2^k
  */ 
  if(party == sci::ALICE){
    // bit_out is intentionally allowed to be 64.  Shift in 128-bit space;
    // `1ULL << 64` is undefined and previously corrupted Algorithm 2's
    // public wrap correction on the formal 37-bit ring path.
    __uint128_t neg_2k = (__uint128_t(1) << bit_out);
    if(bit_out>64){
      std::cout << "error: bit_out > 64" << std::endl;
    }
    neg_2k = ((__uint128_t)neg_2k*(__uint128_t)(p-1)) % p;
    // seal::util::multiply_uint_mod(neg_2k, _32, prime);
    // std::cout << "neg_2k:" << neg_2k << std::endl;
    for(int i=0;i<num_ops;i++){
      output.data[i] = (output.data[i] + neg_2k) % p;
    }
  }
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
  delete[] msb_x;
}

static void ring_to_field_wide_thread(
    int tid, int party, uint64_t *x, uint64_t p, uint64_t *y, int num_ops,
    int bitwidth, int s_in, int s_out, int extension_bits, FPMath *fpmath) {
  const int this_party = (tid & 1) ? (3 - party) : party;
  if (bitwidth <= 1 || bitwidth >= 64 || extension_bits <= bitwidth ||
      extension_bits > 127 || s_out < s_in) {
    throw std::invalid_argument("invalid BLB wide ring-to-field parameters");
  }
  const int shift = s_out - s_in;
  const int wide_bits = extension_bits;
  const int high_bits = wide_bits - 64;
  if (high_bits <= 0 || high_bits >= 64 || shift < 0 || p == 0) {
    throw std::invalid_argument("unsupported BLB wide ring-to-field width");
  }
  const __uint128_t modulus_q = static_cast<__uint128_t>(p);

  // All public scale lifts are performed modulo the CKKS prime.  The layer-0
  // Block-2 schedule imports the l=35 input at scale 90, so l+40+85 is wider
  // than __uint128_t even though the field result is only 61 bits.  Reduce
  // before multiplying by 2^shift and compute the power by square-and-
  // multiply; this is the same local ring-to-field operation and avoids a
  // non-protocol integer-width restriction.
  const auto pow_two_mod_q = [modulus_q](int exponent) {
    __uint128_t base = 2 % modulus_q;
    __uint128_t result = 1 % modulus_q;
    int remaining = exponent;
    while (remaining > 0) {
      if (remaining & 1) {
        result = (result * base) % modulus_q;
      }
      remaining >>= 1;
      if (remaining > 0) {
        base = (base * base) % modulus_q;
      }
    }
    return result;
  };
  const __uint128_t scale_factor = pow_two_mod_q(shift);
  const __uint128_t wide_scale_factor = pow_two_mod_q(wide_bits + shift);

  vector<uint64_t> low(static_cast<size_t>(num_ops));
  vector<uint64_t> high(static_cast<size_t>(num_ops));
  // Appendix C.3 uses the signed l-bit ring representation.  Extend the
  // complete batch before the local modulus change; doing an unsigned
  // extension here turns every negative fixed-point value into a large
  // positive field element and corrupts the next CKKS operator.
  fpmath->fix->xt->s_extend_wide(
      num_ops, x, low.data(), high.data(), bitwidth, wide_bits);

  for (int i = 0; i < num_ops; ++i) {
    const __uint128_t wide =
        (static_cast<__uint128_t>(high[i]) << 64) | low[i];

    // Appendix C.3 changes the modulus locally after the wide signed
    // extension.  s_extend_wide already applies the Alice-side sign-origin
    // correction, so the reconstructed wide value is m modulo 2^(l+40); no
    // second modulus correction is allowed here.
    const __uint128_t scaled = ((wide % modulus_q) * scale_factor) % modulus_q;
    // Appendix C.3: the two wide-ring shares are reduced locally to q, but
    // the signed extension contributes one public 2^(l+40) term to the
    // reconstructed sum.  Keep that term on Alice's share with the required
    // subtraction; omitting it adds a large constant to every MPC->CKKS
    // tensor and corrupts the subsequent QKV ciphertexts.
    // The public scale lift is part of the ring value before the local
    // modulus change.  Therefore the wrap term is 2^(l+40+shift), not the
    // unscaled 2^(l+40) term.  The latter was harmless only when s_in==s_out
    // and corrupted every MPC-to-CKKS import with a scale lift.
    const __uint128_t wide_mod_q = wide_scale_factor;
    uint64_t local_share = static_cast<uint64_t>(scaled);
    if (this_party == sci::ALICE) {
      local_share = static_cast<uint64_t>(
          (scaled + modulus_q - wide_mod_q) % modulus_q);
    }
    y[i] = local_share;
  }
  (void)this_party;
}

// The interactive part of ring-to-field is the extend from the working ring
// to a sufficiently wide ring.  The subsequent reduction modulo each public
// CKKS RNS prime is local.  Keep that interactive extension separate so the
// MPC-to-CKKS RNS adapter can reuse it for all primes in one plaintext.
void ring_extend_thread(int tid, int party, uint64_t *x, uint64_t *y,
                        int num_ops, int bitwidth, int s_in, int s_out,
                        int target_bitwidth, FPMath *fpmath) {
  int this_party = (tid & 1) ? (3 - party) : party;
  FixArray input =
      fpmath->fix->input(this_party, num_ops, x, false, bitwidth, s_in);
  FixArray output;
  if (s_in > s_out) {
    output = fpmath->fix->right_shift(input, s_in - s_out);
  } else if (s_in < s_out) {
    const int shift = s_out - s_in;
    for (int i = 0; i < input.size; ++i) {
      input.data[i] <<= shift;
    }
    input.ell = bitwidth + shift;
    input.s = s_out;
    output = input;
  } else {
    output = input;
  }
  output.s = s_out;
  if (output.ell > target_bitwidth) {
    throw std::invalid_argument("ring extension target is narrower than input");
  }

  // This is the same signed-extension input contract used by the existing
  // ring_to_field path.  The all-zero MSB share is intentional: the
  // extension protocol computes the wrap share without reconstructing the
  // coefficient vector.
  std::vector<uint8_t> msb_x(static_cast<size_t>(output.size), 0);
  if (output.ell < target_bitwidth) {
    output = fpmath->fix->extend(output, target_bitwidth, msb_x.data());
  }
  for (int i = 0; i < num_ops; ++i) {
    y[i] = output.data[i];
  }
}

void NonLinear::gt_p_sub(int nthreads, uint64_t* input, uint64_t p, uint64_t* output, int size, int ell, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gt_p_sub_thread, 
                i, 
                party, 
                &input[offset], 
                p,
                &output[offset], 
                lnum_ops,
                ell,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::gt_p_sub_part1(int nthreads, uint64_t* input, uint64_t p, uint64_t* output, int size, int ell, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gt_p_sub_part1_thread, 
                i, 
                party, 
                &input[offset], 
                p,
                &output[offset], 
                lnum_ops,
                ell,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::gt_p_sub_part2(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gt_p_sub_part2_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

// mod p to mod 2^bit_out, p是密文模数，bit_out是输出位宽
void NonLinear::field_to_ring(int nthreads, uint64_t* input, uint64_t p, uint64_t* output, int size, int bit_out, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                field_to_ring_thread, 
                i, 
                party, 
                &input[offset], 
                p,
                &output[offset], 
                lnum_ops,
                bit_out,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

// p是密文模数，bitwidth是输入位宽
// Given x0 + x1 = x mod 2^k
// Compute h0 + h1 = x mod p with probability > 1 - |x|/2^k
void NonLinear::ring_to_field(int nthreads, uint64_t* input, uint64_t p, uint64_t* output, int size, int bitwidth, int s_in, int s_out){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
          lnum_ops = size - offset;
        } else {
          lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                ring_to_field_thread, 
                i, 
                party, 
                &input[offset], 
                p,
                &output[offset], 
                lnum_ops,
                bitwidth,
                s_in, 
                s_out,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::ring_to_field_wide(int nthreads, uint64_t* input,
                                   uint64_t p, uint64_t* output, int size,
                                   int bitwidth, int s_in, int s_out,
                                   int extension_bits) {
  if (nthreads <= 0 || nthreads > MAX_THREADS || size <= 0) {
    throw std::invalid_argument("invalid BLB wide ring-to-field batch");
  }
  std::vector<std::thread> threads(static_cast<size_t>(nthreads));
  const int chunk_size = size / nthreads;
  for (int i = 0; i < nthreads; ++i) {
    const int offset = i * chunk_size;
    const int count = (i == nthreads - 1) ? size - offset : chunk_size;
    if (count == 0) {
      continue;
    }
    threads[i] = std::thread(
        ring_to_field_wide_thread, i, party, input + offset, p, output + offset,
        count, bitwidth, s_in, s_out, extension_bits, this->fpmath[i]);
  }
  for (auto &thread : threads) thread.join();
}

void NonLinear::ring_extend(int nthreads, uint64_t* input, uint64_t* output,
                            int size, int bitwidth, int s_in, int s_out,
                            int target_bitwidth) {
    if (nthreads <= 0 || size <= 0 || bitwidth <= 0 ||
        target_bitwidth < bitwidth || target_bitwidth > 64) {
        throw std::invalid_argument("invalid ring extension dimensions");
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops = (i == nthreads - 1) ? size - offset : chunk_size;
        if (lnum_ops == 0) {
            continue;
        }
        threads.emplace_back(
            ring_extend_thread, i, party, &input[offset], &output[offset],
            lnum_ops, bitwidth, s_in, s_out, target_bitwidth,
            this->fpmath[i]);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

static void ring_extend_wide_thread(int tid, int party, uint64_t *input,
                                    uint64_t *output_low,
                                    uint64_t *output_high, int num_ops,
                                    int bitwidth, int target_bitwidth,
                                    FPMath *fpmath) {
  (void)party;
  fpmath->fix->xt->s_extend_wide(
      num_ops, input, output_low, output_high, bitwidth, target_bitwidth);
}

void NonLinear::ring_extend_wide(int nthreads, uint64_t* input,
                                 uint64_t* output_low,
                                 uint64_t* output_high, int size,
                                 int bitwidth, int target_bitwidth) {
    if (nthreads <= 0 || nthreads > MAX_THREADS || size <= 0 ||
        bitwidth <= 1 || bitwidth >= 64 || target_bitwidth <= bitwidth ||
        target_bitwidth > 127) {
        throw std::invalid_argument("invalid wide ring extension dimensions");
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));
    const int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        const int offset = i * chunk_size;
        const int count = (i == nthreads - 1) ? size - offset : chunk_size;
        if (count == 0) continue;
        threads.emplace_back(
            ring_extend_wide_thread, i, party, input + offset,
            output_low + offset, output_high + offset, count, bitwidth,
            target_bitwidth, this->fpmath[i]);
    }
    for (auto &thread : threads) thread.join();
}

void mul_thread(int tid, int party, uint64_t *x, uint64_t* z, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  BoolArray all_1 = fpmath->bool_op->input(sci::ALICE, num_ops, uint8_t(1));
  BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, num_ops, uint8_t(0));
  FixArray input_x = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray input_y = fpmath->fix->input(this_party, num_ops, z, true, ell, s);
  BoolArray msb_x = fpmath->fix->MSB(input_x);
  BoolArray msb_y = fpmath->fix->MSB(input_y);
  FixArray output = fpmath->fix->mul(input_x, input_y, ell + s, msb_x.data, msb_y.data);
  output = fpmath->fix->truncate_reduce(output, s);
  output = fpmath->fix->extend(output, 64);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void  NonLinear::n_matrix_mul(
  int nthreads, 
  uint64_t* input_1,
  uint64_t* input_2, 
  uint64_t* output, 
  int n, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell, 
  int s){

  // dim3 x dim2
  uint64_t* input_2_trans = new uint64_t[n*dim2*dim3];
  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim2*dim3;
    for(int j = 0; j < dim3; j ++){
      for(int i = 0; i < dim2; i++){
        input_2_trans[j*dim2 + i + matrix_offset] = 
          input_2[i*dim3 + j + matrix_offset];
      }
    }
  }

  
  uint64_t* input_1_dup = new uint64_t[n*dim1*dim2*dim3];
  uint64_t* input_2_dup = new uint64_t[n*dim1*dim2*dim3];
  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim2*dim3;
    for(int i = 0; i < dim1; i++){
      memcpy(&input_2_dup[i*dim2*dim3 + matrix_offset], &input_2_trans[matrix_offset_2], dim2*dim3*sizeof(uint64_t));
    }
  }

  

  for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim1*dim2;
    for(int i = 0; i < dim1; i++){
      for(int j = 0; j < dim3; j++){
        memcpy(&input_1_dup[i*dim2*dim3 + j*dim2 + matrix_offset], &input_1[i*dim2 + matrix_offset_2], dim2*sizeof(uint64_t));
      }
    }
  }

  int size = n*dim1*dim2*dim3;
  // int size = 128;
  uint64_t* output_tmp = new uint64_t[n*dim1*dim2*dim3];
  std::thread threads[nthreads];

  int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                mul_thread, 
                i, 
                party, 
                &input_1_dup[offset], 
                &input_2_dup[offset], 
                &output_tmp[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }

    // print_ss(&input_2_dup[0], 8, ell, s);
    // print_ss(&input_1_dup[0], 8, ell, s);
    // print_ss(&output_tmp[0], 8, ell, s);
    // return;

    for(int nm = 0; nm < n; nm++){
    int matrix_offset = nm*dim1*dim2*dim3;
    int matrix_offset_2 = nm*dim1*dim3;
      for(int i = 0; i < dim1; i++){
        for(int k = 0; k < dim3; k++){
          output[i*dim3 + k + matrix_offset_2] = 0;
          for(int j = 0; j < dim2; j++){
            output[i*dim3 + k + matrix_offset_2] += output_tmp[i*dim3*dim2 + k*dim2 + j + matrix_offset];
          }
        }
      }
    }

    delete[] input_2_trans;
    delete[] input_1_dup;
    delete[] input_2_dup;
    delete[] output_tmp;
}

void matmul_thread(
  int tid, 
  int party, 
  uint64_t *a, 
  uint64_t* b, 
  uint64_t *c, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell_in_1, 
  int ell_in_2, 
  int ell_out, 
  int s_in_1,
  int s_in_2,
  int s_out, 
  FPMath *fpmath) {
  
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  int extra_scale = s_in_1 + s_in_2 - s_out;
  // auto t_pc = high_resolution_clock::now();
  // cout << "> [TIMING]: mul takes:" << interval_2(t_pc) << " sec" << endl; 
  BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, dim1*dim3, uint8_t(0));
  BoolArray all_1 = fpmath->bool_op->input(sci::ALICE, dim1*dim3, uint8_t(1));
  fpmath->fix->mult->matrix_multiplication(
      dim1,
      dim2,
      dim3,
      a,
      b,
      c,
      ell_in_1,
      ell_in_2,
      ell_out + extra_scale,
      true,
      true,
      true,
      MultMode::None, 
      all_0.data,
      nullptr
    );
    
    FixArray ret = fpmath->fix->input(this_party, dim1*dim3, c, true, ell_out+extra_scale, s_in_1 + s_in_2);
    if(extra_scale > 0){
      ret = fpmath->fix->truncate_reduce(ret, extra_scale);
    }
    // ret = fpmath->fix->extend(ret, 64);
    memcpy(c, ret.data, (dim1*dim3)*sizeof(uint64_t));
}

void  NonLinear::n_matrix_mul_iron(
  int nthreads, 
  uint64_t* input_1,
  uint64_t* input_2, 
  uint64_t* output, 
  int n, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell_in_1, 
  int ell_in_2, 
  int ell_out, 
  int s_in_1,
  int s_in_2,
  int s_out){

  std::thread threads[n];

  for (int i = 0; i < n; ++i) {
      threads[i] =
          std::thread(
              matmul_thread, 
              i, 
              party, 
              &input_1[i*dim1*dim2],
              &input_2[i*dim2*dim3],
              &output[i*dim1*dim3],
              dim1,
              dim2,
              dim3,
              ell_in_1,
              ell_in_2,
              ell_out,
              s_in_1,
              s_in_2,
              s_out,
              this->fpmath[i]);
  }
  for (int i = 0; i < n; ++i) {
      threads[i].join();
  }
}

void p_matmul_thread(
  int tid, 
  int party, 
  uint64_t *a, 
  uint64_t *b, 
  uint64_t *c, 
  int shard, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell_in_1, 
  int ell_in_2, 
  int ell_out, 
  int s_in_1,
  int s_in_2,
  int s_out, 
  FPMath *fpmath) {
  
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  int extra_scale = s_in_1 + s_in_2 - s_out;
  int dim3_shard = dim3 / shard;
  uint64_t* b_shard = new uint64_t[dim2*dim3_shard];
  uint64_t* c_shard = new uint64_t[dim1*dim3_shard];

  for(int i = 0; i < dim2; i++){
    int offset = i*dim3 + tid*dim3_shard;
    memcpy(&b_shard[i*dim3_shard], &b[offset], dim3_shard*sizeof(uint64_t));
  }

  BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, dim1*dim3, uint8_t(0));
  BoolArray all_1 = fpmath->bool_op->input(sci::ALICE, dim1*dim3, uint8_t(1));
  fpmath->fix->mult->matrix_multiplication(
      dim1,
      dim2,
      dim3_shard,
      a,
      b_shard,
      c_shard,
      ell_in_1,
      ell_in_2,
      ell_out + extra_scale,
      true,
      true,
      true,
      MultMode::None
    );
    
    FixArray ret = fpmath->fix->input(this_party, dim1*dim3_shard, c_shard, true, ell_out+extra_scale, s_in_1 + s_in_2);
    if(extra_scale > 0){
      ret = fpmath->fix->truncate_reduce(ret, extra_scale);
    }

    for(int i = 0; i < dim1; i++){
      int offset = i*dim3 + tid*dim3_shard;
      memcpy(&c[offset], &c_shard[i*dim3_shard], dim3_shard*sizeof(uint64_t));
    }

    delete[] b_shard;
    delete[] c_shard;
}


void  NonLinear::p_matrix_mul_iron(
  int nthreads, 
  uint64_t* input_1,
  uint64_t* input_2, 
  uint64_t* output, 
  int dim1, 
  int dim2, 
  int dim3, 
  int ell_in_1, 
  int ell_in_2, 
  int ell_out, 
  int s_in_1,
  int s_in_2,
  int s_out){

  assert(dim3 / nthreads != 0);

  std::thread threads[nthreads];

  for (int i = 0; i < nthreads; ++i) {
      threads[i] =
          std::thread(
              p_matmul_thread, 
              i, 
              party, 
              input_1,
              input_2,
              output,
              nthreads,
              dim1,
              dim2,
              dim3,
              ell_in_1,
              ell_in_2,
              ell_out,
              s_in_1,
              s_in_2,
              s_out,
              this->fpmath[i]);
  }
  for (int i = 0; i < nthreads; ++i) {
      threads[i].join();
  }
}

void right_shift_thread(int tid, int party, uint64_t *x, int a, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath, bool msb=false) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output;
  if(msb){
    uint8_t* msb_x = new uint8_t[num_ops];
    output = fpmath->fix->right_shift(input, a, msb_x);
  }
  else{
    output = fpmath->fix->right_shift(input, a, nullptr);
  }
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::right_shift(int nthreads, uint64_t* input, int a, uint64_t* output, int size, int ell, int s, bool msb){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                right_shift_thread, 
                i, 
                party, 
                &input[offset], 
                a,
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i], msb);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void reduce_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell_in, int ell_out, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell_in, s);
  FixArray output = fpmath->fix->reduce(input, ell_out);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}


void NonLinear::reduce(int nthreads, uint64_t* input, uint64_t* output, int size, int ell_in, int ell_out, int s){
  std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                reduce_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell_in,
                ell_out,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::print_ss(uint64_t* input, int length, int ell, int s){
  FixArray tmp = fpmath[0]->fix->input(party, length, input, true, ell, s);
  fpmath[0]->print(tmp);
}

FixArray NonLinear::to_public(uint64_t* input, int length, int ell, int s){
  FixArray tmp = fpmath[0]->fix->input(party, length, input, true, ell, s);
  tmp = fpmath[0]->fix->extend(tmp, 64);
  tmp = fpmath[0]->fix->output(sci::PUBLIC, tmp);
  return tmp;
}

void cancel_wrap_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, num_ops, uint8_t(0));
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  // FixArray input_2 = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  // for(int i =0; i < num_ops; i++){
  //   input_2.data[i] -= 1 << (ell - 1);
  //   input_2.data[i] &= input_2.ell_mask();
  // }
  // BoolArray wrap, zero_test;
  // tie(wrap, zero_test) = fpmath->fix->wrap_and_zero_test(input);
  FixArray output = fpmath->fix->extend(input, 37);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}


void NonLinear::cancel_wrap(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
  std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                cancel_wrap_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void NonLinear::pruning(
  uint64_t* l, 
  int packing_num,
  int l_dim,
  int l_array_size, 
  int l_ell, 
  int l_s, 
  uint64_t* softmax_v,
  int sv_ell, 
  int sv_s, 
  uint64_t* h1,
  int h1_ell, 
  int h1_s, 
  int input_dim,
  int common_dim,
  uint64_t* softmax_v_pruned,
  uint64_t* h1_pruned){

  assert(l_dim == input_dim);

  FPMath *fpmath_ = this->fpmath[0];


  FixArray sum_l = fpmath_->fix->input(party, l_dim, (uint64_t)0 ,false, l_ell, l_s);
  FixArray softmax_v_fix = fpmath_->fix->input(party, input_dim*common_dim, softmax_v, true, sv_ell, sv_s);
  FixArray h1_fix= fpmath_->fix->input(party, input_dim*common_dim, h1, true, h1_ell, h1_s);

  for(int i = 0; i < packing_num; i++){
    vector<FixArray> l_fix;
    for(int j = 0; j < l_dim; j++){
      int offset = i*l_dim*l_array_size + j*l_array_size;
      l_fix.push_back(fpmath_->fix->input(party, l_array_size, &l[offset], false, l_ell, l_s));
    }
    sum_l = fpmath_->fix->add(sum_l, fpmath_->fix->tree_sum(l_fix));
  }

  // print_ss(sum_l.data, 128, 25, 0);

  FixArray sorted_sum_l;

  // FixArray sorted_tmp;
  // FixArray tmp(party, 128, false, 16, 0);
  // for(int i =0; i < 128; i++){
  //   tmp.data[i] = 128 - i;
  // }

  // tie(sorted_tmp, std::ignore, std::ignore) = 
  //   fpmath_->bitonic_sort_and_swap(tmp, FixArray(), FixArray(), false);

  // print_ss(tmp.data, 8, 16, 0);
  // print_ss(sorted_tmp.data, 8, 16, 0);
  // return;

  // A HACK: fix later

  sum_l = fpmath_->fix->extend(sum_l, l_ell + 1);
  sum_l.signed_ = true;

  tie(sorted_sum_l, std::ignore, std::ignore) = 
    fpmath_->bitonic_sort_and_swap(sum_l, FixArray(), FixArray(), false);

  int median_pos = (l_array_size/2) - 1;

  FixArray median = fpmath_->fix->input(
    party, 
    sorted_sum_l.size, 
    sorted_sum_l.data[median_pos],
    sorted_sum_l.signed_, 
    sorted_sum_l.ell, 
    sorted_sum_l.s
  );

  FixArray all_0 = fpmath_->fix->input(
    party, 
    sorted_sum_l.size, 
    (uint64_t)0,
    sorted_sum_l.signed_, 
    sorted_sum_l.ell, 
    sorted_sum_l.s
  );

  FixArray all_128 = fpmath_->fix->input(
    party, 
    sorted_sum_l.size, 
    (uint64_t)64,
    sorted_sum_l.signed_, 
    sorted_sum_l.ell, 
    sorted_sum_l.s
  );

  FixArray indices = fpmath_->fix->input(
    party, 
    sorted_sum_l.size, 
    (uint64_t)0,
    sorted_sum_l.signed_, 
    sorted_sum_l.ell, 
    sorted_sum_l.s
  );

  if (party == sci::ALICE){
    for(int i = 0; i < sorted_sum_l.size; i++){
      indices.data[i] = i;
    }
  }

  BoolArray cmp = fpmath_->fix->GT(sum_l, median);
  FixArray indice_plus_scores = fpmath_->fix->if_else(cmp, all_128, all_0);
  indice_plus_scores = fpmath_->fix->add(indice_plus_scores, indices);

  FixArray res_softmax_v;
  FixArray res_h1;

  tie(std::ignore, res_softmax_v, res_h1) = 
    fpmath_->bitonic_sort_and_swap(indice_plus_scores, softmax_v_fix, h1_fix, true);

  memcpy(softmax_v_pruned, res_softmax_v.data, res_softmax_v.size*sizeof(uint64_t) / 2);
  memcpy(h1_pruned, res_h1.data, res_h1.size*sizeof(uint64_t) / 2);
}

void convert_l_to_p_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, int l, uint64_t p, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  BoolArray wrap = fpmath->fix->wrap(input);
  FixArray output = fpmath->fix->if_else(wrap, input, input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::convert_l_to_p(int nthreads, uint64_t* input, uint64_t* output, int l, uint64_t p, int size, int ell, int s){
  std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                convert_l_to_p_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                l,
                p,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}
