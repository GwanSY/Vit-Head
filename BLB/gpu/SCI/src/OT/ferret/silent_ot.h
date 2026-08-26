// Author: Zhicong Huang
#ifndef CHEETAH_SILENT_OT_H
#define CHEETAH_SILENT_OT_H

#include <emp-ot/cot.h>
#include <emp-ot/ferret/ferret_cot.h>
#include <math.h>

#include <algorithm>
#include <stdexcept>

#include "OT/ot-utils.h"
#include "OT/ot.h"
#include "utils/mitccrh.h"
#include "utils/performance.h"
#include "OT/split-utils.h"
#include "OT/np.h"
using namespace std;
namespace cheetah {

template <typename IO>
class SilentOT : public sci::OT<SilentOT<IO>> {
 public:
  FerretCOT<IO>* ferret;
  cheetah::MITCCRH<8> mitccrh;
  
  sci::block128 *k0 = nullptr, *k1 = nullptr, *qT = nullptr, *tT = nullptr,
           *tmp = nullptr, block_s;
  sci::PRG128 *G0, *G1;
  bool *s = nullptr, *extended_r = nullptr, setup = false;
  IO *io = nullptr;
  sci::CRH crh;
  // warm up之后会导致执行rcot，产生通信
  SilentOT(int party, int threads, IO** ios, bool malicious = false,
           bool run_setup = true, std::string pre_file = "",
           bool warm_up = true) {
    ferret = new FerretCOT<IO>(party, threads, ios, malicious, run_setup, pre_file);
    if (warm_up) {
      std::cout << "warm up\n";
      block tmp;
      ferret->rcot(&tmp, 1);
    }
  }

  ~SilentOT() { delete ferret; }

  void send_impl(const block* data0, const block* data1, int64_t length) {
    send_ot_cm_cc(data0, data1, length);
  }

  void recv_impl(block* data, const bool* b, int64_t length) {
    recv_ot_cm_cc(data, b, length);
  }

  template <typename T>
  void send_impl(T** data, int length, int l) {
    send_ot_cm_cc(data, length, l);
  }

  template <typename T>
  void recv_impl(T* data, const uint8_t* b, int length, int l) {
    recv_ot_cm_cc(data, b, length, l);
  }

  template <typename T>
  void send_impl(T** data, int length, int N, int l) {
    send_ot_cm_cc(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T* data, const uint8_t* b, int length, int N, int l) {
    recv_ot_cm_cc(data, b, length, N, l);
  }

  void send_cot(uint64_t* data0, const uint64_t* corr, int length, int l) {
    send_ot_cam_cc(data0, corr, length, l);
  }

  void recv_cot(uint64_t* data, const bool* b, int length, int l) {
    recv_ot_cam_cc(data, b, length, l);
  }

  // chosen additive message, chosen choice
  // Sender chooses one message 'corr'. A correlation is defined by the addition
  // function: f(x) = x + corr Sender receives a random message 'x' as output
  // ('data0').

  // generating COT with any corr
  void send_ot_cam_cc(uint64_t* data0, const uint64_t* corr, int64_t length,
                      int l) {
    uint64_t modulo_mask = (1ULL << l) - 1;
    if (l == 64) modulo_mask = -1;
    block* rcm_data = new block[length];
    
    ferret->io->flush();
    send_ot_rcm_cc(rcm_data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    uint32_t y_size = (uint32_t)std::ceil((ot_bsize * l) / (float(64)));
    uint32_t corrected_y_size, corrected_bsize;
    uint64_t y[y_size];
    uint64_t corr_data[ot_bsize];
 
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = rcm_data[j];
        pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
      }

      ferret->mitccrh.template hash<ot_bsize, 2>(pad); //将COT转化为ROT

      for (int j = i; j < i + ot_bsize and j < length; ++j) {
        data0[j] = _mm_extract_epi64(pad[2 * (j - i)], 0) & modulo_mask;
        corr_data[j - i] =
            (corr[j] + data0[j] + _mm_extract_epi64(pad[2 * (j - i) + 1], 0)) &
            modulo_mask;
      }
      corrected_y_size = (uint32_t)std::ceil((std::min(ot_bsize, length - i) * l) /
                                        ((float)sizeof(uint64_t) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_cot_messages(y, corr_data, corrected_y_size, corrected_bsize,
                             l);
      ferret->io->send_data(y, sizeof(uint64_t) * (corrected_y_size));
    }

    delete[] rcm_data;
  }

  // chosen additive message, chosen choice
  // Receiver chooses a choice bit 'b', and
  // receives 'x' if b = 0, and 'x + corr' if b = 1

  // generating COT with any corr
  void recv_ot_cam_cc(uint64_t* data, const bool* b, int64_t length, int l) {
    uint64_t modulo_mask = (1ULL << l) - 1;
    if (l == 64) modulo_mask = -1;

    block* rcm_data = new block[length];
    recv_ot_rcm_cc(rcm_data, b, length);
    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block pad[ot_bsize];

    uint32_t recvd_size = (uint32_t)std::ceil((ot_bsize * l) / (float(64)));
    uint32_t corrected_recvd_size, corrected_bsize;
    uint64_t corr_data[ot_bsize];
    uint64_t recvd[recvd_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size =
          (uint32_t)std::ceil((std::min(ot_bsize, length - i) * l) / (float(64)));
      corrected_bsize = std::min(ot_bsize, length - i);

      memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);

      ferret->io->recv_data(recvd, sizeof(uint64_t) * corrected_recvd_size);

      sci::unpack_cot_messages(corr_data, recvd, corrected_bsize, l);

      for (int j = i; j < i + ot_bsize and j < length; ++j) {
        if (b[j])
          data[j] = (corr_data[j - i] - _mm_extract_epi64(pad[j - i], 0)) &
                    modulo_mask;
        else
          data[j] = _mm_extract_epi64(pad[j - i], 0) & modulo_mask;
      }
    }

    delete[] rcm_data;
  }

  // chosen message, chosen choice

  // data type为block，所以不需要pack_ot_messages
  void send_ot_cm_cc(const block* data0, const block* data1, int64_t length) {
    block* data = new block[length];
    send_ot_rcm_cc(data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data[j];
        pad[2 * (j - i) + 1] = data[j] ^ ferret->Delta;
      }
      // here, ferret depends on the template parameter "IO", making mitccrh
      // also dependent, hence we have to explicitly tell the compiler that
      // "hash" is a template function. See:
      // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class 
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = pad[2 * (j - i)] ^ data0[j];
        pad[2 * (j - i) + 1] = pad[2 * (j - i) + 1] ^ data1[j];
      }
      ferret->io->send_data(pad, 2 * sizeof(block) * std::min(ot_bsize, length - i));
    }
    delete[] data;
  }

  // chosen message, chosen choice
  void recv_ot_cm_cc(block* data, const bool* r, int64_t length) {
    recv_ot_rcm_cc(data, r, length);

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block res[2 * ot_bsize];
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
      ferret->io->recv_data(res, 2 * sizeof(block) * std::min(ot_bsize, length - i));
      for (int64_t j = 0; j < ot_bsize and j < length - i; ++j) {
        data[i + j] = res[2 * j + r[i + j]] ^ pad[j];
      }
    }
  }

  // chosen message, chosen choice.
  // Here, the 2nd dim of data is always 2. We use T** instead of T*[2] or two
  // arguments of T*, in order to be general and compatible with the API of
  // 1-out-of-N OT.
  // 根据l（bit length）将多个要发送的message打包到一个数据类型T中发送
  // 通过pack_ot_messages实现
  template <typename T>
  void send_ot_cm_cc(T** data, int64_t length, int l) {
    block* rcm_data = new block[length];
    send_ot_rcm_cc(rcm_data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    uint32_t y_size =
        (uint32_t)std::ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_y_size, corrected_bsize;
    T y[y_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = rcm_data[j];
        pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
      }
      // here, ferret depends on the template parameter "IO", making mitccrh
      // also dependent, hence we have to explicitly tell the compiler that
      // "hash" is a template function. See:
      // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);

      corrected_y_size = (uint32_t)std::ceil((2 * std::min(ot_bsize, length - i) * l) /
                                        ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_ot_messages<T>((T*)y, data + i, pad, corrected_y_size,
                               corrected_bsize, l, 2); // 最后的2表示的是2选1 OT

      ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
    }
    delete[] rcm_data;
  }

  // chosen message, chosen choice
  // Here, r[i]'s value is always 0 or 1. We use uint8_t instead of bool, in
  // order to be general and compatible with the API of 1-out-of-N OT.
  template <typename T>
  void recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int l) {
    block* rcm_data = new block[length];
    recv_ot_rcm_cc(rcm_data, (const bool*)r, length);

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block pad[ot_bsize];

    uint32_t recvd_size =
        (uint32_t)std::ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_recvd_size, corrected_bsize;
    T recvd[recvd_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size = (uint32_t)std::ceil(
          (2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

      memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);

      sci::unpack_ot_messages<T>(data + i, r + i, (T*)recvd, pad,
                                 corrected_bsize, l, 2);
    }
    delete[] rcm_data;
  }

  // random correlated message, chosen choice
  void send_ot_rcm_cc(block* data0, int64_t length) {
    ferret->send_cot(data0, length);
  }

  // random correlated message, chosen choice
  void recv_ot_rcm_cc(block* data, const bool* b, int64_t length) {
    ferret->recv_cot(data, b, length);
  }

  // random message, chosen choice
  void send_ot_rm_cc(block* data0, block* data1, int64_t length) {
    send_ot_rcm_cc(data0, length);
    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[ot_bsize * 2];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data0[j];
        pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
      }
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        data0[j] = pad[2 * (j - i)];
        data1[j] = pad[2 * (j - i) + 1];
      }
    }
  }

  // random message, chosen choice
  void recv_ot_rm_cc(block* data, const bool* r, int64_t length) {
    recv_ot_rcm_cc(data, r, length); // Ferret生成的选择比特是 特定的
    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
	  std::memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
	  std::memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
    }
  }

  // random message, random choice
  void send_ot_rm_rc(block* data0, block* data1, int64_t length) {
    ferret->rcot(data0, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[ot_bsize * 2];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data0[j];
        pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
      }
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        data0[j] = pad[2 * (j - i)];
        data1[j] = pad[2 * (j - i) + 1];
      }
    }
  }

  // random message, random choice
  void recv_ot_rm_rc(block* data, bool* r, int64_t length) {
    ferret->rcot(data, length); // Ferret生成的选择比特是 随机的
    for (int64_t i = 0; i < length; i++) {
      r[i] = sci::getLSB(data[i]);
    }

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
	  std::memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
	  std::memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
    }
  }

  // random message, random choice
  template <typename T>
  void send_ot_rm_rc(T* data0, T* data1, int64_t length, int l) {
    block* rm_data0 = new block[length];
    block* rm_data1 = new block[length];
    send_ot_rm_rc(rm_data0, rm_data1, length);

    T mask = (T)((1ULL << l) - 1ULL);

    for (int64_t i = 0; i < length; i++) {
      data0[i] = ((T)_mm_extract_epi64(rm_data0[i], 0)) & mask;
      data1[i] = ((T)_mm_extract_epi64(rm_data1[i], 0)) & mask;
    }

    delete[] rm_data0;
    delete[] rm_data1;
  }

  // random message, random choice
  template <typename T>
  void recv_ot_rm_rc(T* data, bool* r, int64_t length, int l) {
    block* rm_data = new block[length];
    recv_ot_rm_rc(rm_data, r, length);

    T mask = (T)((1ULL << l) - 1ULL);

    for (int64_t i = 0; i < length; i++) {
      data[i] = ((T)_mm_extract_epi64(rm_data[i], 0)) & mask;
    }

    delete[] rm_data;
  }

  // chosen message, chosen choice.
  // One-oo-N OT, where each message has l bits. Here, the 2nd dim of data is N.
  template <typename T>
  void send_ot_cm_cc(T** data, int64_t length, int N, int l) {
    int logN = (int)std::ceil(log2(N));

    block* rm_data0 = new block[length * logN];
    block* rm_data1 = new block[length * logN];
    send_ot_rm_cc(rm_data0, rm_data1, length * logN);

    block pad[ot_bsize * N];
    uint32_t y_size =
        (uint32_t)std::ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_y_size, corrected_bsize;
    T y[y_size];

    block* hash_in0 = new block[N - 1];
    block* hash_in1 = new block[N - 1];
    block* hash_out = new block[2 * N - 2];
    int idx = 0;
    for (int x = 0; x < logN; x++) {
      for (int y = 0; y < (1 << x); y++) {
        hash_in0[idx] = makeBlock(y, 0);
        hash_in1[idx] = makeBlock((1 << x) + y, 0);
        idx++;
      }
    }
    // std::cout << "N:" << N << std::endl;
    // std::cout << "logN:" << logN << std::endl;

    for (int64_t i = 0; i < length; i += ot_bsize) {
	  std::memset(pad, 0, sizeof(block) * N * ot_bsize);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        mitccrh.renew_ks(rm_data0 + j * logN, logN);
        mitccrh.hash_exp(hash_out, hash_in0, logN);
        mitccrh.renew_ks(rm_data1 + j * logN, logN);
        mitccrh.hash_exp(hash_out + N - 1, hash_in1, logN);

        for (int64_t k = 0; k < N; k++) {
          idx = 0;
          for (int64_t s = 0; s < logN; s++) {
            int mask = (1 << s) - 1;
            int pref = k & mask;
            if ((k & (1 << s)) == 0)
              pad[(j - i) * N + k] ^= hash_out[idx + pref];
            else
              pad[(j - i) * N + k] ^= hash_out[idx + N - 1 + pref];
            idx += 1 << s;
          }
        }
      }

      corrected_y_size = (uint32_t)std::ceil((std::min(ot_bsize, length - i) * N * l) /
                                        ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_ot_messages<T>((T*)y, data + i, pad, corrected_y_size,
                               corrected_bsize, l, N);

      ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
    }

    delete[] hash_in0;
    delete[] hash_in1;
    delete[] hash_out;
    delete[] rm_data0;
    delete[] rm_data1;
  }

  // chosen message, chosen choice
  // One-oo-N OT, where each message has l bits. Here, r[i]'s value is in [0,
  // N).
  template <typename T>
  void recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int N, int l) {
    int logN = (int)std::ceil(log2(N));

    block* rm_data = new block[length * logN];
    bool* b_choices = new bool[length * logN];
    for (int64_t i = 0; i < length; i++) {
      for (int64_t j = 0; j < logN; j++) {
        b_choices[i * logN + j] = (bool)((r[i] & (1 << j)) >> j);
      }
    }
    recv_ot_rm_cc(rm_data, b_choices, length * logN);

    block pad[ot_bsize];

    uint32_t recvd_size =
        (uint32_t)std::ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_recvd_size, corrected_bsize;
    T recvd[recvd_size];

    block* hash_out = new block[logN];
    block* hash_in = new block[logN];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size = (uint32_t)std::ceil(
          (std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

	  std::memset(pad, 0, sizeof(block) * ot_bsize);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        for (int64_t s = 0; s < logN; s++)
          hash_in[s] = makeBlock(r[j] & ((1 << (s + 1)) - 1), 0);
        mitccrh.renew_ks(rm_data + j * logN, logN);
        mitccrh.hash_single(hash_out, hash_in, logN);

        for (int64_t s = 0; s < logN; s++) {
          pad[j - i] ^= hash_out[s];
        }
      }

      sci::unpack_ot_messages<T>(data + i, r + i, (T*)recvd, pad,
                                 corrected_bsize, l, N);
    }
    delete[] hash_in;
    delete[] hash_out;
    delete[] rm_data;
    delete[] b_choices;
  }

  // void send_batched_got(uint64_t* data, int num_ot, int l,
  //                       int msgs_per_ot = 1) {
  //   throw std::logic_error("Not implemented");
  // }

  // void recv_batched_got(uint64_t* data, const uint8_t* r, int num_ot, int l,
  //                       int msgs_per_ot = 1) {
  //   throw std::logic_error("Not implemented");
  // }

  
  // void send_batched_cot(uint64_t* data0, uint64_t* corr,
  //                       std::vector<int> msg_len, int num_ot,
  //                       int msgs_per_ot = 1) {
  //   throw std::logic_error("Not implemented");

  //   int num_msg_len = msg_len.size();
  //   // The number of OTs of a particular message bitlength
  //   // Simplifying assumption: Each message bitlength has equal number of OTs
  //   int dim = num_ot / num_msg_len;
  //   uint64_t modulo_mask[num_msg_len];
  //   for (int bit_idx = 0; bit_idx < num_msg_len; bit_idx++) {
  //     modulo_mask[bit_idx] =
  //         (msg_len[bit_idx] == 64 ? -1 : ((1ULL << msg_len[bit_idx]) - 1));
  //   }
  //   int max_num_hashes = std::ceil((64 * msgs_per_ot) / 128.0);
  //   int max_pad_len = dim * max_num_hashes;
  //   sci::block128 *pad = new sci::block128[2 * max_pad_len];
  //   sci::block128 *y_per_ot = new sci::block128[msgs_per_ot];
  //   // uint64_t* unpacked_pad0 = new uint64_t[msgs_per_ot];
  //   // uint64_t* unpacked_pad1 = new uint64_t[msgs_per_ot];
  //   // uint64_t* corr_data = new uint64_t[dim*msgs_per_ot];
  //   uint8_t *y =
  //       new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
  //   for (int i = 0; i < num_ot / dim; i++) {
  //     int bit_idx = i;
  //     int lmsg_len = msg_len[bit_idx];
  //     uint64_t lmodulo_mask = modulo_mask[bit_idx];
  //     int num_hashes = std::ceil((lmsg_len * msgs_per_ot) / 128.0);
  //     int bsize =
  //         std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
  //         num_hashes;
  //     for (int j = 0; j < dim; j += (bsize / num_hashes)) {
  //       for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
  //         int ot_idx = i * dim + k;
  //         int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
  //         for (int h = 0; h < num_hashes; h++) {
  //           pad[(k - j) * num_hashes + h] = sci::xorBlocks(qT[ot_idx], sci::toBlock(h));
  //           pad[pad1_offset + (k - j) * num_hashes + h] =
  //               sci::xorBlocks(pad[(k - j) * num_hashes + h], block_s);
  //         }
  //       }
  //       if (bsize <= (dim - j) * num_hashes)
  //         crh.Hn(pad, pad, 2 * bsize);
  //       else
  //         crh.Hn(pad, pad, 2 * (dim - j) * num_hashes);

  //       int lnum_ot = std::min(bsize / num_hashes, dim - j);
  //       int32_t ysize_per_ot = std::ceil(msgs_per_ot * lmsg_len / (8.0));

  //       for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
  //         int ot_idx = i * dim + k;
  //         int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
  //         sci::block128 *pad0_ptr = pad + (k - j) * num_hashes;
  //         sci::block128 *pad1_ptr = pad + pad1_offset + (k - j) * num_hashes;
  //         for (int h = 0; h < msgs_per_ot; h++) {
  //           int msg_idx = ot_idx * msgs_per_ot + h;
  //           int32_t corr_idx = (k - j) * msgs_per_ot + h;
  //           uint64_t unpacked_pad0 = sci::readFromPackedArr(
  //               (uint8_t *)pad0_ptr, num_hashes, h * lmsg_len, lmsg_len);
  //           // unpacked_pad1[h] = readFromPackedArr((uint8_t*)pad1_ptr,
  //           // num_hashes,
  //           //         h*msg_len[bit_idx], msg_len[bit_idx]);
  //           data0[msg_idx] = unpacked_pad0 & lmodulo_mask;
  //           uint64_t corr_data = (corr[msg_idx] + unpacked_pad0) & lmodulo_mask;
  //           sci::writeToPackedArr((uint8_t *)y_per_ot, ysize_per_ot, h * lmsg_len,
  //                            lmsg_len, corr_data);
  //         }
  //         for (int h = 0; h < num_hashes; h++) {
  //           y_per_ot[h] = sci::xorBlocks(y_per_ot[h], pad1_ptr[h]);
  //         }
  //         memcpy(y + (ysize_per_ot * (k - j)), y_per_ot, ysize_per_ot);
  //       }
  //       io->send_data(y, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
  //     }
  //   }
  //   delete[] pad;
  //   // delete[] unpacked_pad0;
  //   // delete[] unpacked_pad1;
  //   // delete[] corr_data;
  //   delete[] y;
  //   delete[] y_per_ot;
  //   delete[] qT;
  // }

  // Batched COT sender with messages of different bitlengths
  // msgs_per_ot specifies the number of OTs with same choice bit (to be
  // batched)
  void send_batched_cot(uint64_t *data0, uint64_t *corr,
                        std::vector<int> msg_len, int num_ot,
                        int msgs_per_ot = 1) {
    // send_pre(num_ot);
    block* rcm_data = new block[num_ot];
    send_ot_rcm_cc(rcm_data, num_ot);

    int num_msg_len = msg_len.size();
    // The number of OTs of a particular message bitlength
    // Simplifying assumption: Each message bitlength has equal number of OTs
    int dim = num_ot / num_msg_len;
    uint64_t modulo_mask[num_msg_len];
    for (int bit_idx = 0; bit_idx < num_msg_len; bit_idx++) {
      modulo_mask[bit_idx] =
          (msg_len[bit_idx] == 64 ? -1 : ((1ULL << msg_len[bit_idx]) - 1));
    }
    int max_num_hashes = std::ceil((64 * msgs_per_ot) / 128.0);
    int max_pad_len = dim * max_num_hashes;
    sci::block128 *pad = new sci::block128[2 * max_pad_len];
    sci::block128 *y_per_ot = new sci::block128[msgs_per_ot];
    // uint64_t* unpacked_pad0 = new uint64_t[msgs_per_ot];
    // uint64_t* unpacked_pad1 = new uint64_t[msgs_per_ot];
    // uint64_t* corr_data = new uint64_t[dim*msgs_per_ot];
    uint8_t *y =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
    for (int i = 0; i < num_ot / dim; i++) {
      int bit_idx = i;
      int lmsg_len = msg_len[bit_idx];
      uint64_t lmodulo_mask = modulo_mask[bit_idx];
      int num_hashes = std::ceil((lmsg_len * msgs_per_ot) / 128.0);
      int bsize =
          std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
          num_hashes;
      for (int j = 0; j < dim; j += (bsize / num_hashes)) {
        for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
          int ot_idx = i * dim + k;
          int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
          for (int h = 0; h < num_hashes; h++) {
            pad[(k - j) * num_hashes + h] = sci::xorBlocks(rcm_data[ot_idx], sci::toBlock(h));
            pad[pad1_offset + (k - j) * num_hashes + h] =
                sci::xorBlocks(pad[(k - j) * num_hashes + h], block_s);
          }
        }
        if (bsize <= (dim - j) * num_hashes)
          crh.Hn(pad, pad, 2 * bsize);
        else
          crh.Hn(pad, pad, 2 * (dim - j) * num_hashes);

        int lnum_ot = std::min(bsize / num_hashes, dim - j);
        int32_t ysize_per_ot = std::ceil(msgs_per_ot * lmsg_len / (8.0));

        for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
          int ot_idx = i * dim + k;
          int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
          sci::block128 *pad0_ptr = pad + (k - j) * num_hashes;
          sci::block128 *pad1_ptr = pad + pad1_offset + (k - j) * num_hashes;
          for (int h = 0; h < msgs_per_ot; h++) {
            int msg_idx = ot_idx * msgs_per_ot + h;
            int32_t corr_idx = (k - j) * msgs_per_ot + h;
            
            uint64_t unpacked_pad0 = sci::readFromPackedArr(
                (uint8_t *)pad0_ptr, num_hashes, h * lmsg_len, lmsg_len);
            // unpacked_pad1[h] = readFromPackedArr((uint8_t*)pad1_ptr,
            // num_hashes,
            //         h*msg_len[bit_idx], msg_len[bit_idx]);
            data0[msg_idx] = unpacked_pad0 & lmodulo_mask;
            uint64_t corr_data = (corr[msg_idx] + unpacked_pad0) & lmodulo_mask;
            sci::writeToPackedArr((uint8_t *)y_per_ot, ysize_per_ot, h * lmsg_len,
                             lmsg_len, corr_data);
           
          }
          for (int h = 0; h < num_hashes; h++) {
            y_per_ot[h] = sci::xorBlocks(y_per_ot[h], pad1_ptr[h]);
          }
          memcpy(y + (ysize_per_ot * (k - j)), y_per_ot, ysize_per_ot);
        }
        ferret->io->send_data(y, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
      }
    }
    delete[] pad;
    // delete[] unpacked_pad0;
    // delete[] unpacked_pad1;
    // delete[] corr_data;
    delete[] y;
    delete[] y_per_ot;
    // delete[] qT;
  }

  // void recv_batched_cot(uint64_t* data, bool* b, std::vector<int> msg_len,
  //                       int num_ot, int msgs_per_ot = 1) {
  //   throw std::logic_error("Not implemented");
  //   int num_msg_len = msg_len.size();
  //   // The number of OTs of a particular message bitlength
  //   // Simplifying assumption: Each message bitlength has equal number of OTs
  //   int dim = num_ot / num_msg_len;
  //   uint64_t modulo_mask[num_msg_len];
  //   for (int bit_idx = 0; bit_idx < num_msg_len; bit_idx++) {
  //     modulo_mask[bit_idx] =
  //         (msg_len[bit_idx] == 64 ? -1 : ((1ULL << msg_len[bit_idx]) - 1));
  //   }
  //   int max_num_hashes = std::ceil((64 * msgs_per_ot) / 128.0);
  //   int max_pad_len = dim * max_num_hashes;
  //   sci::block128 *pad = new sci::block128[max_pad_len];
  //   // uint64_t* unpacked_pad = new uint64_t[msgs_per_ot];
  //   // uint64_t* corr_data = new uint64_t[dim*msgs_per_ot];
  //   uint8_t *y =
  //       new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
  //   for (int i = 0; i < num_ot / dim; i++) {
  //     int bit_idx = i;
  //     int lmsg_len = msg_len[bit_idx];
  //     int num_hashes = std::ceil((lmsg_len * msgs_per_ot) / 128.0);
  //     int bsize =
  //         std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
  //         num_hashes;
  //     for (int j = 0; j < dim; j += (bsize / num_hashes)) {
  //       int lnum_ot = std::min(bsize / num_hashes, dim - j);
  //       int32_t ysize_per_ot = std::ceil(msgs_per_ot * lmsg_len / (8.0));

  //       io->recv_data(y, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
  //       for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
  //         int ot_idx = i * dim + k;
  //         for (int h = 0; h < num_hashes; h++) {
  //           pad[(k - j) * num_hashes + h] = sci::xorBlocks(tT[ot_idx], sci::toBlock(h));
  //         }
  //       }
  //       if (bsize <= (dim - j) * num_hashes)
  //         crh.Hn(pad, pad, bsize);
  //       else
  //         crh.Hn(pad, pad, (dim - j) * num_hashes);
  //       for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
  //         int ot_idx = i * dim + k;
  //         sci::block128 *pad_ptr = pad + (k - j) * num_hashes;
  //         sci::block128 *y_ptr = (sci::block128 *)(y + (ysize_per_ot * (k - j)));
  //         if (b[ot_idx]) {
  //           for (int h = 0; h < num_hashes; h++) {
  //             pad_ptr[h] = sci::xorBlocks(pad_ptr[h], _mm_loadu_si128(y_ptr + h));
  //           }
  //         }
  //         for (int h = 0; h < msgs_per_ot; h++) {
  //           int msg_idx = ot_idx * msgs_per_ot + h;
  //           int corr_idx = (k - j) * msgs_per_ot + h;
  //           uint64_t unpacked_pad = sci::readFromPackedArr(
  //               (uint8_t *)pad_ptr, num_hashes, h * lmsg_len, lmsg_len);
  //           data[msg_idx] = unpacked_pad; // & modulo_mask[bit_idx];
  //         }
  //       }
  //     }
  //   }
  //   delete[] pad;
  //   // delete[] unpacked_pad;
  //   // delete[] corr_data;
  //   delete[] y;
  //   delete[] tT;
  // }

  // Batched COT receiver with messages of different bitlengths
  // msgs_per_ot specifies the number of OTs with same choice bit (to be
  // batched)
  void recv_batched_cot(uint64_t *data, bool *b, std::vector<int> msg_len,
                        int num_ot, int msgs_per_ot = 1) {
    // recv_pre(b, num_ot);
    block* rcm_data = new block[num_ot];
    recv_ot_rcm_cc(rcm_data, (const bool*)b, num_ot);

    int num_msg_len = msg_len.size();
    // The number of OTs of a particular message bitlength
    // Simplifying assumption: Each message bitlength has equal number of OTs
    int dim = num_ot / num_msg_len;
    uint64_t modulo_mask[num_msg_len];
    for (int bit_idx = 0; bit_idx < num_msg_len; bit_idx++) {
      modulo_mask[bit_idx] =
          (msg_len[bit_idx] == 64 ? -1 : ((1ULL << msg_len[bit_idx]) - 1));
    }
    int max_num_hashes = std::ceil((64 * msgs_per_ot) / 128.0);
    int max_pad_len = dim * max_num_hashes;
    sci::block128 *pad = new sci::block128[max_pad_len];
    // uint64_t* unpacked_pad = new uint64_t[msgs_per_ot];
    // uint64_t* corr_data = new uint64_t[dim*msgs_per_ot];
    uint8_t *y =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
    for (int i = 0; i < num_ot / dim; i++) {
      int bit_idx = i;
      int lmsg_len = msg_len[bit_idx];
      int num_hashes = std::ceil((lmsg_len * msgs_per_ot) / 128.0);
      int bsize =
          std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
          num_hashes;
      for (int j = 0; j < dim; j += (bsize / num_hashes)) {
        int lnum_ot = std::min(bsize / num_hashes, dim - j);
        int32_t ysize_per_ot = std::ceil(msgs_per_ot * lmsg_len / (8.0));

        ferret->io->recv_data(y, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
        for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
          int ot_idx = i * dim + k;
          for (int h = 0; h < num_hashes; h++) {
            pad[(k - j) * num_hashes + h] = sci::xorBlocks(rcm_data[ot_idx], sci::toBlock(h));
          }
        }
        if (bsize <= (dim - j) * num_hashes)
          crh.Hn(pad, pad, bsize);
        else
          crh.Hn(pad, pad, (dim - j) * num_hashes);
        for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
          int ot_idx = i * dim + k;
          sci::block128 *pad_ptr = pad + (k - j) * num_hashes;
          sci::block128 *y_ptr = (sci::block128 *)(y + (ysize_per_ot * (k - j)));
          if (b[ot_idx]) {
            for (int h = 0; h < num_hashes; h++) {
              pad_ptr[h] = sci::xorBlocks(pad_ptr[h], _mm_loadu_si128(y_ptr + h));
            }
          }
          for (int h = 0; h < msgs_per_ot; h++) {
            int msg_idx = ot_idx * msgs_per_ot + h;
            int corr_idx = (k - j) * msgs_per_ot + h;
            uint64_t unpacked_pad = sci::readFromPackedArr(
                (uint8_t *)pad_ptr, num_hashes, h * lmsg_len, lmsg_len);
            data[msg_idx] = unpacked_pad; // & modulo_mask[bit_idx];
          }
        }
      }
    }
    delete[] pad;
    // delete[] unpacked_pad;
    // delete[] corr_data;
    delete[] y;
    // delete[] tT;
  }

  // General OT sender with message length > 64
  // msgs_per_ot specifies the number of OTs with same choice bit (to be
  // batched)
  // 有msgs_per_ot个长度为l比特的message，有相同的选择比特
  void send_batched_got(uint64_t *data, int num_ot, int l,
                        int msgs_per_ot = 1) {
    // this->l = l;
    // send_pre(num_ot);
    block* rcm_data = new block[num_ot];
    send_ot_rcm_cc(rcm_data, num_ot);
    

    int dim = num_ot;
    uint64_t modulo_mask = (l == 64 ? -1 : ((1ULL << l) - 1));
    int max_num_hashes = std::ceil((l * msgs_per_ot) / 128.0);
    int max_pad_len = dim * max_num_hashes;
    sci::block128 *pad = new sci::block128[2 * max_pad_len];
    sci::block128 *y0_per_ot = new sci::block128[msgs_per_ot];
    sci::block128 *y1_per_ot = new sci::block128[msgs_per_ot];
    uint8_t *y0 =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
    uint8_t *y1 =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];

    int num_hashes = std::ceil((l * msgs_per_ot) / 128.0); // 反应msgs_per_ot个message需要多少个block
    int bsize = std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
                num_hashes;
    for (int j = 0; j < dim; j += (bsize / num_hashes)) {
      for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
        int ot_idx = k;
        int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
        for (int h = 0; h < num_hashes; h++) {
          pad[(k - j) * num_hashes + h] = sci::xorBlocks(rcm_data[ot_idx], sci::toBlock(h));
          pad[pad1_offset + (k - j) * num_hashes + h] =
              sci::xorBlocks(pad[(k - j) * num_hashes + h], block_s);
        }
      }
      if (bsize <= (dim - j) * num_hashes)
        crh.Hn(pad, pad, 2 * bsize);
      else
        crh.Hn(pad, pad, 2 * (dim - j) * num_hashes);

      int lnum_ot = std::min(bsize / num_hashes, dim - j);
      int32_t ysize_per_ot = std::ceil(msgs_per_ot * l / (8.0));

      for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
        int ot_idx = k;
        int pad1_offset = std::min(bsize / num_hashes, dim - j) * num_hashes;
        sci::block128 *pad0_ptr = pad + (k - j) * num_hashes;
        sci::block128 *pad1_ptr = pad + pad1_offset + (k - j) * num_hashes;
        for (int h = 0; h < msgs_per_ot; h++) {
          int msg_idx = ot_idx * msgs_per_ot + h;
          sci::writeToPackedArr((uint8_t *)y0_per_ot, ysize_per_ot, h * l, l,
                           data[msg_idx * 2]);
          sci::writeToPackedArr((uint8_t *)y1_per_ot, ysize_per_ot, h * l, l,
                           data[msg_idx * 2 + 1]);
        }
        for (int h = 0; h < num_hashes; h++) {
          y0_per_ot[h] = sci::xorBlocks(y0_per_ot[h], pad0_ptr[h]);
          y1_per_ot[h] = sci::xorBlocks(y1_per_ot[h], pad1_ptr[h]);
        }
        memcpy(y0 + (ysize_per_ot * (k - j)), y0_per_ot, ysize_per_ot);
        memcpy(y1 + (ysize_per_ot * (k - j)), y1_per_ot, ysize_per_ot);
      }
      ferret->io->send_data(y0, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
      ferret->io->send_data(y1, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
    }
    delete[] pad;
    // delete[] unpacked_pad0;
    // delete[] unpacked_pad1;
    // delete[] corr_data;
    delete[] y0;
    delete[] y0_per_ot;
    delete[] y1;
    delete[] y1_per_ot;
    // delete[] qT;

  }

  // General OT receiver with message length > 64
  void recv_batched_got(uint64_t *data, const uint8_t *r, int num_ot, int l,
                        int msgs_per_ot = 1) {
    // this->l = l;
    // recv_pre((bool *)r, num_ot);
    block* rcm_data = new block[num_ot];
    recv_ot_rcm_cc(rcm_data, (const bool*)r, num_ot);

    int dim = num_ot;
    uint64_t modulo_mask = (l == 64 ? -1 : ((1ULL << l) - 1));

    int max_num_hashes = std::ceil((l * msgs_per_ot) / 128.0);
    int max_pad_len = dim * max_num_hashes;
    sci::block128 *pad = new sci::block128[max_pad_len];
    uint8_t *y0 =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];
    uint8_t *y1 =
        new uint8_t[dim * msgs_per_ot * (sizeof(uint64_t) / sizeof(uint8_t))];

    int num_hashes = std::ceil((l * msgs_per_ot) / 128.0);
    int bsize = std::min(int(std::ceil(sci::AES_BATCH_SIZE / double(num_hashes))), dim) *
                num_hashes;
    for (int j = 0; j < dim; j += (bsize / num_hashes)) {
      int lnum_ot = std::min(bsize / num_hashes, dim - j);
      int32_t ysize_per_ot = std::ceil(msgs_per_ot * l / (8.0));

      ferret->io->recv_data(y0, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
      ferret->io->recv_data(y1, sizeof(uint8_t) * ysize_per_ot * lnum_ot);
      for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
        int ot_idx = k;
        for (int h = 0; h < num_hashes; h++) {
          pad[(k - j) * num_hashes + h] = sci::xorBlocks(rcm_data[ot_idx], sci::toBlock(h));
        }
      }
      if (bsize <= (dim - j) * num_hashes)
        crh.Hn(pad, pad, bsize);
      else
        crh.Hn(pad, pad, (dim - j) * num_hashes);
      for (int k = j; k < j + (bsize / num_hashes) and k < dim; k++) {
        int ot_idx = k;
        sci::block128 *pad_ptr = pad + (k - j) * num_hashes;
        sci::block128 *y0_ptr = (sci::block128 *)(y0 + (ysize_per_ot * (k - j)));
        sci::block128 *y1_ptr = (sci::block128 *)(y1 + (ysize_per_ot * (k - j)));
        if (r[ot_idx]) {
          for (int h = 0; h < num_hashes; h++) {
            pad_ptr[h] = sci::xorBlocks(pad_ptr[h], _mm_loadu_si128(y1_ptr + h));
          }
        } else {
          for (int h = 0; h < num_hashes; h++) {
            pad_ptr[h] = sci::xorBlocks(pad_ptr[h], _mm_loadu_si128(y0_ptr + h));
          }
        }
        for (int h = 0; h < msgs_per_ot; h++) {
          int msg_idx = ot_idx * msgs_per_ot + h;
          int corr_idx = (k - j) * msgs_per_ot + h;
          data[msg_idx] =
              sci::readFromPackedArr((uint8_t *)pad_ptr, num_hashes, h * l, l);
        }
      }
    }
    delete[] pad;
    delete[] y0;
    delete[] y1;
  }
};

template <typename IO>
class SilentOTN : public sci::OT<SilentOTN<IO>> {
 private:
  const int N;
 public:
  SilentOT<IO>* silent_ot;
  
  int getN() const {
        return N;
    }

  // SilentOTN(SilentOT<IO>* silent_ot, int N) {
  //   this->silent_ot = silent_ot;
  //   this->N = N;
  //   std::cout << "silent OT N:" << N << std::endl;
  // }
   SilentOTN(SilentOT<IO>* ot, int n) 
        : silent_ot(ot), N(n) // 使用初始化列表初始化 const 成员变量
    {
        // std::cout << "silent OT N:" << N << std::endl;
    }

  template <typename T>
  void send_impl(T** data, int length, int l) {
    silent_ot->send_impl(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T* data, const uint8_t* b, int length, int l) {
    silent_ot->recv_impl(data, b, length, N, l);
  }

  
};

}  // namespace cheetah

#endif // CHEETAH_SILENT_OT_H
