/*
Authors: Deevashwer Rathee
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

#include "Millionaire/millionaire.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace sci;
using namespace std;

#define MAX_THREADS 8

int party, port = 32000;
int num_threads = 8;
string address = "127.0.0.1";
bool six_comparison = true;

int dim = 1000000;//1ULL << 16;
int bw_x = 32;
int s_x = 28;

uint8_t *res;
uint64_t *input;
int bitlength = 36;

sci::IOPack *iopackArr[MAX_THREADS];
sci::OTPack *otpackArr[MAX_THREADS];

void relu_thread(int tid, uint8_t *rev, uint64_t *input, int num_ops, uint64_t bitlength) {
  MillionaireProtocol *mill;
  if (tid & 1) {
    mill = new MillionaireProtocol(3 - party, iopackArr[tid], otpackArr[tid]);
  } else {
    mill = new MillionaireProtocol(party, iopackArr[tid], otpackArr[tid]);
  }
  mill->compare(rev, input, num_ops, bitlength);

  delete mill;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of ReLU operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("six", six_comparison, "ReLU6?");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/
  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/
  PRG128 prg;

  uint64_t *input = new uint64_t[dim];
  res = new uint8_t[dim];

  prg.random_data(input, dim * sizeof(uint64_t));

  /************** Fork Threads ****************/
  /********************************************/
  // cheetah里silent OT会做warm up影响统计
  uint64_t total_comm = 0;
  uint64_t thread_comm[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm();
  }

  auto start = clock_start();
  std::thread relu_threads[num_threads];
  int chunk_size = dim / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_ops;
    if (i == (num_threads - 1)) {
      lnum_ops = dim - offset;
    } else {
      lnum_ops = chunk_size;
    }
    relu_threads[i] =
        std::thread(relu_thread, i, res, input, lnum_ops, bitlength);
  }
  for (int i = 0; i < num_threads; ++i) {
    relu_threads[i].join();
  }
  std::cout << "compare success\n";
  long long t = time_from(start);

  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    total_comm += thread_comm[i];
  }

  /************** Verification ****************/
  /********************************************/
//   if (party == ALICE) {
//     iopackArr[0]->io->send_data(x, dim * sizeof(uint64_t));
//     iopackArr[0]->io->send_data(y, dim * sizeof(uint64_t));
//   } else { // party == BOB
//     uint64_t *x0 = new uint64_t[dim];
//     uint64_t *y0 = new uint64_t[dim];
//     iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
//     iopackArr[0]->io->recv_data(y0, dim * sizeof(uint64_t));

//     for (int i = 0; i < dim; i++) {
//       int64_t X = signed_val(x[i] + x0[i], bw_x);
//       int64_t Y = signed_val(y[i] + y0[i], bw_x);
//       int64_t expectedY = X;
//       if (X < 0)
//         expectedY = 0;
//       if (six != 0) {
//         if (X > int64_t(six))
//           expectedY = six;
//       }
//       // cout << X << "\t" << Y << "\t" << expectedY << endl;
//       assert(Y == expectedY);
//     }

//     cout << "ReLU" << (six == 0 ? "" : "6") << " Tests Passed" << endl;

//     delete[] x0;
//     delete[] y0;
//   }

  /**** Process & Write Benchmarking Data *****/
  /********************************************/
  cout << "Number of comp/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "one comp cost:\t" << (t / double(dim)) << std::endl;
  cout << "comp Time\t" << t / (1000.0) << " ms" << endl;
  cout << "comp Bytes Sent\t" << (total_comm) << " byte" << endl;

  /******************* Cleanup ****************/
  /********************************************/
  delete[] res;
  delete[] input;
  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
}
