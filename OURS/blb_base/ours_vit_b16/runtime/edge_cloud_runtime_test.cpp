#include "edge_cloud_runtime.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace ours_vit;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> static void rejects(Function function) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  require(failed, "invalid lifecycle/contract was accepted");
}

struct Channel {
  std::string digest;
  unsigned char peer_role = 2;
  int sent = 0, received = 0;
  void send_data(const void *, int count) { sent += count; }
  void recv_data(void *out, int count) {
    require(count == 33, "unexpected setup payload");
    for (int i = 0; i < 32; ++i)
      static_cast<unsigned char *>(out)[i] = std::stoi(digest.substr(2*i, 2), nullptr, 16);
    static_cast<unsigned char *>(out)[32] = peer_role;
    received += count;
  }
  void flush() {}
};

struct Backend {
  struct Linear {
    int prepared = 0, released = 0;
    void weights_preprocess_layer(int &, int) { ++prepared; }
    void release_preprocessed_layer(int) { ++released; }
  } linear;
  int configured = 0, normalized = 0, output = 0;
  bool grouped = false, native = false;
  std::vector<HeadProtocol> bound;
  void configure_heterogeneous_attention_layer(int, const std::vector<HeadProtocol> &p,
                                               const std::vector<double> &, Version) {
    ++configured; bound = p;
  }
  int softmax_he(int qk) { ++normalized; return qk + 1; }
  int heterogeneous_attention_norm(int qk, const std::vector<HeadProtocol> &p,
                                      const std::vector<double> &, int tokens, int,
                                      bool group) {
    require(tokens == 197 && p.size() == 12, "shape/heads changed");
    ++normalized; grouped = group; native = true; return qk + 2;
  }
  int heterogeneous_attention_output(int qk, int) { ++output; return qk + 3; }
  int Softmax_v_linear_layer_norm_part1(int qk, int) { ++output; return qk + 4; }
};

static HeteroPlan mixed_plan() {
  std::ostringstream out;
  out << "OURS_VIT_HEAD_PLAN_V1\n"
      << "layers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
      << "layer,physical_head,original_head,protocol,alpha\n";
  for (int layer = 0; layer < 12; ++layer)
    for (int head = 0; head < 12; ++head)
      out << layer << ',' << head << ',' << head << ','
          << (head < 6 ? "softmax_mpc,1" : "he_linear,0.015625") << '\n';
  std::istringstream input(out.str());
  return HeteroPlan::parse(input, "edge-cloud-test");
}

struct SocketChannel {
  int fd;
  void send_data(const void *data, int count) {
    const auto *bytes = static_cast<const char *>(data);
    for (int n = 0; n < count;) {
      const int done = send(fd, bytes + n, count - n, MSG_NOSIGNAL);
      if (done <= 0) throw std::runtime_error("test transport send failed");
      n += done;
    }
  }
  void recv_data(void *data, int count) {
    auto *bytes = static_cast<char *>(data);
    for (int n = 0; n < count;) {
      const int done = recv(fd, bytes + n, count - n, 0);
      if (done <= 0) throw std::runtime_error("test transport receive failed");
      n += done;
    }
  }
  void flush() {}
};

static void two_process_agreement(bool mismatch) {
  int sockets[2];
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair failed");
  timeval timeout{5, 0};
  for (int fd : sockets) {
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  }
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  const bool cloud = child != 0;
  close(sockets[cloud ? 1 : 0]);
  SocketChannel channel{sockets[cloud ? 0 : 1]};
  auto model = EdgeCloudModelSpec{
      mismatch && !cloud ? "modified_vit" : "vit", 197, 256, 768, 64, 1000, 12};
  EdgeCloudRuntime runtime(cloud ? 1 : 2, model, mixed_plan(), Version::HA);
  bool rejected = false;
  try { runtime.agree(channel); } catch (const std::runtime_error &error) {
    rejected = true;
    if (!mismatch) std::cerr << "handshake role=" << runtime.role_name() << ": " << error.what() << '\n';
  }
  close(channel.fd);
  if (!cloud) _exit(rejected == mismatch ? 0 : 1);
  int status = 0;
  const auto waited = waitpid(child, &status, 0);
  if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    std::cerr << "child=" << child << " wait=" << waited << " status=" << status << '\n';
  require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "edge process agreement failed");
  require(rejected == mismatch, "cloud process agreement failed");
}

int main() {
  try {
    auto model = EdgeCloudModelSpec{"test_vit_v1", 197, 256, 768, 64, 1000, 12};
    for (Version version : {Version::B, Version::H, Version::HA, Version::HP, Version::HAP, Version::J, Version::BP}) {
      auto plan = (version == Version::B || version == Version::BP) ? HeteroPlan::all_softmax() : mixed_plan();
      EdgeCloudRuntime cloud(1, model, plan, version), edge(2, model, plan, version);
      require(cloud.contract_digest() == edge.contract_digest(), "role leaked into public contract");
      require(cloud.public_contract().find("conversion=algorithm2_coefficient_extension_v1") != std::string::npos,
              "conversion wire format is not bound into the contract");
      require(cloud.is_cloud() && edge.is_edge(), "role ownership changed");
      Backend cb, eb;
      int weights = 0;
      rejects([&] { cloud.prepare_layer(cb, weights, 0); });
      Channel cc{edge.contract_digest(), 2}, ec{cloud.contract_digest(), 1};
      cloud.agree(cc); edge.agree(ec);
      rejects([&] { cloud.agree(cc); });
      for (int layer = 0; layer < 12; ++layer) {
        for (auto pair : {std::make_pair(&cloud, &cb), std::make_pair(&edge, &eb)}) {
          auto &runtime = *pair.first;
          auto &backend = *pair.second;
          runtime.prepare_layer(backend, weights, layer);
          rejects([&] { runtime.execute_node(layer, "qkv", [] { return 0; }); });
          runtime.execute_node(layer, "ln1", [] { return 0; });
          runtime.execute_node(layer, "qkv", [] { return 0; });
          int qk = 10;

          qk = runtime.attention_norm(backend, qk, layer);
          if (version != Version::B) {
            require(backend.grouped == uses_packing(version), "packing dispatch changed");
            require(backend.native == (version != Version::B), "native HE dispatch changed");
          }
          if (version == Version::HA)
            require(runtime.layer_plan(layer).alphas.back() == 0.015625 &&
                    runtime.layer_plan(layer).protocols.back() == HeadProtocol::HeLinear,
                    "compiled heterogeneous plan/alpha changed");
          const int result = runtime.attention_output(backend, qk, layer);
          require(result == 15,
                  "native attention operator selection changed");
          rejects([&] { runtime.release_layer(backend, layer); });
          for (const char *node : {"ln2", "ffn1", "gelu", "ffn2"})
            runtime.execute_node(layer, node, [] { return 0; });
          runtime.release_layer(backend, layer);
        }
      }
      require(cb.linear.prepared == 12 && cb.linear.released == 12, "cloud cache lifetime changed");
      require(eb.linear.prepared == 0 && eb.linear.released == 0, "edge accessed model weights");
      require(cc.sent == 33 && cc.received == 33 && ec.sent == 33 && ec.received == 33,
              "node dispatch added online transport");
      require(cb.configured == (version != Version::B ? 12 : 0), "head configuration changed");
      require(cb.normalized == 12 && cb.output == 12, "operator count changed");
    }
    auto plan = mixed_plan();
    EdgeCloudRuntime a(1, model, plan, Version::HA);
    model.revision = "modified_model_v2";
    EdgeCloudRuntime b(2, model, plan, Version::HA);
    Channel mismatch{b.contract_digest()};
    rejects([&] { a.agree(mismatch); });
    Channel wrong_role{a.contract_digest(), 1};
    rejects([&] { a.agree(wrong_role); });
    model.block_nodes.insert(model.block_nodes.begin(), "future_model_operator");
    EdgeCloudRuntime future(1, model, plan, Version::HA);
    Channel same{future.contract_digest()};
    future.agree(same);
    Backend backend; int weights = 0;
    future.prepare_layer(backend, weights, 0);
    require(future.execute_node(0, "future_model_operator", [] { return 42; }) == 42,
            "model extension did not use native executor");
    rejects([&] { EdgeCloudRuntime invalid(3, model, plan, Version::HA); });
    two_process_agreement(false);
    two_process_agreement(true);
    std::cout << "OURS_EDGE_CLOUD_TEST status=PASS versions=B,H,HA,HP,HAP,J,BP roles=2 blocks=12 "
                 "model_extension=PASS mismatch_rejection=PASS two_process_handshake=PASS online_messages=0\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
