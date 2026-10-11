
#pragma once
// Metadata-only observation. No decrypted slots, shares, weights or masks logged.
#include "ckks_bert.h"
#include <chrono>
#include <functional>
#include <mutex>
#include "execution_observer.h"
namespace fusion_observe {
inline thread_local std::string call_path;
inline thread_local std::string public_heads;
struct Context {
 std::string &target, previous;
 Context(std::string &t, std::string value):target(t),previous(t){target=std::move(value);}
 ~Context(){target=previous;}
};
template<class Function> decltype(auto) call(const char *f,int line,const char *expression,Function &&fn){
 Context context(call_path,call_path+(call_path.empty()?"":"|")+f+":"+std::to_string(line)+":"+expression);
 return std::forward<Function>(fn)();
}
inline long long ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline std::mutex &lock(){static std::mutex m;return m;}
struct Scope {
 const char *kind,*function;int line;size_t count;std::function<uint64_t()> bytes;
 long long begin;uint64_t before;std::string detail;
 Scope(const char*k,const char*f,int l,size_t n,std::function<uint64_t()> b):kind(k),function(f),line(l),count(n),bytes(b),begin(ns()),before(bytes()){}
 ~Scope(){auto end=ns();auto sent=bytes()-before;std::lock_guard<std::mutex>guard(lock());
 std::cout<<"FUSION_OBSERVE kind="<<kind<<" function="<<function<<" source_line="<<line<<" count="<<count<<" begin_ns="<<begin<<" end_ns="<<end<<" sent_bytes="<<sent<<" call_path="<<(call_path.empty()?"none":call_path)<<" "<<public_heads<<" "<<detail<<" payload_values_logged=0"<<std::endl;}
};
template<class IO,class CTS>void send(IO*io,CTS&cts,const char*f,int l){Scope s("CT_SEND",f,l,cts.size(),[io](){return io->counter;});send_encrypted_vector(io,cts);}
template<class CTX,class IO,class CTS>void recv(CTX*ctx,IO*io,CTS&cts,const char*f,int l){Scope s("CT_RECV",f,l,cts.size(),[io](){return io->counter;});recv_encrypted_vector(ctx,io,cts);}
template<class HE,class PTS,class CTS>void encrypt_send(HE*he,PTS&pts,CTS&cts,const char*f,int l){Scope s("ENCRYPT_SEND",f,l,pts.size(),[he](){return he->io->counter;});he->encrypt_send(pts,cts);}
}
#define send_encrypted_vector(io,cts) fusion_observe::send(io,cts,__func__,__LINE__)
#define recv_encrypted_vector(ctx,io,cts) fusion_observe::recv(ctx,io,cts,__func__,__LINE__)
