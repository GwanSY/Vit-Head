#include "execution_observer.h"

#pragma once
// BLB four-head kernel used by the validated optimized attention runtime.
#include "he.h"
#include <functional>
#include <chrono>
#include <iomanip>
using CT=UnifiedCiphertext;
struct MhpKernel {
 HE_CKKS &he;
 uint64_t rotations=0,ctct=0,ctpt=0,rescales=0;
 static constexpr int M=256,H=4,R=16,S=16384;
 MhpKernel(HE_CKKS &h):he(h){}
 double drop(const CT &x){for(auto c=he.context->first_context_data();c;c=c->next_context_data())if(c->parms().coeff_modulus().size()==x.coeff_modulus_size())return double(c->parms().coeff_modulus().back().value());throw std::runtime_error("level not found");}
 CT rot(const CT &x,int n){n%=S;if(!n)return x; CT y(LOCATION::DEVICE);execution_observe::he_call(0,[&](){return he.devaluator->rotate_vector(x,n,*he.dgal_keys,y);});rotations++;return y;}
 void add(CT &a,bool &set,const CT &b){if(!set){a=b;set=true;}else he.devaluator->add_inplace(a,b);}
 CT mask(const CT &x,const std::vector<double> &v,double bits){
  UnifiedPlaintext p(LOCATION::DEVICE);double ps=drop(x)*std::exp2(bits)/x.scale();
  he.dencoder->encode(*he.dcontext,v,ps,p,x.dcipher().chain_index());
  CT y(LOCATION::DEVICE);execution_observe::he_call(2,[&](){return he.devaluator->multiply_plain(x,p,y);});ctpt++;return y;
 }
 void rescale(CT &x,double bits){he.devaluator->rescale_to_next_inplace(x);rescales++;x.scale()=std::exp2(bits);}
 CT rowrot(const CT &x,int n,double bits){
  n=(n%M+M)%M;std::vector<double> v(S,1.0);
  if(!n){CT y=mask(x,v,bits);rescale(y,bits);return y;}
  CT a=rot(x,n),b=rot(x,n-M);
  for(int s=0;s<S;s++)v[s]=(s%M<M-n);
  a=mask(a,v,bits);for(double &z:v)z=1-z;b=mask(b,v,bits);
  he.devaluator->add_inplace(a,b);rescale(a,bits);return a;
 }
 CT preprocess(const CT &x){
  std::vector<CT> babies;for(int n:{0,1,2,3,-256,-255,-254,-253})babies.push_back(rot(x,n));
  CT out(LOCATION::DEVICE);bool have=false;
  for(int g=0;g<4;g++){
   int shift=4*g;CT acc(LOCATION::DEVICE);bool has=false;
   for(int i=0;i<4;i++)for(int wrap=0;wrap<2;wrap++){
    int t=shift+i;if(wrap && t==0)continue;
    std::vector<double> v(S);for(int s=0;s<S;s++){
     int q=(s-shift+S)%S;v[s]=(q/(H*M)==t) && (wrap?q%M>=M-t:q%M<M-t);
    }
    CT term=mask(babies[i+4*wrap],v,50);add(acc,has,term);
   }
   CT term=rot(acc,shift);add(out,have,term);
  }
  rescale(out,50);return out;
 }
 std::vector<CT> run(std::vector<CT> A,std::vector<CT> B,bool preprocessed,double outbits,double factor=1.0){
  int K=A.size();
  if(!preprocessed){for(CT &b:B)b=preprocess(b);for(CT &a:A)he.devaluator->mod_switch_to_next_inplace(a);}
  std::vector<std::vector<CT>> baby(R);
  for(int i=0;i<R;i++)for(int k=0;k<K;k++)baby[i].push_back(rowrot(B[k],i,50));
  std::vector<CT> out(R,CT(LOCATION::DEVICE));std::vector<bool> have(R,false);
  for(int j=0;j<R;j++){
   int shift=j*R;std::vector<CT> ar;for(int k=0;k<K;k++)ar.push_back(rowrot(A[k],-shift,50));
   CT accum[2][2]={{CT(LOCATION::DEVICE),CT(LOCATION::DEVICE)},{CT(LOCATION::DEVICE),CT(LOCATION::DEVICE)}};
   bool set[2][2]={{false,false},{false,false}};
   for(int i=0;i<R;i++){
    CT p(LOCATION::DEVICE);bool hp=false;
    for(int k=0;k<K;k++){CT t(LOCATION::DEVICE);execution_observe::he_call(1,[&](){return he.devaluator->multiply(ar[k],baby[i][k],t);});ctct++;add(p,hp,t);}
    he.devaluator->relinearize_inplace(p,*he.drelin_keys);he.devaluator->rescale_to_next_inplace(p);rescales++;
    p=rot(p,-i*H*M);
    for(int half=0;half<2;half++)for(int wrap=0;wrap<2;wrap++){
     if((half&&i==0)||(wrap&&shift==0))continue;
     std::vector<double> v(S);bool nonzero=false;
     for(int s=0;s<S;s++) {bool yes=(half?s/(H*M)<i:s/(H*M)>=i) && (wrap?s%M<shift:s%M>=shift);v[s]=yes?factor:0;nonzero|=yes;}
     if(!nonzero)continue;
     // Fuse AV final mask and following WO ct-pt before rescaling.
     // Encode the mask at final output scale directly; do not spend a level here.
     double mask_target=preprocessed?outbits-std::log2(drop(p)):outbits;
     CT t=mask(p,v,mask_target);add(accum[half][wrap],set[half][wrap],t);
    }
   }
   for(int half=0;half<2;half++)for(int wrap=0;wrap<2;wrap++)if(set[half][wrap]){
    if(preprocessed)accum[half][wrap].scale()=std::exp2(outbits);
    else rescale(accum[half][wrap],outbits);
    CT t=rot(accum[half][wrap],shift-wrap*M);int dest=(j+half)%R;
    if(!have[dest]){out[dest]=std::move(t);have[dest]=true;}else he.devaluator->add_inplace(out[dest],t);
   }
   std::cout<<"MHP_PROGRESS giant="<<j<<" rotations="<<rotations<<" ctct="<<ctct<<std::endl;
  }
  return out;
 }
};
