/*
HE instance
- Only support BFV for now
*/
#ifndef HE_H__
#define HE_H__

#include "LinearHE/utils-HE.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include <algorithm>

using namespace sci;
using namespace std;
using namespace seal;

template <typename T>
T EncodeToFxp(double x, int fxp) {
  T u = std::roundf(std::abs(x) * static_cast<double>(1L << fxp));
  if (std::signbit(x)) {
    return -u;
  }
  return u;
}

class HE
{
public:

    int party;
    NetIO *io;

    size_t poly_modulus_degree;
    uint64_t plain_mod;
    // plain_mod / 2
    uint64_t plain_mod_2;

    SEALContext *context;
	Encryptor *encryptor;
	Decryptor *decryptor;
	Evaluator *evaluator;
	BatchEncoder *encoder;
	GaloisKeys *gal_keys;
	RelinKeys *relin_keys;
	Ciphertext *zero;

    HE();
    HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod);
    HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod_bit, bool bit);

    ~HE();
	
};

class HE_CKKS
{
public:
    using scalar_t = uint64_t;	
	using complex_t = std::complex<scalar_t>;
    using complex_double = std::complex<double>;
    // using MPCArith = seal::util::Arithmetic<complex_t, complex_t, scalar_t>;
    using FFTHandler = seal::util::DWTHandler<complex_t, complex_t, scalar_t>;
    int party;
    NetIO *io;
    vector<std::size_t> matrix_reps_index_map_;
	vector<complex_double> root_powers_2n;
  	vector<complex_double> inv_root_powers_2n;
    vector<complex_t> root_powers_2n_scaled;
  	vector<complex_t> inv_root_powers_2n_scaled;
    std::vector<complex_t> encode_mat_;
    std::vector<complex_t> decode_mat_;
	vector<complex_double> root_powers_n;
  	vector<complex_double> inv_root_powers_n;
    
    size_t poly_modulus_degree;
    size_t num_slots;
    // fft_scale是mpc的scale，he_scale是he的scale，he_scale由EVA在MSE<1e-11情况下得到
    int fft_scale;
    int he_scale;

    // MPCArith mpc_arith_;
    FFTHandler fft_handler_;
    // uint64_t plain_mod;
    SEALContext *context;
	Encryptor *encryptor;
	Decryptor *decryptor;
	Evaluator *evaluator;
	CKKSEncoder *encoder;
	GaloisKeys *gal_keys = new GaloisKeys();
	RelinKeys *relin_keys = new RelinKeys();
	Ciphertext *zero;

    HE_CKKS(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> modulus_bits,int fft_scale,int he_scale){
        this->party = party;
        this->poly_modulus_degree = poly_modulus_degree;
        this->io = io;
        this->fft_scale = fft_scale;
        this->he_scale = he_scale;
        // set index map and root map
        int logn = std::log2(poly_modulus_degree);
        this->num_slots = poly_modulus_degree >> 1;
        matrix_reps_index_map_.resize(poly_modulus_degree);
        uint64_t gen = 3;
        uint64_t pos = 1;
        // 此处获取2n次原根
        uint64_t m = static_cast<uint64_t>(poly_modulus_degree) << 1;
        cout << m << endl;
        // #pragma omp parallel for
        for (size_t i = 0; i < num_slots; i++)
        {
            // Position in normal bit order
            uint64_t index1 = (pos - 1) >> 1;
            uint64_t index2 = (m - pos - 1) >> 1;

            // Set the bit-reversed locations
            matrix_reps_index_map_[i] = seal::util::safe_cast<size_t>(seal::util::reverse_bits(index1, logn));
            matrix_reps_index_map_[num_slots | i] = seal::util::safe_cast<size_t>(seal::util::reverse_bits(index2, logn));

            // Next primitive root
            pos *= gen;
            pos &= (m - 1);
        }
        // root_powers_2n.resize(poly_modulus_degree);
        // inv_root_powers_2n.resize(poly_modulus_degree);
        root_powers_2n_scaled.resize(poly_modulus_degree);
        inv_root_powers_2n_scaled.resize(poly_modulus_degree);
        // root_powers_n.resize(poly_modulus_degree);
        // inv_root_powers_n.resize(poly_modulus_degree);

        seal::util::ComplexRoots complex_roots_2n(static_cast<size_t>(m),
                                            seal::MemoryManager::GetPool());
        // seal::util::ComplexRoots complex_roots_n(static_cast<size_t>(poly_modulus_degree),
        //                                     seal::MemoryManager::GetPool());
        #pragma omp parallel for num_threads(64)
        for (size_t i = 1; i < poly_modulus_degree; i++) {
            auto rp_2n = complex_roots_2n.get_root(seal::util::reverse_bits(i, logn));
            // auto rp_n = complex_roots_n.get_root(seal::util::reverse_bits(i, logn));

            // auto rp_2n = complex_roots_2n.get_root(i);
            // auto rp_n = complex_roots_n.get_root(i);

            // root_powers_2n[i].real(rp_2n.real());
            // root_powers_2n[i].imag(rp_2n.imag());
            // root_powers_n[i].real(rp_n.real());
            // root_powers_n[i].imag(rp_n.imag());

            root_powers_2n_scaled[i].real((scalar_t)(rp_2n.real() * (1 << fft_scale)));
            root_powers_2n_scaled[i].imag((scalar_t)(rp_2n.imag() * (1 << fft_scale)));

            auto inv_rp_2n = std::conj(complex_roots_2n.get_root(seal::util::reverse_bits(i - 1, logn) + 1));
            
            // auto inv_rp_n = std::conj(complex_roots_n.get_root(seal::util::reverse_bits(i - 1, logn) + 1));
            // auto inv_rp_2n = std::conj(complex_roots_2n.get_root(i));
            // auto inv_rp_n = std::conj(complex_roots_n.get_root(i));

            // inv_root_powers_2n[i].real(inv_rp_2n.real());
            // inv_root_powers_2n[i].imag(inv_rp_2n.imag());
            inv_root_powers_2n_scaled[i].real((scalar_t)(inv_rp_2n.real()* (1 << fft_scale)));
            inv_root_powers_2n_scaled[i].imag((scalar_t)(inv_rp_2n.imag()* (1 << fft_scale)));
            
            // inv_root_powers_n[i].real(inv_rp_n.real());
            // inv_root_powers_n[i].imag(inv_rp_n.imag());
            // if(i==2){
            //   std::cout << inv_root_powers_[i].real() << std::endl;
            // }
        }
        // decode_mat_.resize(this->num_slots * this->poly_modulus_degree);
        // encode_mat_.resize(this->poly_modulus_degree * this->num_slots);
        // cout << "ok1" << endl;
        // uint64_t gen_pow = 1;
        // #pragma omp parallel for num_threads(64)
        // for (size_t r = 0; r < this->num_slots; r++) {
        //     #pragma omp parallel for num_threads(64)
        //     for (size_t c = 0; c < this->poly_modulus_degree; c++) {
        //         // V[r, c] = w^{5^r * c}
        //         auto v = complex_roots_2n.get_root((gen_pow * c) & (m - 1));
        //         decode_mat_[r * poly_modulus_degree + c].real(EncodeToFxp<scalar_t>(v.real(), fft_scale));
        //         decode_mat_[r * poly_modulus_degree + c].imag(EncodeToFxp<scalar_t>(v.imag(), fft_scale));
        //         // transpose V^T
        //         encode_mat_[c * num_slots + r].real(EncodeToFxp<scalar_t>(v.real(), fft_scale));
        //         encode_mat_[c * num_slots + r].imag(EncodeToFxp<scalar_t>(v.imag(), fft_scale));
        //     }
        //     gen_pow *= gen;
        //     gen_pow &= (m - 1);
        // }
        
        // mpc_arith_ = MPCArith();
        auto mpc_arith_ = seal::util::Arithmetic<complex_t, complex_t, scalar_t>(fft_scale);
        fft_handler_ = seal::util::DWTHandler<complex_t, complex_t, scalar_t>(mpc_arith_);
        // Generate keys
        EncryptionParameters parms(scheme_type::ckks);
        // parms.set_use_special_prime(false);
        auto modulus = seal::CoeffModulus::Create(poly_modulus_degree, modulus_bits);
        
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(modulus);
        context = new SEALContext(parms, true, seal::sec_level_type::tc128);
        this->context = context;
        // cout << " ok3" << endl;
        encoder = new CKKSEncoder(*context);
        evaluator = new Evaluator(*context);
        // cout << "OK4" << endl;
        KeyGenerator keygen(*context);
        SecretKey sec_key = keygen.secret_key();
        PublicKey pub_key;
        keygen.create_public_key(pub_key);
        GaloisKeys gal_keys_;
        // cout << "begin to create galois keys" << endl;
        // vector<int> steps;
        // for(int i=1;i<num_slots;i*=2){
        //     steps.push_back(i);
        // }
        // for(int i=-128;i<=128;i++){
        //     steps.push_back(i);
        // }
        // for(int i=128;i<num_slots;i+=128){
        //     steps.push_back(i);
        // }
        keygen.create_galois_keys(gal_keys_);
        // keygen.create_galois_keys(steps, gal_keys_);
        // cout << "end to create galois keys" << endl;
        RelinKeys relin_keys_;
        keygen.create_relin_keys(relin_keys_);
        gal_keys = new GaloisKeys(gal_keys_);
        relin_keys = new RelinKeys(relin_keys_);
        encryptor = new Encryptor(*context, pub_key);
        decryptor = new Decryptor(*context, sec_key);
        // if (party == BOB) { // BOB, client
        //     // bool visit[num_slots];
        //     // for(int i=0;i<num_slots;i++){
        //     //     visit[i] = false;
        //     // }
        //     // vector<int> steps;
        //     // for(int i=1;i<num_slots;i*=2){
        //     //     steps.push_back(i);
        //     //     visit[i] = true;
        //     // }
        //     // for(int i=-32;i<=32;i++){
        //     //     if(!visit[i]||i==0){
        //     //         steps.push_back(i);
        //     //         visit[i] = true;
        //     //     }
        //     // }
        //     // for(int i=128;i<num_slots/2;i+=128){
        //     //     if(!visit[i]){
        //     //         steps.push_back(i);
        //     //         visit[i] = true;
        //     //     }
        //     // }
        //     // for(int i=128;i<num_slots;i+=128){
        //     //     steps.push_back(i);
        //     // }
            

        //     stringstream os;
        //     pub_key.save(os);
        //     uint64_t pk_size = os.tellp();
        //     gal_keys_.save(os);
        //     uint64_t gk_size = (uint64_t)os.tellp() - pk_size;
        //     cout << "gk_size:" << gk_size << endl;
        //     relin_keys_.save(os);
        //     uint64_t rk_size = (uint64_t)os.tellp() - pk_size - gk_size;

        //     string keys_ser = os.str();
        //     io->send_data(&pk_size, sizeof(uint64_t));
        //     io->send_data(&gk_size, sizeof(uint64_t));
        //     io->send_data(&rk_size, sizeof(uint64_t));
        //     io->send_data(keys_ser.c_str(), pk_size + gk_size + rk_size);

        //     encryptor = new Encryptor(*context, pub_key);
        //     decryptor = new Decryptor(*context, sec_key);
        //     cout << "pk_size + gk_size + rk_size:" << pk_size + gk_size + rk_size << endl;
        //     cout << "BOB > HE instance initialized: " << endl;
        // }
        // else // party == ALICE, server
        // {
        //     uint64_t pk_size;
        //     uint64_t gk_size;
        //     uint64_t rk_size;
        //     io->recv_data(&pk_size, sizeof(uint64_t));
        //     io->recv_data(&gk_size, sizeof(uint64_t));
        //     io->recv_data(&rk_size, sizeof(uint64_t));
            
        //     char *key_share = new char[pk_size + gk_size + rk_size];
        //     io->recv_data(key_share, pk_size + gk_size + rk_size);
        //     stringstream is;
        //     cout << "receive ok" << endl;
        //     PublicKey pub_key;
        //     is.write(key_share, pk_size);
        //     cout << "OK0" << endl;
        //     pub_key.load(*context, is);
        //     cout << "OK1" << endl;
        //     gal_keys = new GaloisKeys();
        //     cout << "OK2" << endl;
        //     is.write(key_share + pk_size, gk_size);
        //     gal_keys->load(*context, is);
        //     relin_keys = new RelinKeys();
        //     is.write(key_share + pk_size + gk_size, rk_size);
        //     relin_keys->load(*context, is);
        //     delete[] key_share;

        //     cout << "key ok" << endl;
        //     encryptor = new Encryptor(*context, pub_key);
        //     cout << "pk_size + gk_size + rk_size:" << pk_size + gk_size + rk_size << endl;
        //     cout << "ALICE > HE instance initialized: " << endl;
        // }
        cout << "> HE instance initialized: " << endl;
        cout << "-> Poly Mod Degree: " << poly_modulus_degree << endl;
        cout << "-> Coeff Mod: " ;
        cout << "fft_scale: " << fft_scale << endl;
        for(auto mod: modulus_bits){
            cout << mod << " ";
        }
        cout << endl;
    }

    vector<scalar_t> ckks_decode(vector<complex_t> x, int degree, bool half=false);

	vector<scalar_t> ckks_encode(vector<scalar_t> x, int slot, bool half=false);

    void ckks_ntt(vector<scalar_t> x, int degree,double scale, Plaintext &destination);

    vector<std::complex<HE_CKKS::scalar_t>> ckks_intt(Plaintext &plain);

	// Complex* negative_cyclic_fft(Complex* x, int size);

	vector<complex_t> fft(vector<complex_t> x, int degree, bool use_matrix_mul);
	vector<complex_t> ifft(vector<complex_t> x, int degree, bool use_matrix_mul);

    // get Q of ct
    uint64_t get_Q(Ciphertext ct){
        const seal::SEALContext &context2 = context->get_context_data(ct.parms_id())->parms();
        auto &context_data = *context2.key_context_data();
        const uint64_t Q = *context_data.total_coeff_modulus();
        // cout << "modulus:" << *(cntxt->)
        return Q;
    }

    // get Q of pt
    uint64_t get_Q(Plaintext pt){
        const seal::SEALContext &context2 = context->get_context_data(pt.parms_id())->parms();
        auto &context_data = *context2.key_context_data();
        const uint64_t Q = *context_data.total_coeff_modulus();
        // cout << "modulus:" << *(cntxt->)
        return Q;
    }

    // get the default Q
    uint64_t get_Q(){
        const seal::SEALContext &context2 = context->get_context_data(context->first_parms_id())->parms();
        auto &context_data = *context2.key_context_data();
        const uint64_t Q = *context_data.total_coeff_modulus();
        // cout << "modulus:" << *(cntxt->)
        return Q;
    }

    inline void print_parameters(const seal::SEALContext &context)
    {
        auto &context_data = *context.key_context_data();
        /*
        Which scheme are we using?
        */
        std::string scheme_name;
        switch (context_data.parms().scheme())
        {
        case seal::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case seal::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        case seal::scheme_type::bgv:
            scheme_name = "BGV";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
        }
        std::cout << "/" << std::endl;
        std::cout << "| Encryption parameters :" << std::endl;
        std::cout << "|   scheme: " << scheme_name << std::endl;
        std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

        /*
        Print the size of the true (product) coefficient modulus.
        */
        std::cout << "|   coeff_modulus size: ";
        std::cout << context_data.total_coeff_modulus_bit_count() << " (";
        auto coeff_modulus = context_data.parms().coeff_modulus();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
        {
            std::cout << coeff_modulus[i].bit_count() << " + ";
        }
        std::cout << coeff_modulus.back().bit_count();
        std::cout << ") bits" << std::endl;

        std::cout << "|   coeff_modulus : ";
        std::cout <<  " (";
        for (std::size_t i = 0; i < coeff_modulus_size; i++)
        {
            //coeff_modulus[i].data()返回的是value的pointer
            std::cout << *coeff_modulus[i].data() << " + ";
        }
        std::cout << ") " << std::endl;
        /*
        For the BFV scheme print the plain_modulus parameter.
        */
        if (context_data.parms().scheme() == seal::scheme_type::bfv)
        {
            std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
        }

        std::cout << "\\" << std::endl;
    }
};
#endif