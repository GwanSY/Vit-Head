#include "he.h"
#include "FloatingPoint/fp-math.h"

#include <cstring>
#include <limits>

namespace {

using SecureComplex = HE_CKKS::complex_t;
using SecureScalar = HE_CKKS::scalar_t;

struct SecureFftCounters {
    std::uint64_t stages = 0;
    std::uint64_t extend_calls = 0;
    std::uint64_t truncate_calls = 0;
    std::uint64_t public_complex_multiplications = 0;
    std::uint64_t packed_scalar_components = 0;
    int product_ell = 0;
};

std::uint64_t secure_ring_mask(int ell) {
    if (ell <= 0 || ell > 64) {
        throw std::invalid_argument("secure CKKS FFT ring width is invalid");
    }
    return ell == 64 ? ~std::uint64_t{0}
                     : ((std::uint64_t{1} << ell) - 1);
}

std::int64_t public_signed_word(std::uint64_t word) {
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(word),
                  "unexpected fixed-width integer size");
    std::memcpy(&result, &word, sizeof(word));
    return result;
}

std::uint64_t public_unsigned_word(std::int64_t word) {
    std::uint64_t result = 0;
    std::memcpy(&result, &word, sizeof(word));
    return result;
}

std::int64_t public_arithmetic_shift(__int128 value, int shift) {
    if (shift <= 0 || shift >= 63) {
        throw std::invalid_argument("public CKKS root shift is invalid");
    }
    const __int128 divisor = __int128{1} << shift;
    __int128 shifted = 0;
    if (value >= 0) {
        shifted = value / divisor;
    } else {
        // Match a two's-complement arithmetic right shift: floor(value/2^s).
        shifted = -((-value + divisor - 1) / divisor);
    }
    if (shifted < std::numeric_limits<std::int64_t>::min() ||
        shifted > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("scaled public CKKS root exceeds int64");
    }
    return static_cast<std::int64_t>(shifted);
}

SecureComplex multiply_public_root_scalar(
    const SecureComplex &root, std::uint64_t scalar, int root_scale) {
    const std::int64_t root_real = public_signed_word(root.real());
    const std::int64_t root_imag = public_signed_word(root.imag());
    const std::int64_t scalar_signed = public_signed_word(scalar);
    const std::int64_t result_real = public_arithmetic_shift(
        static_cast<__int128>(root_real) * scalar_signed, root_scale);
    const std::int64_t result_imag = public_arithmetic_shift(
        static_cast<__int128>(root_imag) * scalar_signed, root_scale);
    return SecureComplex(public_unsigned_word(result_real),
                         public_unsigned_word(result_imag));
}

SecureComplex ring_add(const SecureComplex &lhs, const SecureComplex &rhs,
                       std::uint64_t mask) {
    return SecureComplex((lhs.real() + rhs.real()) & mask,
                         (lhs.imag() + rhs.imag()) & mask);
}

SecureComplex ring_sub(const SecureComplex &lhs, const SecureComplex &rhs,
                       std::uint64_t mask) {
    return SecureComplex((lhs.real() - rhs.real()) & mask,
                         (lhs.imag() - rhs.imag()) & mask);
}

// Multiply a complete FFT stage by its public fixed-point complex roots. Real
// and imaginary components are packed into one MPC call so each stage uses a
// single sign extension and a single exact truncate-reduce.
std::vector<SecureComplex> secure_multiply_public_roots(
    const std::vector<SecureComplex> &input,
    const std::vector<SecureComplex> &roots, int ell, int value_scale,
    int root_scale, FixOp *fix, SecureFftCounters &counters) {
    if (fix == nullptr || input.empty() || input.size() != roots.size()) {
        throw std::invalid_argument("secure CKKS FFT root batch is invalid");
    }
    // Two signed products contribute to each real/imag component, hence two
    // guard bits beyond ell+root_scale. Keep SCI fixed-point arithmetic below
    // the signed 64-bit boundary and fail closed otherwise.
    const int product_ell = ell + root_scale + 2;
    if (ell <= 1 || root_scale <= 0 || product_ell > 63) {
        throw std::invalid_argument(
            "secure CKKS FFT product width exceeds 63 bits");
    }
    const std::uint64_t input_mask = secure_ring_mask(ell);
    const std::uint64_t product_mask = secure_ring_mask(product_ell);
    const std::size_t component_count = input.size() * 2;
    if (component_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("secure CKKS FFT batch is too large");
    }

    FixArray packed(fix->party, static_cast<int>(component_count), true, ell,
                    value_scale);
    for (std::size_t index = 0; index < input.size(); ++index) {
        packed.data[2 * index] = input[index].real() & input_mask;
        packed.data[2 * index + 1] = input[index].imag() & input_mask;
    }
    FixArray extended = fix->extend(packed, product_ell);
    ++counters.extend_calls;

    FixArray products(fix->party, static_cast<int>(component_count), true,
                      product_ell, value_scale + root_scale);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::uint64_t value_real = extended.data[2 * index];
        const std::uint64_t value_imag = extended.data[2 * index + 1];
        const std::uint64_t root_real = roots[index].real() & product_mask;
        const std::uint64_t root_imag = roots[index].imag() & product_mask;
        products.data[2 * index] =
            (value_real * root_real - value_imag * root_imag) & product_mask;
        products.data[2 * index + 1] =
            (value_real * root_imag + value_imag * root_real) & product_mask;
    }

    FixArray truncated = fix->truncate_reduce(products, root_scale);
    ++counters.truncate_calls;
    if (truncated.ell < ell || truncated.s != value_scale) {
        throw std::runtime_error("secure CKKS FFT truncation metadata mismatch");
    }
    if (truncated.ell > ell) {
        truncated = fix->reduce(truncated, ell);
    }

    std::vector<SecureComplex> output(input.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = SecureComplex(truncated.data[2 * index] & input_mask,
                                      truncated.data[2 * index + 1] &
                                          input_mask);
    }
    counters.public_complex_multiplications += input.size();
    counters.packed_scalar_components += component_count;
    counters.product_ell = product_ell;
    return output;
}

std::vector<SecureComplex> secure_transform_from_rev(
    std::vector<SecureComplex> values, int ell, int value_scale,
    int root_scale, const std::vector<SecureComplex> &roots,
    std::uint64_t inverse_degree, FixOp *fix, SecureFftCounters &counters) {
    const std::size_t n = values.size();
    if (n < 2 || (n & (n - 1)) != 0 || roots.size() < n) {
        throw std::invalid_argument("secure CKKS inverse FFT shape is invalid");
    }
    const std::uint64_t value_mask = secure_ring_mask(ell);
    std::size_t root_index = 0;
    std::size_t gap = 1;
    std::size_t m = n >> 1;

    for (; m > 1; m >>= 1) {
        std::vector<SecureComplex> differences;
        std::vector<SecureComplex> stage_roots;
        std::vector<std::size_t> destinations;
        differences.reserve(n >> 1);
        stage_roots.reserve(n >> 1);
        destinations.reserve(n >> 1);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < m; ++group) {
            if (++root_index >= roots.size()) {
                throw std::out_of_range("secure CKKS inverse root index");
            }
            const SecureComplex root = roots[root_index];
            for (std::size_t index = 0; index < gap; ++index) {
                const std::size_t x_index = offset + index;
                const std::size_t y_index = x_index + gap;
                const SecureComplex lhs = values[x_index];
                const SecureComplex rhs = values[y_index];
                values[x_index] = ring_add(lhs, rhs, value_mask);
                differences.push_back(ring_sub(lhs, rhs, value_mask));
                stage_roots.push_back(root);
                destinations.push_back(y_index);
            }
            offset += gap << 1;
        }
        const auto products = secure_multiply_public_roots(
            differences, stage_roots, ell, value_scale, root_scale, fix,
            counters);
        for (std::size_t index = 0; index < products.size(); ++index) {
            values[destinations[index]] = products[index];
        }
        ++counters.stages;
        gap <<= 1;
    }

    if (++root_index >= roots.size()) {
        throw std::out_of_range("secure CKKS inverse final root index");
    }
    const SecureComplex root = roots[root_index];
    const SecureComplex scaled_root =
        multiply_public_root_scalar(root, inverse_degree, root_scale);
    const SecureComplex inverse_scalar(inverse_degree, 0);
    std::vector<SecureComplex> final_inputs;
    std::vector<SecureComplex> final_roots;
    std::vector<std::size_t> final_destinations;
    final_inputs.reserve(n);
    final_roots.reserve(n);
    final_destinations.reserve(n);
    for (std::size_t index = 0; index < gap; ++index) {
        const SecureComplex lhs = values[index];
        const SecureComplex rhs = values[index + gap];
        final_inputs.push_back(ring_add(lhs, rhs, value_mask));
        final_roots.push_back(inverse_scalar);
        final_destinations.push_back(index);
        final_inputs.push_back(ring_sub(lhs, rhs, value_mask));
        final_roots.push_back(scaled_root);
        final_destinations.push_back(index + gap);
    }
    const auto final_products = secure_multiply_public_roots(
        final_inputs, final_roots, ell, value_scale, root_scale, fix,
        counters);
    for (std::size_t index = 0; index < final_products.size(); ++index) {
        values[final_destinations[index]] = final_products[index];
    }
    ++counters.stages;
    return values;
}

std::vector<SecureComplex> secure_transform_to_rev(
    std::vector<SecureComplex> values, int ell, int value_scale,
    int root_scale, const std::vector<SecureComplex> &roots, FixOp *fix,
    SecureFftCounters &counters) {
    const std::size_t n = values.size();
    if (n < 2 || (n & (n - 1)) != 0 || roots.size() < n) {
        throw std::invalid_argument("secure CKKS forward FFT shape is invalid");
    }
    const std::uint64_t value_mask = secure_ring_mask(ell);
    std::size_t root_index = 0;
    std::size_t gap = n >> 1;
    std::size_t m = 1;
    for (;;) {
        std::vector<SecureComplex> multiplicands;
        std::vector<SecureComplex> stage_roots;
        std::vector<std::size_t> x_destinations;
        std::vector<std::size_t> y_destinations;
        std::vector<SecureComplex> lhs_values;
        multiplicands.reserve(n >> 1);
        stage_roots.reserve(n >> 1);
        x_destinations.reserve(n >> 1);
        y_destinations.reserve(n >> 1);
        lhs_values.reserve(n >> 1);
        std::size_t offset = 0;
        for (std::size_t group = 0; group < m; ++group) {
            if (++root_index >= roots.size()) {
                throw std::out_of_range("secure CKKS forward root index");
            }
            const SecureComplex root = roots[root_index];
            for (std::size_t index = 0; index < gap; ++index) {
                const std::size_t x_index = offset + index;
                const std::size_t y_index = x_index + gap;
                lhs_values.push_back(values[x_index]);
                multiplicands.push_back(values[y_index]);
                stage_roots.push_back(root);
                x_destinations.push_back(x_index);
                y_destinations.push_back(y_index);
            }
            offset += gap << 1;
        }
        const auto products = secure_multiply_public_roots(
            multiplicands, stage_roots, ell, value_scale, root_scale, fix,
            counters);
        for (std::size_t index = 0; index < products.size(); ++index) {
            values[x_destinations[index]] =
                ring_add(lhs_values[index], products[index], value_mask);
            values[y_destinations[index]] =
                ring_sub(lhs_values[index], products[index], value_mask);
        }
        ++counters.stages;
        if (m == (n >> 1)) {
            break;
        }
        gap >>= 1;
        m <<= 1;
    }
    return values;
}

void print_secure_fft_counters(const char *direction, std::size_t degree,
                               int ell, int value_scale, int root_scale,
                               const SecureFftCounters &counters) {
    std::cout << "SECURE_CKKS_FFT direction=" << direction
              << " degree=" << degree << " ell=" << ell
              << " value_scale=" << value_scale
              << " root_scale=" << root_scale
              << " product_ell=" << counters.product_ell
              << " stages=" << counters.stages
              << " extend_calls=" << counters.extend_calls
              << " truncate_calls=" << counters.truncate_calls
              << " public_complex_multiplies="
              << counters.public_complex_multiplications
              << " packed_scalar_components="
              << counters.packed_scalar_components << std::endl;
}

} // namespace

template<typename T> void write_to_file(std::string filename, T data)
{
    std::ofstream file(filename);
    // printf("data size:%ld\n",data.size());
    // fflush(stdout);
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)  // 不在最后一个元素后添加逗号
        {
            file << ",";
        }
    }
    file.close();
}

template<typename T> std::vector<T> read_from_file2(std::string filename)
{
    std::vector<T> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line, ','))
    {
        std::istringstream iss(line);
        T value;
        iss >> value;
        data.push_back(value);
    }
    file.close();
    return data;
}
HE::HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod){
    this->party = party;
    this->poly_modulus_degree = poly_modulus_degree;
    this->io = io;
	this->plain_mod = plain_mod;
	this->plain_mod_2 = (plain_mod -1) / 2;

    // Generate keys
    EncryptionParameters parms(scheme_type::bfv);

	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(
		CoeffModulus::Create(poly_modulus_degree, coeff_bit_sizes));
	parms.set_plain_modulus(plain_mod);

	context = new SEALContext(parms, true, seal::sec_level_type::tc128);
	encoder = new BatchEncoder(*context);
	evaluator = new Evaluator(*context);

	if (party == BOB) {
		KeyGenerator keygen(*context);
		SecretKey sec_key = keygen.secret_key();
		PublicKey pub_key;
		keygen.create_public_key(pub_key);
		GaloisKeys gal_keys_;
		keygen.create_galois_keys(gal_keys_);
		RelinKeys relin_keys_;
		keygen.create_relin_keys(relin_keys_);

		stringstream os;
		pub_key.save(os);
		uint64_t pk_size = os.tellp();
		gal_keys_.save(os);
		uint64_t gk_size = (uint64_t)os.tellp() - pk_size;
		relin_keys_.save(os);
		uint64_t rk_size = (uint64_t)os.tellp() - pk_size - gk_size;

		string keys_ser = os.str();
		io->send_data(&pk_size, sizeof(uint64_t));
		io->send_data(&gk_size, sizeof(uint64_t));
		io->send_data(&rk_size, sizeof(uint64_t));
		io->send_data(keys_ser.c_str(), pk_size + gk_size + rk_size);

		stringstream os_sk;
		sec_key.save(os_sk);
		uint64_t sk_size = os_sk.tellp();
		string keys_ser_sk = os_sk.str();
		io->send_data(&sk_size, sizeof(uint64_t));
		io->send_data(keys_ser_sk.c_str(), sk_size);

		encryptor = new Encryptor(*context, pub_key);
		decryptor = new Decryptor(*context, sec_key);
	}
	else // party == ALICE
	{
		uint64_t pk_size;
		uint64_t gk_size;
		uint64_t rk_size;
		io->recv_data(&pk_size, sizeof(uint64_t));
		io->recv_data(&gk_size, sizeof(uint64_t));
		io->recv_data(&rk_size, sizeof(uint64_t));
		char *key_share = new char[pk_size + gk_size + rk_size];
		io->recv_data(key_share, pk_size + gk_size + rk_size);
		stringstream is;
		PublicKey pub_key;
		is.write(key_share, pk_size);
		pub_key.load(*context, is);
		gal_keys = new GaloisKeys();
		is.write(key_share + pk_size, gk_size);
		gal_keys->load(*context, is);
		relin_keys = new RelinKeys();
		is.write(key_share + pk_size + gk_size, rk_size);
		relin_keys->load(*context, is);
		delete[] key_share;

		uint64_t sk_size;
		io->recv_data(&sk_size, sizeof(uint64_t));
		char *key_share_sk = new char[sk_size];
		io->recv_data(key_share_sk, sk_size);
		stringstream is_sk;
		SecretKey sec_key;
		is_sk.write(key_share_sk, sk_size);
		sec_key.load(*context, is_sk);
		delete[] key_share_sk;
		decryptor = new Decryptor(*context, sec_key);

		encryptor = new Encryptor(*context, pub_key);
		vector<uint64_t> pod_matrix(poly_modulus_degree, 0ULL);
		Plaintext tmp;
		encoder->encode(pod_matrix, tmp);
		zero = new Ciphertext;
		encryptor->encrypt(tmp, *zero);
	}
    cout << "> HE instance initialized: " << endl;
    cout << "-> Poly Mod Degree: " << poly_modulus_degree << endl;
    cout << "-> Coeff Mod: " ;
	for(auto mod: coeff_bit_sizes){
		cout << mod << " ";
	}
	cout << endl;
    cout << "-> Plaintext Mod: " << plain_mod 
        << "(" << int (log2(plain_mod)) << " bits)" << endl;  
    cout << endl;
}


HE::HE(int party,
        NetIO* io,
        size_t poly_modulus_degree, 
        vector<int> coeff_bit_sizes, 
        uint64_t plain_mod_bit, bool bit){
    this->party = party;
    this->poly_modulus_degree = poly_modulus_degree;
    this->io = io;
    // Generate keys
    EncryptionParameters parms(scheme_type::bfv);

	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(
		CoeffModulus::Create(poly_modulus_degree, coeff_bit_sizes));
	auto seal_plain_mod = PlainModulus::Batching(poly_modulus_degree, plain_mod_bit);
	parms.set_plain_modulus(seal_plain_mod);
	this->plain_mod = seal_plain_mod.value();
	this->plain_mod_2 = (plain_mod -1) / 2;
	context = new SEALContext(parms, true, seal::sec_level_type::tc128);
	encoder = new BatchEncoder(*context);
	evaluator = new Evaluator(*context);

	if (party == BOB) {
		KeyGenerator keygen(*context);
		SecretKey sec_key = keygen.secret_key();
		PublicKey pub_key;
		keygen.create_public_key(pub_key);
		GaloisKeys gal_keys_;
		keygen.create_galois_keys(gal_keys_);
		RelinKeys relin_keys_;
		keygen.create_relin_keys(relin_keys_);

		stringstream os;
		pub_key.save(os);
		uint64_t pk_size = os.tellp();
		gal_keys_.save(os);
		uint64_t gk_size = (uint64_t)os.tellp() - pk_size;
		relin_keys_.save(os);
		uint64_t rk_size = (uint64_t)os.tellp() - pk_size - gk_size;

		string keys_ser = os.str();
		io->send_data(&pk_size, sizeof(uint64_t));
		io->send_data(&gk_size, sizeof(uint64_t));
		io->send_data(&rk_size, sizeof(uint64_t));
		io->send_data(keys_ser.c_str(), pk_size + gk_size + rk_size);

		stringstream os_sk;
		sec_key.save(os_sk);
		uint64_t sk_size = os_sk.tellp();
		string keys_ser_sk = os_sk.str();
		io->send_data(&sk_size, sizeof(uint64_t));
		io->send_data(keys_ser_sk.c_str(), sk_size);

		encryptor = new Encryptor(*context, pub_key);
		decryptor = new Decryptor(*context, sec_key);
	}
	else // party == ALICE
	{
		uint64_t pk_size;
		uint64_t gk_size;
		uint64_t rk_size;
		io->recv_data(&pk_size, sizeof(uint64_t));
		io->recv_data(&gk_size, sizeof(uint64_t));
		io->recv_data(&rk_size, sizeof(uint64_t));
		char *key_share = new char[pk_size + gk_size + rk_size];
		io->recv_data(key_share, pk_size + gk_size + rk_size);
		stringstream is;
		PublicKey pub_key;
		is.write(key_share, pk_size);
		pub_key.load(*context, is);
		gal_keys = new GaloisKeys();
		is.write(key_share + pk_size, gk_size);
		gal_keys->load(*context, is);
		relin_keys = new RelinKeys();
		is.write(key_share + pk_size + gk_size, rk_size);
		relin_keys->load(*context, is);
		delete[] key_share;

		uint64_t sk_size;
		io->recv_data(&sk_size, sizeof(uint64_t));
		char *key_share_sk = new char[sk_size];
		io->recv_data(key_share_sk, sk_size);
		stringstream is_sk;
		SecretKey sec_key;
		is_sk.write(key_share_sk, sk_size);
		sec_key.load(*context, is_sk);
		delete[] key_share_sk;
		decryptor = new Decryptor(*context, sec_key);

		encryptor = new Encryptor(*context, pub_key);
		vector<uint64_t> pod_matrix(poly_modulus_degree, 0ULL);
		Plaintext tmp;
		encoder->encode(pod_matrix, tmp);
		zero = new Ciphertext;
		encryptor->encrypt(tmp, *zero);
	}
    cout << "> HE instance initialized: " << endl;
    cout << "-> Poly Mod Degree: " << poly_modulus_degree << endl;
    cout << "-> Coeff Mod: " ;
	for(auto mod: coeff_bit_sizes){
		cout << mod << " ";
	}
	cout << endl;
    cout << "-> Plaintext Mod: " << plain_mod 
        << "(" << int (log2(plain_mod)) << " bits)" << endl;  
    cout << endl;
}



// 本地decode(不含intt，因为intt->fft需要一次模数转换) 不改变输入的位宽与scale，输入维度是多项式度数N，输出N/2个元素
vector<HE_CKKS::scalar_t> HE_CKKS::ckks_decode(vector<HE_CKKS::complex_t> x, int degree, bool half){
	if(half){
		vector<HE_CKKS::scalar_t> y(degree);
		vector<HE_CKKS::complex_t> complex_y(degree/2);
		complex_y = this->fft(x, degree, false);
		// puts("image");
		for(int i =0; i < degree/2; i++){
			y[i] = complex_y[matrix_reps_index_map_[i]].real();
			y[i+degree/2] = complex_y[matrix_reps_index_map_[i]].imag();
			// y[i] = complex_y[i].real();
			// cout<< complex_y[i].imag()/pow(2,fft_scale) << endl;
		}
		return y;
	}
	else{
		vector<HE_CKKS::scalar_t> y(degree/2);
		vector<HE_CKKS::complex_t> complex_y(degree/2);
		complex_y = this->fft(x, degree, false);
		// puts("image");
		// cout << "fft ok" << endl;
		for(int i =0; i < degree/2; i++){
			y[i] = complex_y[matrix_reps_index_map_[i]].real();
			// y[i] = complex_y[i].real();
			// cout<< complex_y[i].imag()/pow(2,fft_scale) << endl;
		}
		return y;
	}
}

// 本地encode(不含ntt，因为ifft->ntt中间需要一次模数转换)，不改变输入的位宽与scale，输入需已经是定点数表示，size是N/2个元素，输出是多项式度数N个复数
vector<HE_CKKS::scalar_t> HE_CKKS::ckks_encode(vector<HE_CKKS::scalar_t> x, int slot, bool half){
	int degree = slot * 2;
    vector<HE_CKKS::scalar_t> y(degree);
    vector<HE_CKKS::complex_t> complex_x(degree);
	vector<HE_CKKS::complex_t> complex_y(degree);
    for (std::size_t i = 0; i < slot; i++){
		complex_x[matrix_reps_index_map_[i]] = HE_CKKS::complex_t(x[i], 0);
		// TODO: if values are real, the following values should be set to zero, and multiply results by 2.
		complex_x[matrix_reps_index_map_[i + slot]] = std::conj(HE_CKKS::complex_t(x[i], 0));
	}
	// cout<<"print complex_x"<<endl;
	// for(int i = 0; i < size*2; i++){
	// 	cout << complex_x[i].real()/pow(2,10) << " " << complex_x[i].imag()/pow(2,10) << endl;
	// }
	// cout << "before ifft" << endl;
    complex_y = this->ifft(complex_x, degree, false);
	// cout << "end ifft" << endl;
    // puts("image");
    for(int i =0; i < degree; i++){
        y[i] = complex_y[i].real();
        // cout<< complex_y[i].imag()/pow(2,fft_scale) << endl;
    }
    return y;
}

vector<HE_CKKS::scalar_t> HE_CKKS::ckks_encode_secure(
    const vector<HE_CKKS::scalar_t> &x, int slot, int bitwidth, int scale,
    FixOp *fix, bool half) {
    (void)half;
    if (slot <= 0 || x.size() != static_cast<size_t>(slot) ||
        static_cast<size_t>(slot * 2) != poly_modulus_degree || fix == nullptr) {
        throw invalid_argument("secure CKKS encode input shape is invalid");
    }
    const int degree = slot * 2;
    const uint64_t mask = secure_ring_mask(bitwidth);
    vector<HE_CKKS::complex_t> values(degree);
    for (int index = 0; index < slot; ++index) {
        const uint64_t share = x[index] & mask;
        values[matrix_reps_index_map_[index]] =
            HE_CKKS::complex_t(share, 0);
        values[matrix_reps_index_map_[index + slot]] =
            HE_CKKS::complex_t(share, 0);
    }

    const uint64_t inverse_degree =
        EncodeToFxp<uint64_t>(1.0 / static_cast<double>(degree), fft_scale);
    SecureFftCounters counters;
    values = secure_transform_from_rev(
        std::move(values), bitwidth, scale, fft_scale,
        inv_root_powers_2n_scaled, inverse_degree, fix, counters);

    vector<HE_CKKS::scalar_t> output(degree);
    for (int index = 0; index < degree; ++index) {
        output[index] = values[index].real() & mask;
    }
    print_secure_fft_counters("encode", degree, bitwidth, scale, fft_scale,
                              counters);
    return output;
}

vector<HE_CKKS::scalar_t> HE_CKKS::ckks_decode_secure(
    const vector<HE_CKKS::complex_t> &x, int degree, int bitwidth, int scale,
    FixOp *fix, bool half) {
    if (degree <= 0 || x.size() != static_cast<size_t>(degree) ||
        static_cast<size_t>(degree) != poly_modulus_degree || fix == nullptr) {
        throw invalid_argument("secure CKKS decode input shape is invalid");
    }
    const uint64_t mask = secure_ring_mask(bitwidth);
    vector<HE_CKKS::complex_t> values(x.size());
    for (size_t index = 0; index < x.size(); ++index) {
        values[index] = HE_CKKS::complex_t(x[index].real() & mask,
                                           x[index].imag() & mask);
    }
    SecureFftCounters counters;
    values = secure_transform_to_rev(
        std::move(values), bitwidth, scale, fft_scale, root_powers_2n_scaled,
        fix, counters);

    vector<HE_CKKS::scalar_t> output(half ? degree : degree / 2);
    for (int index = 0; index < degree / 2; ++index) {
        output[index] =
            values[matrix_reps_index_map_[index]].real() & mask;
        if (half) {
            output[index + degree / 2] =
                values[matrix_reps_index_map_[index]].imag() & mask;
        }
    }
    print_secure_fft_counters("decode", degree, bitwidth, scale, fft_scale,
                              counters);
    return output;
}

// 本地定点数ntt，size是poly degree，输入输出均mod Q，输入范围是[-Q/2,Q/2],输出[0,Q]且构建了RNS。认为输入输出scale相同
void HE_CKKS::ckks_ntt(vector<HE_CKKS::scalar_t> x, int size,double scale, Plaintext &destination){
	// Next, conduct NTT in RNS.
	auto context_data_ptr = context->get_context_data(context->first_parms_id());
	auto &context_data = *context_data_ptr;
	auto &parms = context_data.parms();
	auto &coeff_modulus = parms.coeff_modulus();
	std::size_t coeff_modulus_size = coeff_modulus.size();
	std::size_t coeff_count = parms.poly_modulus_degree();
	auto ntt_tables = context_data.small_ntt_tables();
	MemoryPoolHandle pool = MemoryManager::GetPool();
	int64_t max_coeff = 0;
	for (std::size_t i = 0; i < coeff_count; i++)
	{
		max_coeff = std::max<>(max_coeff, (int64_t)std::fabs((int64_t)x[i]));
	}
	// cout << max_coeff << endl;
	// Verify that the values are not too large to fit in coeff_modulus
	// Note that we have an extra + 1 for the sign bit
	// Don't compute logarithmis of numbers less than 1

	int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, (int64_t)1))));
	// cout << "max_coeff_bit_count = " << max_coeff_bit_count << endl;
	// cout << "total_coeff_modulus_bit_count = " << context_data.total_coeff_modulus_bit_count() << endl;
	if (max_coeff_bit_count > context_data.total_coeff_modulus_bit_count())
	{
		throw std::invalid_argument("encoded values are too large");
	}

	double two_pow_64 = std::pow(2.0, 64);

	// Resize destination to appropriate size
	// Need to first set parms_id to zero, otherwise resize
	// will throw an exception.
	destination.parms_id() = parms_id_zero;
	destination.resize(seal::util::mul_safe(coeff_count, coeff_modulus_size));
	// for (std::size_t j = 0; j < coeff_modulus_size; j++)
	// {
	// 	std::cout<< "coeff_modulus["<<j<<"] = " << coeff_modulus[j].value() <<std::endl;
	// }
	// Use faster decomposition methods(RNS) when possible
	if (max_coeff_bit_count <= 64)
	{
		for (std::size_t i = 0; i < coeff_count; i++)
		{
			int64_t coeffd = (int64_t)x[i];
			bool is_negative = std::signbit(coeffd);
			
			std::uint64_t coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

			if (is_negative)
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = util::negate_uint_mod(
						util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]);
				}
			}
			else
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = util::barrett_reduce_64(coeffu, coeff_modulus[j]);
				}
			}
		}
	}
	else if (max_coeff_bit_count <= 128)
	{
		for (std::size_t i = 0; i < coeff_count; i++)
		{
			int64_t coeffd = (int64_t)x[i];
			bool is_negative = std::signbit(coeffd);
			coeffd = std::fabs(coeffd);

			std::uint64_t coeffu[2]{ static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64)),
										static_cast<std::uint64_t>(coeffd / two_pow_64) };

			if (is_negative)
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = util::negate_uint_mod(
						util::barrett_reduce_128(coeffu, coeff_modulus[j]), coeff_modulus[j]);
				}
			}
			else
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = util::barrett_reduce_128(coeffu, coeff_modulus[j]);
				}
			}
		}
	}
	else
	{
		// Slow case
		auto coeffu(util::allocate_uint(coeff_modulus_size, pool));
		for (std::size_t i = 0; i < coeff_modulus_size; i++)
		{
			int64_t coeffd = (int64_t)x[i];
			bool is_negative = std::signbit(coeffd);
			coeffd = std::fabs(coeffd);

			// We are at this point guaranteed to fit in the allocated space
			util::set_zero_uint(coeff_modulus_size, coeffu.get());
			auto coeffu_ptr = coeffu.get();
			while (coeffd >= 1)
			{
				*coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
				coeffd /= two_pow_64;
			}

			// Next decompose this coefficient
			context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);

			// Finally replace the sign if necessary
			if (is_negative)
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = util::negate_uint_mod(coeffu[j], coeff_modulus[j]);
				}
			}
			else
			{
				for (std::size_t j = 0; j < coeff_modulus_size; j++)
				{
					destination[i + (j * coeff_count)] = coeffu[j];
				}
			}
		}
	}
	// std::vector<uint64_t> my_destination;
	// for(int i=0;i<destination.coeff_count();i++){
	// 	my_destination.push_back(destination[i]);
	// }
	// write_to_file<std::vector<uint64_t>>("outputs/legacy_debug/destination.txt",my_destination);
	// my_destination = read_from_file2<uint64_t>("outputs/legacy_debug/destination_seal.txt");
	// for(int i=0;i<destination.coeff_count();i++){
	// 	destination[i] = my_destination[i];
	// }
	// Transform to NTT domain
	for (std::size_t i = 0; i < coeff_modulus_size; i++)
	{
		seal::util::ntt_negacyclic_harvey(destination.data(i * coeff_count), ntt_tables[i]);
	}
	destination.parms_id() = context->first_parms_id();
	destination.scale() = scale;
	return;
}

void HE_CKKS::ckks_ntt_rns(
    const vector<vector<HE_CKKS::scalar_t>> &rns_coefficients,
    double scale, Plaintext &destination) {
	if (!isfinite(scale) || scale <= 0.0) {
		throw invalid_argument("CKKS RNS encoding scale must be finite and positive");
	}
	auto context_data_ptr = context->first_context_data();
	if (!context_data_ptr) {
		throw invalid_argument("CKKS RNS encoding context is unavailable");
	}
	auto &parms = context_data_ptr->parms();
	auto &coeff_modulus = parms.coeff_modulus();
	const size_t coeff_modulus_size = coeff_modulus.size();
	const size_t coeff_count = parms.poly_modulus_degree();
	if (rns_coefficients.size() != coeff_modulus_size) {
		throw invalid_argument("CKKS RNS encoding limb count mismatch");
	}
	for (size_t limb = 0; limb < coeff_modulus_size; ++limb) {
		if (rns_coefficients[limb].size() != coeff_count) {
			throw invalid_argument("CKKS RNS encoding coefficient count mismatch");
		}
	}

	destination.parms_id() = parms_id_zero;
	destination.resize(seal::util::mul_safe(coeff_count,
	                                        coeff_modulus_size));
	auto ntt_tables = context_data_ptr->small_ntt_tables();
	for (size_t limb = 0; limb < coeff_modulus_size; ++limb) {
		const auto &modulus = coeff_modulus[limb];
		uint64_t *destination_limb = destination.data(limb * coeff_count);
		for (size_t coefficient = 0; coefficient < coeff_count;
		     ++coefficient) {
			const uint64_t residue = rns_coefficients[limb][coefficient];
			if (residue >= modulus.value()) {
				throw invalid_argument("CKKS RNS coefficient is outside its prime field");
			}
			destination_limb[coefficient] = residue;
		}
		seal::util::ntt_negacyclic_harvey(destination_limb,
		                                  ntt_tables[limb]);
	}
	destination.parms_id() = context_data_ptr->parms_id();
	destination.scale() = scale;
}

// 本地intt，输入输出均mod Q，输入是Plaintext, 范围是[0,Q],输出[-Q/2,Q/2]。暂不考虑scale。
vector<std::complex<HE_CKKS::scalar_t>> HE_CKKS::ckks_intt(const Plaintext &plain){
	auto context_data_ptr = context->get_context_data(plain.parms_id());
	auto &context_data = *context_data_ptr;
	auto &parms = context_data.parms();
	std::size_t coeff_modulus_size = parms.coeff_modulus().size();
	std::size_t coeff_count = parms.poly_modulus_degree();
	std::size_t rns_poly_uint64_count = util::mul_safe(coeff_count, coeff_modulus_size);
	auto ntt_tables = context_data.small_ntt_tables();
	auto decryption_modulus = context_data.total_coeff_modulus();
	auto upper_half_threshold = context_data.upper_half_threshold();
	int logn = util::get_power_of_two(coeff_count);
	// Create mutable copy of input
	MemoryPoolHandle pool = MemoryManager::GetPool();
	auto plain_copy(util::allocate_uint(rns_poly_uint64_count, pool));
	util::set_uint(plain.data(), rns_poly_uint64_count, plain_copy.get());
	// cout << "coeff_modulus_size:" << coeff_modulus_size << endl;
	// Transform each polynomial from NTT domain
	for (std::size_t i = 0; i < coeff_modulus_size; i++)
	{
		util::inverse_ntt_negacyclic_harvey(plain_copy.get() + (i * coeff_count), ntt_tables[i]);
	}
	// cout << "OK" << endl;
	// CRT-compose the polynomial
	context_data.rns_tool()->base_q()->compose_array(plain_copy.get(), coeff_count, pool);

	// Create floating-point representations of the multi-precision integer coefficients
	double inv_scale = double(1.0);
	double two_pow_64 = pow(2,64);
	std::vector<std::complex<scalar_t>> res(coeff_count,0);
	for (std::size_t i = 0; i < coeff_count; i++)
	{
		res[i] = 0.0;
		if (util::is_greater_than_or_equal_uint(
				plain_copy.get() + (i * coeff_modulus_size), upper_half_threshold, coeff_modulus_size))
		{
			double scaled_two_pow_64 = inv_scale;
			for (std::size_t j = 0; j < coeff_modulus_size; j++, scaled_two_pow_64 *= two_pow_64)
			{
				if (plain_copy[i * coeff_modulus_size + j] > decryption_modulus[j])
				{
					auto diff = plain_copy[i * coeff_modulus_size + j] - decryption_modulus[j];
					res[i] += diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
				}
				else
				{
					auto diff = decryption_modulus[j] - plain_copy[i * coeff_modulus_size + j];
					res[i] -= diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
				}
			}
		}
		else
		{
			double scaled_two_pow_64 = inv_scale;
			for (std::size_t j = 0; j < coeff_modulus_size; j++, scaled_two_pow_64 *= two_pow_64)
			{
				auto curr_coeff = plain_copy[i * coeff_modulus_size + j];
				res[i] += curr_coeff ? static_cast<double>(curr_coeff) * scaled_two_pow_64 : 0.0;
			}
		}

		// Scaling instead incorporated above; this can help in cases
		// where otherwise pow(two_pow_64, j) would overflow due to very
		// large coeff_modulus_size and very large scale
		// res[i] = res_accum * inv_scale;
	}
	return res;
}

void HE_CKKS::imag_repack(const UnifiedCiphertext &half, UnifiedCiphertext &real, UnifiedCiphertext &imag) {
	const auto loc = half.location();
	UnifiedCiphertext tmp(loc);
	
	if (loc == LOCATION::HOST) {
		// tmp <- conj(x + y*j) = x - y*j
		evaluator->complex_conjugate(half, *gal_keys, tmp);
		
		// imag <- (x - y*j) - (x + y*j) = 0 - 2y*j
		evaluator->sub(tmp, half, imag);
		
		// imag <- j * (-2y*j) = 2y
		// MulImageUnitInplace(imag, *he->context);
		evaluator->multiply_plain_inplace(imag, iunit);

		// real <- x + y*j + (x - y*j) = 2x
		evaluator->add(half, tmp, real);
	} else {
		devaluator->complex_conjugate(half, *dgal_keys, tmp);
		devaluator->sub(tmp, half, imag);
		devaluator->multiply_plain_inplace(imag, iunit);
		devaluator->add(half, tmp, real);
	}
}


//输入的x已经是定点数表示
vector<HE_CKKS::complex_t> HE_CKKS::fft(vector<HE_CKKS::complex_t> x, int degree, bool use_matrix_mul){
    int slot = degree/2;
    if(degree<=1){
        return x;
    }
    vector<HE_CKKS::complex_t> y(degree);
	using ST = typename std::make_signed<scalar_t>::type;
	if (use_matrix_mul){
		#pragma omp parallel for
		for(int r=0; r<slot; ++r){
			// cout << "r:" << r << endl;
			y[r] = HE_CKKS::complex_t(0, 0);
			// #pragma omp parallel for 
			for (size_t c = 0; c < degree; ++c) {
				y[r] += decode_mat_[r * degree + c] * x[c];
			}
			scalar_t real_dest = static_cast<scalar_t>(static_cast<ST>(y[r].real()) >> (this->fft_scale));
			scalar_t imag_dest = static_cast<scalar_t>(static_cast<ST>(y[r].imag()) >> (this->fft_scale));
			y[r] = HE_CKKS::complex_t(real_dest, imag_dest);
		}
	}
	else{
		fft_handler_.transform_to_rev(x.data(), seal::util::get_power_of_two(degree),
                                    root_powers_2n_scaled.data());
		return x;
	}
    return y;
}

vector<HE_CKKS::complex_t> HE_CKKS::ifft(vector<HE_CKKS::complex_t> x, int degree, bool use_matrix_mul){
    size_t logn = std::log2(degree);
	int slot = degree/2;
    if(degree<=1){
        return x;
    }
    vector<HE_CKKS::complex_t> y(degree, HE_CKKS::complex_t(0U, 0U));
	// cout << "scale:" << fft_scale << endl;
	if(use_matrix_mul){
		using ST = typename std::make_signed<scalar_t>::type;
		#pragma omp parallel for 
		for (size_t r = 0; r < degree; r++) {
			scalar_t tmp = 0;
			// By definition, m(X) = V * conj(v) + conj(V) * v
			// #pragma omp parallel for 
			for (size_t c = 0; c < slot; c++) {
				tmp += 2 * encode_mat_[r * slot + c].real() * x[c].real();
				tmp += 2 * encode_mat_[r * slot + c].imag() * x[c].imag();
			}
			y[r] = static_cast<scalar_t>(static_cast<ST>(tmp) >> (this->fft_scale + logn));
		}
	}
	else{
		// complex_t* x_ptr = x.data();
		scalar_t fix = EncodeToFxp<scalar_t>(1. / degree, fft_scale);
		fft_handler_.transform_from_rev(x.data(), seal::util::get_power_of_two(degree),
                                    inv_root_powers_2n_scaled.data(), &fix);
		return x;
	}
    return y;
}
