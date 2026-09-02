#include "he.h"
#include <cstring>
#include <limits>

namespace {

using WideScalar = HE_CKKS::wide_scalar_t;
using WideComplex = HE_CKKS::wide_complex_t;

WideScalar wide_mask(int bits) {
    if (bits <= 0 || bits >= 127) {
        throw std::invalid_argument("invalid wide CKKS FFT ring width");
    }
    return (WideScalar{1} << bits) - 1;
}

WideScalar wide_sign_extend(WideScalar value, int from_bits, int to_bits) {
    if (from_bits <= 1 || from_bits >= 127 || to_bits <= from_bits ||
        to_bits >= 127) {
        throw std::invalid_argument("invalid wide CKKS sign extension");
    }
    const WideScalar from_mask = wide_mask(from_bits);
    const WideScalar to_mask = wide_mask(to_bits);
    value &= from_mask;
    if ((value >> (from_bits - 1)) & 1) {
        value |= to_mask ^ from_mask;
    }
    return value & to_mask;
}

WideScalar wide_arithmetic_right_shift(WideScalar value, int bits,
                                       int shift, WideScalar output_mask) {
    if (bits <= 1 || bits >= 127 || shift < 0 || shift >= bits) {
        throw std::invalid_argument("invalid wide arithmetic right shift");
    }
    const WideScalar input_mask = wide_mask(bits);
    value &= input_mask;
    WideScalar shifted = value >> shift;
    if ((value >> (bits - 1)) & 1) {
        const int low_bits = bits - shift;
        const WideScalar low_mask = (WideScalar{1} << low_bits) - 1;
        shifted |= input_mask ^ low_mask;
    }
    return shifted & output_mask;
}

int64_t signed_public_word(uint64_t word) {
    int64_t result = 0;
    static_assert(sizeof(result) == sizeof(word),
                  "unexpected fixed-width root word");
    std::memcpy(&result, &word, sizeof(word));
    return result;
}

WideScalar root_modulus_word(uint64_t word, WideScalar modulus) {
    const int64_t signed_word = signed_public_word(word);
    if (signed_word >= 0) {
        return static_cast<WideScalar>(signed_word);
    }
    const WideScalar magnitude =
        static_cast<WideScalar>(-(signed_word + 1)) + WideScalar{1};
    return (modulus - magnitude) & (modulus - WideScalar{1});
}

WideComplex wide_add(const WideComplex &lhs, const WideComplex &rhs,
                     WideScalar mask) {
    return {(lhs.real + rhs.real) & mask, (lhs.imag + rhs.imag) & mask};
}

WideComplex wide_sub(const WideComplex &lhs, const WideComplex &rhs,
                     WideScalar mask) {
    return {(lhs.real - rhs.real) & mask, (lhs.imag - rhs.imag) & mask};
}

WideComplex wide_mul_root(const WideComplex &value, const complex<uint64_t> &root,
                          int width, int root_scale) {
    if (width <= 1 || root_scale <= 0 || width + root_scale >= 127) {
        throw std::invalid_argument("wide CKKS FFT product width is invalid");
    }
    // A root is a signed fixed-point integer at `root_scale` bits.  The
    // product must therefore be retained for width+root_scale bits before
    // the arithmetic right shift.  Reducing it to Z_(2^width) first loses
    // exactly the high root_scale bits that become the top width bits after
    // the shift; that made l=43 Encode/Decode non-additive at the paper's
    // high CKKS scales.
    const int product_bits = width + root_scale;
    const WideScalar product_modulus = WideScalar{1} << product_bits;
    const WideScalar product_mask = product_modulus - WideScalar{1};
    const WideScalar value_mask = wide_mask(width);
    // The input is a signed value in Z_(2^width).  It must be sign-extended
    // into the product ring before multiplying by the signed fixed-point
    // root; zero extension turns every negative share into a large positive
    // value once width is raised for the 20-bit root fractional part.
    const WideScalar real_value =
        wide_sign_extend(value.real, width, product_bits);
    const WideScalar imag_value =
        wide_sign_extend(value.imag, width, product_bits);
    const WideScalar root_real = root_modulus_word(root.real(), product_modulus);
    const WideScalar root_imag = root_modulus_word(root.imag(), product_modulus);
    const WideScalar real =
        (real_value * root_real - imag_value * root_imag) & product_mask;
    const WideScalar imag =
        (real_value * root_imag + imag_value * root_real) & product_mask;
    return {wide_arithmetic_right_shift(real, product_bits, root_scale,
                                         value_mask),
            wide_arithmetic_right_shift(imag, product_bits, root_scale,
                                         value_mask)};
}

vector<WideComplex> wide_fft_forward(
    vector<WideComplex> values, int width, int root_scale,
    const vector<complex<uint64_t>> &roots) {
    const size_t n = values.size();
    if (n == 0 || (n & (n - 1)) != 0 || roots.size() < n) {
        throw std::invalid_argument("invalid wide CKKS FFT shape");
    }
    const WideScalar mask = wide_mask(width);
    size_t root_index = 0;
    size_t gap = n >> 1;
    size_t m = 1;
    for (;;) {
        size_t offset = 0;
        for (size_t group = 0; group < m; ++group) {
            if (++root_index >= roots.size()) {
                throw std::out_of_range("wide CKKS FFT root index");
            }
            const auto &root = roots[root_index];
            for (size_t index = 0; index < gap; ++index) {
                const size_t x_index = offset + index;
                const size_t y_index = x_index + gap;
                const WideComplex lhs = values[x_index];
                const WideComplex product =
                    wide_mul_root(values[y_index], root, width, root_scale);
                values[x_index] = wide_add(lhs, product, mask);
                values[y_index] = wide_sub(lhs, product, mask);
            }
            offset += gap << 1;
        }
        if (m == (n >> 1)) {
            break;
        }
        gap >>= 1;
        m <<= 1;
    }
    return values;
}

WideScalar wide_mul_scalar(WideScalar value, uint64_t scalar, int width,
                           int scalar_scale) {
    if (width <= 1 || scalar_scale <= 0 || width + scalar_scale >= 127) {
        throw std::invalid_argument("wide CKKS scalar product width is invalid");
    }
    // `scalar` is also a fixed-point value.  Preserve its fractional bits
    // through multiplication, then shift them away in the wider product
    // ring; masking before the shift is not equivalent.
    const int product_bits = width + scalar_scale;
    const WideScalar product_mask = wide_mask(product_bits);
    const WideScalar product =
        (wide_sign_extend(value, width, product_bits) *
         static_cast<WideScalar>(scalar)) & product_mask;
    return wide_arithmetic_right_shift(product, product_bits, scalar_scale,
                                       wide_mask(width));
}

complex<uint64_t> wide_root_mul_scalar(const complex<uint64_t> &root,
                                       uint64_t scalar, int width,
                                       int root_scale) {
    const int product_bits = root_scale * 2;
    if (product_bits >= 127 || width <= 1 || width >= 127) {
        throw std::invalid_argument("wide CKKS root-scalar width is invalid");
    }
    const WideScalar product_modulus = WideScalar{1} << product_bits;
    const WideScalar product_mask = product_modulus - WideScalar{1};
    const WideScalar root_real = root_modulus_word(root.real(), product_modulus);
    const WideScalar root_imag = root_modulus_word(root.imag(), product_modulus);
    const WideScalar real_product =
        (root_real * static_cast<WideScalar>(scalar)) & product_mask;
    const WideScalar imag_product =
        (root_imag * static_cast<WideScalar>(scalar)) & product_mask;
    const WideScalar product_result_mask = wide_mask(product_bits);
    const WideScalar real_result = wide_arithmetic_right_shift(
        real_product, product_bits, root_scale, product_result_mask);
    const WideScalar imag_result = wide_arithmetic_right_shift(
        imag_product, product_bits, root_scale, product_result_mask);
    // mulRootScalar returns a public root word, not a value in the source
    // ring.  When the two root-scale factors fit below 64 bits, the result's
    // sign bit is below bit 63 and must be extended before wide_mul_root
    // interprets the word as a signed fixed-point root.  Without this,
    // fft_scale=20 turns every negative final twiddle into a large positive
    // root; fft_scale=40 happened to hide the bug because the product already
    // occupied at least 64 bits.
    auto root_word = [&](WideScalar value) -> uint64_t {
        if (product_bits < 64) {
            value = wide_sign_extend(value, product_bits, 64);
        }
        return static_cast<uint64_t>(value);
    };
    return {root_word(real_result), root_word(imag_result)};
}

vector<WideComplex> wide_fft_inverse(
    vector<WideComplex> values, int width, int root_scale,
    const vector<complex<uint64_t>> &roots, uint64_t inverse_degree_scale) {
    const size_t n = values.size();
    if (n == 0 || (n & (n - 1)) != 0 || roots.size() < n ||
        inverse_degree_scale == 0) {
        throw std::invalid_argument("invalid wide inverse CKKS FFT shape");
    }
    const WideScalar mask = wide_mask(width);
    size_t root_index = 0;
    size_t gap = 1;
    size_t m = n >> 1;
    for (; m > 1; m >>= 1) {
        size_t offset = 0;
        for (size_t group = 0; group < m; ++group) {
            if (++root_index >= roots.size()) {
                throw std::out_of_range("wide inverse CKKS FFT root index");
            }
            const auto &root = roots[root_index];
            for (size_t index = 0; index < gap; ++index) {
                const size_t x_index = offset + index;
                const size_t y_index = x_index + gap;
                const WideComplex lhs = values[x_index];
                const WideComplex rhs = values[y_index];
                values[x_index] = wide_add(lhs, rhs, mask);
                values[y_index] = wide_mul_root(
                    wide_sub(lhs, rhs, mask), root, width, root_scale);
            }
            offset += gap << 1;
        }
        gap <<= 1;
    }

    if (++root_index >= roots.size()) {
        throw std::out_of_range("wide inverse CKKS FFT final root index");
    }
    const auto scaled_root = wide_root_mul_scalar(
        roots[root_index], inverse_degree_scale, width, root_scale);
    size_t offset = 0;
    for (size_t group = 0; group < 1; ++group) {
        for (size_t index = 0; index < gap; ++index) {
            const size_t x_index = offset + index;
            const size_t y_index = x_index + gap;
            const WideComplex lhs = values[x_index];
            const WideComplex rhs = values[y_index];
            values[x_index] = {
                wide_mul_scalar((lhs.real + rhs.real) & mask,
                                inverse_degree_scale, width, root_scale),
                wide_mul_scalar((lhs.imag + rhs.imag) & mask,
                                inverse_degree_scale, width, root_scale)};
            const WideComplex difference = wide_sub(lhs, rhs, mask);
            values[y_index] = wide_mul_root(difference, scaled_root, width,
                                            root_scale);
        }
        offset += gap << 1;
    }
    return values;
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
        sci::NetIO* io,
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

	if (party == sci::BOB) {
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
        sci::NetIO* io,
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

	if (party == sci::BOB) {
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

vector<HE_CKKS::scalar_t> HE_CKKS::ckks_decode_wide(
    const vector<wide_complex_t> &x, int degree, int output_ell,
    int source_scale, int target_scale, bool half, int working_ell) const {
    if (degree <= 0 || x.size() != static_cast<size_t>(degree) ||
        output_ell <= 1 || output_ell >= 63 || source_scale < target_scale ||
        source_scale >= 127 || target_scale < 0 || fft_scale <= 0) {
        throw invalid_argument("invalid wide CKKS decode metadata");
    }
    const int width = working_ell > 0 ? working_ell : output_ell;
    if (width <= 1 || width >= 127) {
        throw invalid_argument("invalid wide CKKS working ring width");
    }
    const int final_shift = source_scale - target_scale;
    if (width + fft_scale >= 127 || final_shift < 0 ||
        final_shift >= width) {
        throw invalid_argument("wide CKKS decode exceeds integer width");
    }
    const auto transformed = wide_fft_forward(
        x, width, fft_scale, root_powers_2n_scaled);
    const uint64_t output_mask = (uint64_t{1} << output_ell) - 1;
    vector<scalar_t> result(static_cast<size_t>(half ? degree : degree / 2));
    for (int index = 0; index < degree / 2; ++index) {
        const auto &value = transformed[matrix_reps_index_map_[index]];
        result[index] = static_cast<uint64_t>(wide_arithmetic_right_shift(
            value.real, width, final_shift, output_mask));
        if (half) {
            result[index + degree / 2] = static_cast<uint64_t>(
                wide_arithmetic_right_shift(value.imag, width, final_shift,
                                            output_mask));
        }
    }
    return result;
}

vector<HE_CKKS::wide_complex_t> HE_CKKS::ckks_encode_wide(
    const vector<wide_complex_t> &x, int degree, int working_ell) const {
    if (degree <= 0 || x.size() != static_cast<size_t>(degree) ||
        (degree & (degree - 1)) != 0 || fft_scale <= 0) {
        throw invalid_argument("invalid wide CKKS encode metadata");
    }
    const int width = working_ell > 0 ? working_ell : 43 + 40;
    if (width <= 1 || width >= 127 || width + fft_scale >= 127) {
        throw invalid_argument("wide CKKS encode exceeds integer width");
    }
    const scalar_t inverse_degree_scale =
        EncodeToFxp<scalar_t>(1.0 / static_cast<double>(degree), fft_scale);
    if (inverse_degree_scale == 0) {
        throw invalid_argument("wide CKKS encode inverse scale rounded to zero");
    }
    return wide_fft_inverse(x, width, fft_scale,
                            inv_root_powers_2n_scaled,
                            inverse_degree_scale);
}

#if 0 // Removed experimental public-field FFT; paper Algorithm 2 uses local Encode then ring-to-field.
namespace {

uint64_t field_add(uint64_t a, uint64_t b, uint64_t modulus) {
    const __uint128_t sum = static_cast<__uint128_t>(a) + b;
    return static_cast<uint64_t>(sum % modulus);
}

uint64_t field_sub(uint64_t a, uint64_t b, uint64_t modulus) {
    return a >= b ? a - b : static_cast<uint64_t>(
        static_cast<__uint128_t>(a) + modulus - b);
}

uint64_t field_mul(uint64_t a, uint64_t b, uint64_t modulus) {
    return static_cast<uint64_t>(
        (static_cast<__uint128_t>(a) * b) % modulus);
}

uint64_t field_inv_power_of_two(int bits, uint64_t modulus) {
    uint64_t result = 1;
    // modulus is an odd CKKS prime, so (q+1)/2 is the inverse of 2.
    const uint64_t inv_two = (modulus + 1) / 2;
    for (int i = 0; i < bits; ++i) {
        result = field_mul(result, inv_two, modulus);
    }
    return result;
}

uint64_t field_signed_word(uint64_t word, uint64_t modulus) {
    int64_t signed_word = 0;
    static_assert(sizeof(signed_word) == sizeof(word),
                  "unexpected fixed-width CKKS root word");
    std::memcpy(&signed_word, &word, sizeof(word));
    if (signed_word >= 0) {
        return static_cast<uint64_t>(signed_word) % modulus;
    }
    const uint64_t magnitude =
        static_cast<uint64_t>(-(signed_word + 1)) + uint64_t{1};
    return magnitude % modulus == 0
               ? 0
               : modulus - (magnitude % modulus);
}

HE_CKKS::complex_t field_add_complex(const HE_CKKS::complex_t &a,
                                     const HE_CKKS::complex_t &b,
                                     uint64_t modulus) {
    return {field_add(a.real(), b.real(), modulus),
            field_add(a.imag(), b.imag(), modulus)};
}

HE_CKKS::complex_t field_sub_complex(const HE_CKKS::complex_t &a,
                                     const HE_CKKS::complex_t &b,
                                     uint64_t modulus) {
    return {field_sub(a.real(), b.real(), modulus),
            field_sub(a.imag(), b.imag(), modulus)};
}

HE_CKKS::complex_t field_mul_root(const HE_CKKS::complex_t &value,
                                  const HE_CKKS::complex_t &root,
                                  uint64_t modulus, uint64_t inv_scale) {
    const uint64_t rr = field_signed_word(root.real(), modulus);
    const uint64_t ri = field_signed_word(root.imag(), modulus);
    const uint64_t real = field_sub(field_mul(value.real(), rr, modulus),
                                    field_mul(value.imag(), ri, modulus),
                                    modulus);
    const uint64_t imag = field_add(field_mul(value.real(), ri, modulus),
                                    field_mul(value.imag(), rr, modulus),
                                    modulus);
    return {field_mul(real, inv_scale, modulus),
            field_mul(imag, inv_scale, modulus)};
}

HE_CKKS::complex_t field_mul_scalar(const HE_CKKS::complex_t &value,
                                    uint64_t scalar, uint64_t modulus,
                                    uint64_t inv_scale) {
    return {field_mul(field_mul(value.real(), scalar, modulus), inv_scale,
                      modulus),
            field_mul(field_mul(value.imag(), scalar, modulus), inv_scale,
                      modulus)};
}

HE_CKKS::complex_t field_mul_root_scalar(
    const HE_CKKS::complex_t &root, uint64_t scalar, uint64_t modulus,
    uint64_t inv_scale) {
    const uint64_t rr = field_signed_word(root.real(), modulus);
    const uint64_t ri = field_signed_word(root.imag(), modulus);
    return {field_mul(field_mul(rr, scalar, modulus), inv_scale, modulus),
            field_mul(field_mul(ri, scalar, modulus), inv_scale, modulus)};
}

} // namespace

vector<HE_CKKS::complex_t> HE_CKKS::ckks_fft_field(
    const vector<complex_t> &input, int degree, uint64_t modulus,
    bool inverse) const {
    if (degree <= 0 || input.size() != static_cast<size_t>(degree) ||
        (degree & (degree - 1)) != 0 || modulus == 0 || fft_scale <= 0) {
        throw invalid_argument("invalid field CKKS FFT metadata");
    }
    const auto &roots = inverse ? inv_root_powers_2n_scaled
                                : root_powers_2n_scaled;
    if (roots.size() < static_cast<size_t>(degree)) {
        throw invalid_argument("field CKKS FFT roots are incomplete");
    }
    const uint64_t inv_scale =
        field_inv_power_of_two(fft_scale, modulus);
    vector<complex_t> values = input;
    const uint64_t field_mask = modulus - 1;
    (void)field_mask;

    if (!inverse) {
        size_t root_index = 0;
        size_t gap = static_cast<size_t>(degree) >> 1;
        size_t m = 1;
        for (;;) {
            size_t offset = 0;
            for (size_t group = 0; group < m; ++group) {
                if (++root_index >= roots.size()) {
                    throw out_of_range("field CKKS FFT root index");
                }
                const auto &root = roots[root_index];
                for (size_t index = 0; index < gap; ++index) {
                    const size_t x_index = offset + index;
                    const size_t y_index = x_index + gap;
                    const complex_t lhs = values[x_index];
                    const complex_t product =
                        field_mul_root(values[y_index], root, modulus,
                                       inv_scale);
                    values[x_index] = field_add_complex(lhs, product, modulus);
                    values[y_index] = field_sub_complex(lhs, product, modulus);
                }
                offset += gap << 1;
            }
            if (m == (static_cast<size_t>(degree) >> 1)) break;
            gap >>= 1;
            m <<= 1;
        }
        return values;
    }

    const scalar_t inverse_degree_scale =
        EncodeToFxp<scalar_t>(1.0 / static_cast<double>(degree), fft_scale);
    if (inverse_degree_scale == 0) {
        throw invalid_argument("field CKKS inverse scale rounded to zero");
    }
    size_t root_index = 0;
    size_t gap = 1;
    size_t m = static_cast<size_t>(degree) >> 1;
    for (; m > 1; m >>= 1) {
        size_t offset = 0;
        for (size_t group = 0; group < m; ++group) {
            if (++root_index >= roots.size()) {
                throw out_of_range("field inverse CKKS FFT root index");
            }
            const auto &root = roots[root_index];
            for (size_t index = 0; index < gap; ++index) {
                const size_t x_index = offset + index;
                const size_t y_index = x_index + gap;
                const complex_t lhs = values[x_index];
                const complex_t rhs = values[y_index];
                values[x_index] = field_add_complex(lhs, rhs, modulus);
                values[y_index] = field_mul_root(
                    field_sub_complex(lhs, rhs, modulus), root, modulus,
                    inv_scale);
            }
            offset += gap << 1;
        }
        gap <<= 1;
    }
    if (++root_index >= roots.size()) {
        throw out_of_range("field inverse CKKS FFT final root index");
    }
    const complex_t scaled_root = field_mul_root_scalar(
        roots[root_index], inverse_degree_scale, modulus, inv_scale);
    for (size_t index = 0; index < gap; ++index) {
        const size_t x_index = index;
        const size_t y_index = x_index + gap;
        const complex_t lhs = values[x_index];
        const complex_t rhs = values[y_index];
        values[x_index] = field_mul_scalar(
            field_add_complex(lhs, rhs, modulus), inverse_degree_scale,
            modulus, inv_scale);
        values[y_index] = field_mul_root(
            field_sub_complex(lhs, rhs, modulus), scaled_root, modulus,
            inv_scale);
    }
    return values;
}

#endif

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
