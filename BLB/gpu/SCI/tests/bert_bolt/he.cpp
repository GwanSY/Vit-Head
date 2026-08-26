#include "he.h"
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
vector<HE_CKKS::scalar_t> HE_CKKS::ckks_decode(vector<HE_CKKS::complex_t> x, int degree){
    vector<HE_CKKS::scalar_t> y(degree/2);
	vector<HE_CKKS::complex_t> complex_y(degree/2);
    complex_y = this->fft(x, degree, false);
	// puts("image");
    for(int i =0; i < degree/2; i++){
        y[i] = complex_y[matrix_reps_index_map_[i]].real();
		// y[i] = complex_y[i].real();
		// cout<< complex_y[i].imag()/pow(2,fft_scale) << endl;
    }
	
    return y;
}

// 本地encode(不含ntt，因为ifft->ntt中间需要一次模数转换)，不改变输入的位宽与scale，输入需已经是定点数表示，size是N/2个元素，输出是多项式度数N个复数
vector<HE_CKKS::scalar_t> HE_CKKS::ckks_encode(vector<HE_CKKS::scalar_t> x, int slot){
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
	// write_to_file<std::vector<uint64_t>>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/destination.txt",my_destination);
	// my_destination = read_from_file2<uint64_t>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/destination_seal.txt");
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
vector<std::complex<HE_CKKS::scalar_t>> HE_CKKS::ckks_intt(Plaintext &plain){
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

//输入的x已经是定点数表示
vector<HE_CKKS::complex_t> HE_CKKS::fft(vector<HE_CKKS::complex_t> x, int degree, bool use_matrix_mul){
    int slot = degree/2;
    if(degree<=1){
        return x;
    }
    vector<HE_CKKS::complex_t> y(degree);
	using ST = typename std::make_signed<scalar_t>::type;
	if (use_matrix_mul){
		#pragma omp parallel for num_threads(8)
		for(int r=0; r<slot; ++r){
			y[r] = HE_CKKS::complex_t(0, 0);
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
		#pragma omp parallel for num_threads(8)
		for (size_t r = 0; r < degree; r++) {
			scalar_t tmp = 0;
			// By definition, m(X) = V * conj(v) + conj(V) * v
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