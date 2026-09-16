set pagination off
set confirm off
set breakpoint pending on
set print pretty off
set print demangle on
set can-use-hw-watchpoints 0

# PhantomCiphertext fields at the wrapper entry are:
# chain_index@0, size@8, poly_degree@16, limbs@24, scale@32,
# correction_factor@40, noise_scale_degree@48, NTT@56.
# rsi/rdx are the two ciphertext references in the SysV ABI.

break 'Linear_ckks::softmax_v(HE_CKKS*, std::vector<std::vector<std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> >, std::allocator<std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> > > >, std::allocator<std::vector<std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> >, std::allocator<std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> > > > > > const&, std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> > const&, FCMetadata const&, std::vector<UnifiedCiphertext, std::allocator<UnifiedCiphertext> >&, double, std::vector<bool, std::allocator<bool> > const*, bool)'
commands
  silent
  enable 2 3
  continue
end

break 'PhantomEvaluator::multiply_inplace(PhantomCiphertext&, PhantomCiphertext const&) const'
commands
  silent
  if (*(unsigned long*)$rsi != *(unsigned long*)$rdx) || (*(unsigned char*)($rsi+56) != *(unsigned char*)($rdx+56)) || (*(unsigned long*)($rsi+24) != *(unsigned long*)($rdx+24))
    printf "OURS_A4_CIPHERTEXT_MISMATCH op=multiply context=%p\n", *(void**)$rdi
    printf "operand1 ptr=%p chain=%lu size=%lu poly=%lu limbs=%lu scale=%g ntt=%u\n", $rsi, *(unsigned long*)$rsi, *(unsigned long*)($rsi+8), *(unsigned long*)($rsi+16), *(unsigned long*)($rsi+24), *(double*)($rsi+32), *(unsigned char*)($rsi+56)
    printf "operand2 ptr=%p chain=%lu size=%lu poly=%lu limbs=%lu scale=%g ntt=%u\n", $rdx, *(unsigned long*)$rdx, *(unsigned long*)($rdx+8), *(unsigned long*)($rdx+16), *(unsigned long*)($rdx+24), *(double*)($rdx+32), *(unsigned char*)($rdx+56)
    printf "operand1 parameter_id=not_stored_in_phantom; operand2 parameter_id=not_stored_in_phantom\n"
    bt 16
    disable 2
  end
  continue
end

break 'PhantomEvaluator::add_inplace(PhantomCiphertext&, PhantomCiphertext const&) const'
commands
  silent
  if (*(unsigned long*)$rsi != *(unsigned long*)$rdx) || (*(unsigned char*)($rsi+56) != *(unsigned char*)($rdx+56)) || (*(unsigned long*)($rsi+24) != *(unsigned long*)($rdx+24))
    printf "OURS_A4_CIPHERTEXT_MISMATCH op=add context=%p\n", *(void**)$rdi
    printf "operand1 ptr=%p chain=%lu size=%lu poly=%lu limbs=%lu scale=%g ntt=%u\n", $rsi, *(unsigned long*)$rsi, *(unsigned long*)($rsi+8), *(unsigned long*)($rsi+16), *(unsigned long*)($rsi+24), *(double*)($rsi+32), *(unsigned char*)($rsi+56)
    printf "operand2 ptr=%p chain=%lu size=%lu poly=%lu limbs=%lu scale=%g ntt=%u\n", $rdx, *(unsigned long*)$rdx, *(unsigned long*)($rdx+8), *(unsigned long*)($rdx+16), *(unsigned long*)($rdx+24), *(double*)($rdx+32), *(unsigned char*)($rdx+56)
    printf "operand1 parameter_id=not_stored_in_phantom; operand2 parameter_id=not_stored_in_phantom\n"
    bt 16
    disable 3
  end
  continue
end

disable 2 3
ignore 1 1

run
