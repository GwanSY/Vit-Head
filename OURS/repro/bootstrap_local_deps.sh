#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
deb_dir="$repo/build_deps/debs"
gmp_root="$repo/build_deps/gmp"
mkdir -p "$deb_dir" "$gmp_root"

if [[ ! -f "$gmp_root/usr/include/x86_64-linux-gnu/gmp.h" ]]; then
  (
    cd "$deb_dir"
    apt-get download libgmp-dev
  )
  shopt -s nullglob
  packages=("$deb_dir"/libgmp-dev_*.deb)
  if [[ "${#packages[@]}" != "1" ]]; then
    echo "LOCAL_DEPS status=FAIL reason=expected_one_libgmp_dev_package count=${#packages[@]}" >&2
    exit 1
  fi
  dpkg-deb -x "${packages[0]}" "$gmp_root"
fi

test -f "$gmp_root/usr/include/x86_64-linux-gnu/gmp.h"
test -f "$gmp_root/usr/lib/x86_64-linux-gnu/libgmp.a"
test -f "$gmp_root/usr/lib/x86_64-linux-gnu/libgmpxx.a"
echo "LOCAL_DEPS status=PASS root=$repo/build_deps"
