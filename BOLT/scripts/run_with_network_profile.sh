#!/usr/bin/env bash
set -euo pipefail

# Run a BOLT command under a temporary Linux tc/netem profile.  This changes
# the selected interface for the lifetime of the child and requires root (or
# CAP_NET_ADMIN).  The profile is opt-in; normal BOLT runs are unshaped.
IFACE="${IFACE:-lo}"
RATE_MBPS="${RATE_MBPS:-100}"
RTT_MS="${RTT_MS:-80}"

if [[ "$#" -eq 0 || "$1" != "--" ]]; then
  echo "usage: IFACE=lo RATE_MBPS=100 RTT_MS=80 $0 -- COMMAND [ARGS...]" >&2
  exit 2
fi
shift
if [[ "$#" -eq 0 ]]; then
  echo "network profile requires a command after --" >&2
  exit 2
fi
if [[ "$EUID" -ne 0 ]]; then
  echo "run this wrapper with sudo or grant CAP_NET_ADMIN" >&2
  exit 2
fi
if ! command -v tc >/dev/null; then
  echo "missing tc (install iproute2)" >&2
  exit 2
fi

case "$RATE_MBPS" in
  ''|*[!0-9]*) echo "RATE_MBPS must be a nonnegative integer" >&2; exit 2 ;;
esac
case "$RTT_MS" in
  ''|*[!0-9]*) echo "RTT_MS must be a nonnegative integer" >&2; exit 2 ;;
esac

# tc delay is one-way.  RTT_MS is split equally over the two egress paths.
ONE_WAY_DELAY_MS=$((RTT_MS / 2))
tc qdisc add dev "$IFACE" root netem \
  rate "${RATE_MBPS}mbit" delay "${ONE_WAY_DELAY_MS}ms"

cleanup() {
  tc qdisc del dev "$IFACE" root >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export BOLT_NETWORK_PROFILE="tc-netem"
export BOLT_NETWORK_INTERFACE="$IFACE"
export BOLT_NETWORK_RATE_MBPS="$RATE_MBPS"
export BOLT_NETWORK_RTT_MS="$RTT_MS"

"$@"
