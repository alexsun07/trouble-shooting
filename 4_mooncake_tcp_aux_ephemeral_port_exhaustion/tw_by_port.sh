#!/usr/bin/env bash
# TIME_WAIT sockets per destination port on this host (run on the host, or in a --network host
# container). Mooncake's TCP transport ports show up as the top entries when it churns.
#   ./tw_by_port.sh            # one snapshot
#   watch -n 25 ./tw_by_port.sh
ss -tan state time-wait | awk 'NR > 1 {print $4}' | sed 's/.*://' | sort | uniq -c | sort -rn | head -8
echo "total time-wait: $(ss -tan state time-wait | tail -n +2 | wc -l)"
echo "ephemeral range: $(cat /proc/sys/net/ipv4/ip_local_port_range)  tcp_tw_reuse: $(cat /proc/sys/net/ipv4/tcp_tw_reuse)"
