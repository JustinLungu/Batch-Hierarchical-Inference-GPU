#!/bin/bash
set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ -n "${DNS_IP:-}" ]]; then
    log "Appending custom DNS: ${DNS_IP}"
    echo "nameserver ${DNS_IP}" >> /etc/resolv.conf
else
    log "DNS_IP not set — using Docker's default resolver"
fi

if [[ -n "${GATEWAY_IP:-}" ]]; then
    log "Setting default gateway to ${GATEWAY_IP}"
    ip route del default || true
    ip route add default via "${GATEWAY_IP}"
else
    log "GATEWAY_IP not set — using Docker's default gateway"
fi

log "Configuring SSH access"
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
if [[ -n "${PASS:-}" ]]; then
    echo "root:${PASS}" | chpasswd
else
    log "Warning: PASS environment variable not set. Root SSH login may be blocked."
fi
service ssh start

chmod +x /app/start.sh

log "Starting edge_device.py"
exec python edge_device/edge_device.py
