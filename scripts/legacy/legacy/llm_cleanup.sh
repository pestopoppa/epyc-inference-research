#!/bin/bash

# LLM Cleanup Utility
# Frees memory, clears caches, and resets hugepages if needed
# Intended for recovery after an unclean llama.cpp / DeepSeek exit

set -e

echo "🔍 Checking hugepage usage..."
grep Huge /proc/meminfo | tee /tmp/huge_before.log

echo "📤 Dropping filesystem caches..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "📤 Killing any leftover llama.cpp processes..."
sudo pkill -f llama-cli 2>/dev/null || true
sudo pkill -f llama 2>/dev/null || true

sleep 1

# Re-check hugepages
echo "🔄 Re-checking hugepage state..."
grep Huge /proc/meminfo | tee /tmp/huge_after_drop_cache.log

# Offer to reset hugepages if still reserved
FREE=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')
TOTAL=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
RSVD=$(grep HugePages_Rsvd /proc/meminfo | awk '{print $2}')

if [[ "$RSVD" -gt 0 ]] || [[ "$FREE" -lt "$TOTAL" ]]; then
  echo "⚠️  Detected hugepages still reserved or in use: $RSVD reserved / $FREE free / $TOTAL total."
  echo "➡️  Would you like to reset hugepages now? (y/n)"
  read -r RESP
  if [[ "$RESP" == "y" ]]; then
    echo "🧨 Resetting hugepages..."
    sudo sysctl -w vm.nr_hugepages=0
    sleep 1
    echo "✅ Re-allocating hugepages to $TOTAL..."
    sudo sysctl -w vm.nr_hugepages=$TOTAL
    grep Huge /proc/meminfo | tee /tmp/huge_final.log
  else
    echo "✅ Skipping hugepage reset."
  fi
else
  echo "✅ Hugepages look clean. No reset needed."
fi

echo "✅ Cleanup complete."
