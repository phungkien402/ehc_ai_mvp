#!/bin/bash
# Run this ON the GPU server to expose Ollama on all interfaces
# Usage: bash scripts/setup_remote_ollama.sh

set -e

echo "Configuring Ollama to accept remote connections..."

# Create systemd override to set OLLAMA_HOST=0.0.0.0
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

systemctl daemon-reload
systemctl restart ollama
sleep 3

if curl -sf http://localhost:11434/api/tags > /dev/null; then
    echo "✓ Ollama is running and accessible"
    echo "  Models available:"
    curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print('  -', m['name']) for m in json.load(sys.stdin)['models']]"
else
    echo "✗ Ollama not responding"
fi

# Open firewall port if ufw is active
if command -v ufw &>/dev/null && ufw status | grep -q "active"; then
    ufw allow 11434/tcp
    echo "✓ ufw: port 11434 opened"
fi

echo ""
echo "✓ Done. Ollama now accepts connections on port 11434 from any host."
