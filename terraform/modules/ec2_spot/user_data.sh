#!/bin/bash
set -euxo pipefail

# System update & dependencies
apt-get update -y
apt-get install -y python3-pip python3-venv git curl

# Set working directory
WORKDIR="/opt/sentinelops"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Clone or pull SentinelOps repository
if [ ! -d "$WORKDIR/.git" ]; then
    git clone https://github.com/kyrtyy/SentinelOps.git .
else
    git pull origin main
fi

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Create systemd service for SentinelOps Model Server
cat <<EOF > /etc/systemd/system/sentinelops.service
[Unit]
Description=SentinelOps Autonomous SRE Model Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$WORKDIR
ExecStart=$WORKDIR/.venv/bin/python $WORKDIR/serve_api.py --model-path outputs/sentinelops-merged --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sentinelops
systemctl start sentinelops
