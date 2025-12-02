#!/bin/bash

# ============================================================
# Installation Script for minigrid-a2c-ppo-dqn
# ============================================================

echo "============================================================"
echo "Installing Dependencies for minigrid-a2c-ppo-dqn"
echo "============================================================"
echo ""

# Step 1: Install PyTorch with CUDA 11.3
echo "[1/4] Installing PyTorch with CUDA 11.3 support..."
echo "This may take several minutes..."
echo ""
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi
echo "[1/4] PyTorch installed successfully!"
echo ""

# Step 2: Install requirements.txt
echo "[2/4] Installing dependencies from requirements.txt..."
echo ""
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi
echo "[2/4] Requirements installed successfully!"
echo ""

# Step 3: Install minigrid
echo "[3/4] Installing minigrid package..."
echo ""
cd minigrid
python -m pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install minigrid"
    cd ..
    exit 1
fi
cd ..
echo "[3/4] Minigrid installed successfully!"
echo ""

# Step 4: Install torch-ac
echo "[4/4] Installing torch-ac package..."
echo ""
cd torch-ac
python -m pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install torch-ac"
    cd ..
    exit 1
fi
cd ..
echo "[4/4] torch-ac installed successfully!"
echo ""

# Install matplotlib for success rate plotting
echo "[Bonus] Installing matplotlib for success rate plotting..."
python -m pip install matplotlib
echo ""

echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "All dependencies have been installed successfully."
echo ""
echo "You can now run training with:"
echo "  python scripts/train.py --algo ppo --env MiniGrid-FourRooms-v0 --episodes 1000"
echo ""
echo "To test success rate tracking:"
echo "  python quick_test_success_rate.py"
echo ""
