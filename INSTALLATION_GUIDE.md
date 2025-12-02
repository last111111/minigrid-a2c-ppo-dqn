# 安装指南 / Installation Guide

## 快速安装 / Quick Installation

### 方法 1: 使用自动安装脚本（推荐）

#### Windows:
双击运行或在命令行执行：
```cmd
install_dependencies.bat
```

#### Linux/Mac:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

---

### 方法 2: 手动安装

#### 步骤 1: 安装 PyTorch（带 CUDA 支持）

**CUDA 11.3 版本**（推荐，适用于较新的 NVIDIA GPU）:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

**CPU 版本**（如果没有 NVIDIA GPU）:
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

**其他 CUDA 版本**:
- CUDA 11.6: 将 `cu113` 改为 `cu116`
- CUDA 10.2: 将 `cu113` 改为 `cu102`

访问 https://pytorch.org/ 查看更多版本。

#### 步骤 2: 安装项目依赖

```bash
pip install -r requirements.txt
```

这将安装：
- tensorboard
- tensorboardX
- numpy
- gym
- matplotlib（用于成功率绘图）
- array2gif
- progress

#### 步骤 3: 安装 minigrid 环境

```bash
cd minigrid
pip install -e .
cd ..
```

#### 步骤 4: 安装 torch-ac 库

```bash
cd torch-ac
pip install -e .
cd ..
```

---

## 验证安装

运行以下命令验证安装：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import gym_minigrid; print('MiniGrid: OK')"
python -c "import torch_ac; print('torch-ac: OK')"
python -c "import matplotlib; print('matplotlib: OK')"
```

预期输出：
```
PyTorch: 1.12.1+cu113
CUDA available: True
MiniGrid: OK
torch-ac: OK
matplotlib: OK
```

---

## 快速测试

安装完成后，运行成功率追踪测试：

```bash
python quick_test_success_rate.py
```

这将：
- 训练一个简单的 PPO 智能体（约 2-3 分钟）
- 验证所有功能正常工作
- 生成成功率图表

---

## 常见问题 / FAQ

### Q1: 安装 PyTorch 时出错
**A**: 确保 pip 是最新版本：
```bash
pip install --upgrade pip
```

### Q2: "CUDA not available" 或找不到 GPU
**A**: 检查以下几点：
1. 确保安装了 NVIDIA 驱动程序
2. 安装了对应的 CUDA 版本（如 CUDA 11.3）
3. 使用了正确的 PyTorch CUDA 版本

如果没有 GPU，使用 CPU 版本的 PyTorch 也可以训练（速度较慢）。

### Q3: gym 版本不兼容
**A**: 本项目需要 gym 0.23.1。如果已安装其他版本：
```bash
pip uninstall gym
pip install gym==0.23.1
```

### Q4: 找不到 gym_minigrid
**A**: 确保已经安装 minigrid 包：
```bash
cd minigrid
pip install -e .
```

### Q5: matplotlib 导入错误
**A**: 安装 matplotlib：
```bash
pip install matplotlib
```

### Q6: Windows 上 bash 脚本无法运行
**A**: 使用 `.bat` 版本的脚本：
```cmd
install_dependencies.bat
```

或者使用 Git Bash / WSL 运行 `.sh` 脚本。

---

## 系统要求

### 最低要求:
- Python 3.7+
- 8GB RAM
- 可选：NVIDIA GPU（用于加速训练）
- 可选：CUDA 11.3+

### 推荐配置:
- Python 3.8 或 3.9
- 16GB RAM
- NVIDIA GPU（GTX 1060 或更高）
- CUDA 11.3
- 10GB 磁盘空间

---

## 下一步

安装完成后，查看以下文档：

1. **README.md** - 项目概述和基本使用
2. **SUCCESS_RATE_TRACKING.md** - 成功率追踪功能说明
3. **训练示例**:
   ```bash
   # PPO 训练
   python scripts/train.py --algo ppo --env MiniGrid-FourRooms-v0 --episodes 1000

   # 测试成功率追踪
   python quick_test_success_rate.py
   ```

---

## 故障排除

如果遇到问题：

1. **检查 Python 版本**:
   ```bash
   python --version  # 应该是 3.7+
   ```

2. **检查 pip 版本**:
   ```bash
   pip --version  # 应该是 20.0+
   ```

3. **清理旧的安装**:
   ```bash
   pip uninstall torch torchvision torchaudio gym gym-minigrid torch-ac -y
   ```
   然后重新运行安装脚本。

4. **使用虚拟环境**（推荐）:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate

   # 然后运行安装脚本
   ```

---

## 更新依赖

如果需要更新已安装的包：

```bash
pip install --upgrade -r requirements.txt
```

---

祝您使用愉快！如有问题，请参考项目文档或提交 issue。
