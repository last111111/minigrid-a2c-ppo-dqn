# 成功率追踪功能说明

## 功能概述

已为 PPO 和 A2C 算法添加了成功率追踪功能，包括：
1. **训练过程中实时显示**最近100次的成功率
2. **训练结束后自动生成**成功率变化图

## 修改的文件

### 1. `torch_ac/algos/base.py`
- 添加 `self.log_success` 列表追踪每个episode的成功/失败
- 在episode结束时记录成功状态（reward > 0 表示成功）
- 在返回的logs中添加 `success_per_episode` 字段

### 2. `scripts/train.py`
- 导入 matplotlib 用于绘图
- 添加成功率统计和显示逻辑
- 训练结束后生成成功率图表

## 使用方法

### 正常训练（会自动追踪成功率）

```bash
python scripts/train.py \
    --algo ppo \
    --env MiniGrid-FourRooms-v0 \
    --episodes 1000 \
    --procs 16
```

### 训练日志输出示例

```
U 50 | E 800 | F 102400 | FPS 1234 | D 83 | rR:μσmM 0.65 0.12 0.10 0.85 | F:μσmM 45.2 12.3 15 89 | H 1.234 | V 0.456 | pL 0.023 | vL 0.015 | ∇ 0.234 | SR100 85.0% | SRb 87.5%
```

**新增字段说明**：
- `SR100`: Success Rate last 100 episodes（最近100次episode的成功率）
- `SRb`: Success Rate batch（当前批次的成功率）

## 输出文件

训练完成后，会在模型目录生成以下文件：

```
storage/{model_name}/
├── status.pt                    # Checkpoint
├── log.txt                      # 文本日志
├── log.csv                      # CSV日志（新增 success_rate_100, success_rate_batch 列）
├── success_rate_plot.png        # 成功率变化图 ⭐ 新增
└── events.out.tfevents.*       # TensorBoard日志
```

## 成功率图表说明

### 左图：滚动平均成功率
- X轴：Episode数
- Y轴：成功率（0-1）
- 蓝色曲线：最近100次episode的滚动平均成功率
- 红色虚线：90%成功率参考线

### 右图：分段成功率柱状图
- X轴：Episode数
- Y轴：成功率（0-1）
- 紫色柱状图：每100个episode为一组的平均成功率

## 训练结束输出示例

```
============================================================
Training completed! Generating success rate plot...

============================================================
SUCCESS RATE SUMMARY
============================================================
Total Episodes: 1600
Overall Success Rate: 78.44%
Final 100 Episodes Success Rate: 92.00%
Success rate plot saved to: storage/MiniGrid-FourRooms-v0_ppo_seed1_25-12-01-15-30-45/success_rate_plot.png
============================================================
```

## 成功判定标准

**成功**: episode结束时 reward > 0（到达目标）
**失败**: episode结束时 reward = 0（超时或掉入岩浆）

这适用于标准的 MiniGrid 环境，因为：
- 到达目标：reward = 1 - 0.9 × (步数/最大步数) ∈ (0, 1]
- 失败：reward = 0

## 测试脚本

运行快速测试（200个episodes）：

```bash
bash test_success_rate.sh
```

或在Windows上：

```bash
python scripts/train.py --algo ppo --env MiniGrid-Empty-5x5-v0 --model test_sr --episodes 200 --procs 4
```

## 查看历史成功率（从CSV日志）

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
df = pd.read_csv('storage/{model_name}/log.csv')

# 查看成功率变化
plt.figure(figsize=(10, 5))
plt.plot(df['episodes'], df['success_rate_100'])
plt.xlabel('Episodes')
plt.ylabel('Success Rate (Last 100)')
plt.title('Success Rate over Training')
plt.grid(True)
plt.savefig('custom_success_plot.png')
```

## 注意事项

1. **仅适用于 PPO 和 A2C**：DQN 有自己的成功追踪机制
2. **成功率基于奖励**：如果环境的奖励机制不同，可能需要调整判断逻辑
3. **图表生成**：需要安装 matplotlib（已添加到导入中）
4. **性能影响**：成功率追踪对训练速度几乎无影响

## 依赖要求

确保安装了 matplotlib：

```bash
pip install matplotlib
```

## 自定义成功判定

如果需要修改成功判定逻辑，编辑 `torch_ac/algos/base.py` 第173-174行：

```python
# 当前逻辑：reward > 0 表示成功
is_success = self.log_episode_return[i].item() > 0

# 自定义示例1：更高的阈值
is_success = self.log_episode_return[i].item() > 0.5

# 自定义示例2：基于步数
is_success = (self.log_episode_return[i].item() > 0 and
              self.log_episode_num_frames[i].item() < 50)
```

## 技术实现细节

### BaseAlgo 修改
1. 初始化时创建 `self.log_success = []`
2. 每次episode结束时记录成功/失败（0或1）
3. 在 `collect_experiences()` 返回的logs中包含成功记录
4. 成功历史不会被重置，保留完整训练历史

### train.py 修改
1. 使用 `collections.deque(maxlen=100)` 维护最近100次记录
2. 每次日志输出计算滚动平均
3. 训练结束调用 matplotlib 生成双面板图表
4. 保存为高分辨率PNG（300 DPI）

## 故障排除

### 问题1：没有显示成功率
**原因**：可能是旧的checkpoint导致
**解决**：删除 `storage/{model_name}/` 重新训练

### 问题2：图表没有生成
**原因**：matplotlib 未安装或训练未正常结束
**解决**：
```bash
pip install matplotlib
# 确保训练自然结束（达到 --episodes 设定值）
```

### 问题3：成功率始终为0
**原因**：环境太难或训练不足
**解决**：
- 尝试更简单的环境（如 MiniGrid-Empty-5x5-v0）
- 增加训练episodes
- 调整超参数（学习率、熵系数等）

## 贡献者

本功能在原有代码基础上添加，保持了代码的向后兼容性。
