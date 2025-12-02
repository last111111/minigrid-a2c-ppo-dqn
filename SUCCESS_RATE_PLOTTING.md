# 成功率可视化功能

## 功能说明

训练过程中会自动生成并保存**成功率变化曲线图**，帮助你实时监控训练进度。

## 使用方法

### 1. 安装依赖（如果还没安装）

```bash
pip install matplotlib>=3.0
```

或者安装所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行训练

```bash
bash train_ppo.sh
```

### 3. 查看成功率图表

训练过程中，每次保存模型时（根据 `--save-interval` 参数），会自动生成成功率曲线图：

**图表位置：** `storage/<model_name>/success_rate.png`

例如：
- `storage/ppo-MultiRoom-N4-S5-50000/success_rate.png`
- `storage/ppo-MultiRoom-N4-S5-100000/success_rate.png`

### 4. 图表更新频率

- 默认每 10 个 update 保存一次（`--save-interval 10`）
- 每次保存时都会更新成功率图表
- 图表显示从训练开始到当前的完整成功率曲线

## 图表说明

### 图表内容

- **横轴（X轴）**：训练更新次数 (Update)
- **纵轴（Y轴）**：成功率 (Success Rate)，范围 0-100%
- **蓝色实线**：实际成功率变化曲线
- **红色虚线**：50% 参考线
- **绿色虚线**：80% 参考线

### 成功率计算

- 基于最近 100 个完成的 episode 计算
- 如果 episode 的总回报 > 0，则算成功（到达目标）
- 成功率 = 成功次数 / 总 episode 数

### 示例

训练日志输出：
```
U 100 | E 1600 | F 204800 | ... | SR 45.00% (45/100)
```
表示：
- 最近 100 个 episode 中有 45 个成功
- 成功率为 45%

对应的图表会在 Y 轴上显示 0.45（45%）

## 实时监控

### 方法 1：定期查看图片

```bash
# Linux/Mac
watch -n 60 display storage/<model_name>/success_rate.png

# 或者直接在文件管理器中打开图片，定期刷新
```

### 方法 2：使用 Python 脚本实时显示

创建 `monitor_training.py`：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

model_dir = "storage/ppo-MultiRoom-N4-S5-50000"
plot_path = f"{model_dir}/success_rate.png"

plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))

while True:
    if os.path.exists(plot_path):
        ax.clear()
        img = mpimg.imread(plot_path)
        ax.imshow(img)
        ax.axis('off')
        plt.pause(10)  # 每10秒刷新一次
    else:
        print(f"Waiting for plot at {plot_path}...")
        time.sleep(5)
```

运行：
```bash
python monitor_training.py
```

## 训练参数调整

如果想更频繁地更新图表，可以调整保存间隔：

```bash
# 原始：每10个update保存一次
python3 -m scripts.train --env MiniGrid-MultiRoom-N4-S5-v0 --algo ppo --model test --frames 50000 --procs 16 --save-interval 10

# 更频繁：每5个update保存一次
python3 -m scripts.train --env MiniGrid-MultiRoom-N4-S5-v0 --algo ppo --model test --frames 50000 --procs 16 --save-interval 5

# 更少：每20个update保存一次
python3 -m scripts.train --env MiniGrid-MultiRoom-N4-S5-v0 --algo ppo --model test --frames 50000 --procs 16 --save-interval 20
```

## 多次训练对比

你可以同时运行多个训练，然后对比不同模型的成功率曲线：

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
models = [
    "ppo-MultiRoom-N4-S5-50000",
    "ppo-MultiRoom-N4-S5-100000",
    "ppo-MultiRoom-N4-S5-150000",
    "ppo-MultiRoom-N4-S5-200000"
]

for ax, model in zip(axes.flat, models):
    img = mpimg.imread(f"storage/{model}/success_rate.png")
    ax.imshow(img)
    ax.set_title(model, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=150)
plt.show()
```

## 故障排除

### 问题 1：图表没有生成

**可能原因：**
- matplotlib 未安装

**解决方法：**
```bash
pip install matplotlib>=3.0
```

### 问题 2：图表显示为空

**可能原因：**
- 训练刚开始，还没有完成足够的 episode

**解决方法：**
- 等待更多 episode 完成
- 查看训练日志，确认 success_count > 0

### 问题 3：无法在服务器上查看图片

**解决方法：**
```bash
# 将图片下载到本地
scp user@server:/path/to/storage/model/success_rate.png ./

# 或者使用 tensorboard（图表也会自动上传到 tensorboard）
tensorboard --logdir storage/
```

## 高级功能

### 自定义图表样式

可以修改 `scripts/train.py` 中的 `plot_success_rate` 函数来自定义图表样式：

```python
def plot_success_rate(updates, success_rates, model_dir):
    plt.figure(figsize=(12, 8))  # 调整图表大小
    plt.plot(updates, success_rates, 'b-', linewidth=3)  # 调整线条粗细

    # 添加更多参考线
    plt.axhline(y=0.9, color='purple', linestyle='--', alpha=0.5, label='90%')

    # 自定义颜色和样式
    plt.style.use('seaborn')  # 使用不同的样式

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # 保存高分辨率版本
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
```
