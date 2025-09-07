# 物体识别与掩码生成工具

基于YOLO-World和SAM（Segment Anything Model）的交互式物体识别和掩码生成工具。

## 功能特性

- 🎯 **智能物体检测**：使用YOLO-World模型检测图像中的物体
- 🎨 **精确掩码生成**：使用SAM模型为选定物体生成精确分割掩码
- 🖱️ **交互式选择**：用户可以选择要处理的图像和目标物体
- 📊 **多种输出格式**：提供检测结果、掩码、叠加图像等多种输出
- 📍 **中心点定位**：自动计算并标记物体中心点

## 环境要求

### Python版本
- Python 3.8 或更高版本

### 必需的依赖库

```bash
pip install ultralytics torch opencv-python numpy
```

### 依赖库详细说明

| 库名 | 版本要求 | 用途 |
|-----|----------|------|
| `ultralytics` | 最新版 | YOLO模型推理 |
| `torch` | ≥1.7.0 | PyTorch深度学习框架 |
| `opencv-python` | ≥4.5.0 | 图像处理 |
| `numpy` | ≥1.19.0 | 数值计算 |

## 模型文件

⚠️ **重要提示**：由于模型文件较大，已在`.gitignore`中排除。使用前需要下载以下模型文件：

### 必需的模型文件

1. **YOLO-World模型** (`yolov8s-world.pt`)
   - 下载地址：[YOLO-World官方releases](https://github.com/AILab-CVC/YOLO-World/releases)
   - 大小：约27MB
   - 用途：物体检测

2. **SAM模型** (`sam_b.pt`)
   - 下载地址：[SAM官方releases](https://github.com/facebookresearch/segment-anything#model-checkpoints)
   - 大小：约375MB
   - 用途：物体分割

### 模型文件放置位置

将下载的模型文件放置在项目根目录下：

```
mask_generate/
├── advanced_mask_demo.py
├── yolov8s-world.pt      # YOLO-World模型
├── sam_b.pt              # SAM模型
├── your_image.jpg        # 测试图像
└── README.md
```

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/HuangJunkai2023/mask_generate.git
cd mask_generate
```

### 2. 安装依赖

```bash
pip install ultralytics torch opencv-python numpy
```

### 3. 下载模型文件

- 下载 `yolov8s-world.pt` 并放置在项目根目录
- 下载 `sam_b.pt` 并放置在项目根目录

### 4. 准备测试图像

将要处理的图像文件（支持格式：.jpg, .jpeg, .png, .bmp, .tiff, .webp）放置在项目根目录下。

## 使用方法

### 运行程序

```bash
python advanced_mask_demo.py
```

### 操作流程

1. **依赖检查**：程序启动时会自动检查所有依赖库是否正确安装
2. **选择图像**：从项目目录中检测到的图像文件中选择要处理的图像
3. **物体检测**：程序使用YOLO-World检测图像中的物体
4. **选择目标**：从检测到的物体列表中选择要生成掩码的目标物体
5. **掩码生成**：使用SAM为选定物体生成精确掩码
6. **结果输出**：程序会生成多种格式的输出文件

### 输出文件说明

程序会在 `advanced_test_results/` 目录下生成以下文件：

| 文件类型 | 命名格式 | 说明 |
|---------|----------|------|
| YOLO检测结果 | `yolo_detection_[图像名].jpg` | 带有检测框的原图 |
| SAM掩码 | `sam_mask_[物体类别]_[图像名].png` | 黑白掩码图像 |
| 叠加图像 | `sam_overlay_[物体类别]_[图像名].jpg` | 原图叠加彩色掩码 |
| 分割结果 | `sam_segmented_[物体类别]_[图像名].jpg` | 只显示目标物体，背景为黑色 |
| 中心点标记 | `sam_center_[物体类别]_[图像名].jpg` | 标记物体中心点的图像 |

## 示例

### 输入图像
- `fruit.jpg` - 水果图像
- `color.png` - 彩色物体图像
- `fang.jpg` - 其他测试图像

### 运行示例

```bash
$ python advanced_mask_demo.py

交互式物体识别和掩码生成
==================================================
=== 依赖库检查结果 ===
已安装的库:
  ✓ opencv-python: 4.8.1
  ✓ torch: 2.1.0
  ✓ numpy: 1.24.3
  ✓ ultralytics: 8.0.196

所有依赖库已安装完毕！

在目录中找到 3 个图片文件:
  1. color.png (1.16 MB)
  2. fang.jpg (0.12 MB)
  3. fruit.jpg (0.14 MB)

请选择要处理的图像 (1-3) 或按回车退出: 3

============================================================
处理图像: fruit.jpg
============================================================
处理图像: d:\mask_generate\fruit.jpg
图像尺寸: (480, 640, 3)
正在加载 YOLO-World 模型...
YOLO检测结果已保存: advanced_test_results/yolo_detection_fruit.jpg
检测到 2 个物体:
  1. apple (置信度: 0.876)
  2. orange (置信度: 0.743)

请选择要生成掩码的物体 (输入 1-2 的数字):
可选择的物体:
  1. apple (置信度: 0.876)
  2. orange (置信度: 0.743)

请输入数字选择物体 (按回车退出): 1

您选择了: apple (置信度: 0.876)
正在初始化 SAM 模型...
正在为选中的 apple 生成SAM掩码...

✓ SAM掩码已保存: advanced_test_results/sam_mask_apple_fruit.png
✓ 叠加图像已保存: advanced_test_results/sam_overlay_apple_fruit.jpg
✓ 分割结果已保存: advanced_test_results/sam_segmented_apple_fruit.jpg
✓ 物体中心点: (320, 240)
✓ 中心点标记图已保存: advanced_test_results/sam_center_apple_fruit.jpg

✓ 成功处理图像: fruit.jpg
```

## 故障排除

### 常见问题

1. **模型文件未找到**
   ```
   错误: 找不到 yolov8s-world.pt 或 sam_b.pt
   解决: 确保模型文件已下载并放置在正确位置
   ```

2. **依赖库安装失败**
   ```bash
   # 使用conda安装（推荐）
   conda install pytorch torchvision torchaudio -c pytorch
   pip install ultralytics opencv-python
   
   # 或使用清华源安装
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics torch opencv-python numpy
   ```

3. **CUDA支持（可选）**
   ```bash
   # 如果有NVIDIA GPU，可以安装CUDA版本的PyTorch以加速推理
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 性能优化

- **GPU加速**：如果有NVIDIA GPU，程序会自动使用CUDA加速
- **模型缓存**：模型首次加载较慢，后续使用会更快
- **图像大小**：较大图像处理时间更长，可以考虑适当缩放

## 技术架构

```
输入图像 → YOLO-World检测 → 用户选择 → SAM分割 → 多格式输出
    ↓           ↓              ↓         ↓         ↓
  图像文件    物体检测框    交互界面   精确掩码   结果文件
```

## 许可证

本项目基于开源许可证发布。具体使用的模型请遵循其各自的许可证：
- YOLO-World: [GPL-3.0 License](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE)
- SAM: [Apache-2.0 License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)

## 贡献

欢迎提交Issues和Pull Requests来改进本项目！

## 更新日志

- **v1.0.0** - 初始版本，支持YOLO检测和SAM分割
- **v1.1.0** - 添加交互式物体选择功能
- **v1.2.0** - 优化输出格式，添加中心点定位

---

如有问题，请在GitHub Issues中反馈。
