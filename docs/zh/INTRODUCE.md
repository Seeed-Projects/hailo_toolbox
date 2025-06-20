# Hailo Tool Box - 智能视觉推理工具箱

## 项目概述

Hailo Tools 是一个专为 Hailo AI 处理器设计的综合性深度学习模型转换和推理工具箱。本项目旨在简化基于 Hailo 设备的 AI 应用开发流程，为开发者提供从模型转换到部署推理的一站式解决方案。通过高度模块化的架构设计和注册机制，用户可以轻松扩展和定制功能，快速验证和部署各种视觉AI模型。

## 🚀 核心特性

### 模型支持
- **多格式兼容**: 支持 Hailo HEF 格式和 ONNX 格式模型
- **模型转换**: 提供完整的模型转换工具链
- **优化推理**: 针对 Hailo 硬件加速器优化的推理引擎
- **量化支持**: 支持 INT8 量化模型的高效推理

### 视觉任务支持
- **目标检测**: YOLOv8、YOLOv5 等主流检测模型
- **图像分割**: 语义分割和实例分割
- **姿态估计**: 人体关键点检测和姿态分析
- **分类识别**: 图像分类和特征提取

### 多样化输入源
- **文件输入**: 图像文件、视频文件、图像文件夹
- **实时流**: USB 摄像头、IP 摄像头、RTSP 流
- **网络输入**: HTTP/HTTPS 图像和视频流
- **多源并发**: 支持同时处理多个输入源

### 后处理与可视化
- **智能后处理**: 内置多种视觉任务的后处理算法
- **实时可视化**: 支持边界框、分割掩码、关键点的实时渲染
- **结果保存**: 支持推理结果的视频保存和导出
- **性能监控**: 实时 FPS 统计和性能分析

### 扩展性设计
- **注册机制**: 基于注册模式的模块化架构
- **插件系统**: 支持自定义后处理函数和可视化方案
- **配置驱动**: 灵活的配置文件支持
- **开发友好**: 简洁的 API 接口，便于二次开发

## 🏗️ 项目架构

### 模块化设计
本项目采用高度模块化的架构设计，每个功能模块通过注册机制进行管理。主要模块包括：

```
hailo_tools/
├── cli/                    # 命令行接口模块
│   ├── infer.py           # 推理命令行工具
│   └── convert.py         # 转换命令行工具
├── sources/               # 输入源管理模块
│   ├── base.py           # 输入源基类定义
│   ├── file.py           # 文件输入源实现
│   ├── webcam.py         # 摄像头输入源
│   ├── ip_camera.py      # IP摄像头输入源
│   └── multi.py          # 多源管理器
├── process/              # 推理处理模块
│   ├── inference.py      # 推理引擎核心
│   ├── postprocess.py    # 后处理算法库
│   └── visualization.py  # 可视化渲染引擎
└── utils/               # 工具函数模块
    ├── config.py        # 配置管理
    ├── logging.py       # 日志系统
    └── registry.py      # 注册管理器
```

### 核心设计模式

#### 1. 注册机制 (Registry Pattern)
- **模块注册**: 所有功能模块通过注册器进行统一管理
- **动态加载**: 支持运行时动态加载和注册新模块
- **解耦设计**: 模块之间松耦合，便于维护和扩展

#### 2. 工厂模式 (Factory Pattern)
- **源工厂**: 根据输入类型自动创建相应的输入源实例
- **处理器工厂**: 根据模型类型和任务类型创建处理器
- **统一接口**: 提供统一的创建接口，简化使用复杂度

#### 3. 策略模式 (Strategy Pattern)
- **同步策略**: 多源输入的不同同步策略
- **后处理策略**: 不同视觉任务的后处理策略
- **可视化策略**: 多样化的结果可视化方案

## 🛠️ 技术栈

### 核心依赖
- **Python 3.8+**: 主要开发语言
- **OpenCV 4.5+**: 计算机视觉处理库
- **NumPy**: 数值计算和数组操作
- **ONNX Runtime**: ONNX 模型推理引擎
- **Hailo SDK**: Hailo 硬件加速支持

### 开发工具
- **Click**: 命令行接口框架
- **PyYAML**: 配置文件解析
- **Tqdm**: 进度条显示
- **Pillow**: 图像处理扩展
- **Requests**: HTTP 请求处理

### 测试框架
- **Pytest**: 单元测试框架
- **Coverage**: 代码覆盖率统计
- **Mock**: 测试模拟工具

## 📋 使用场景

### 工业应用
- **质量检测**: 工业产品缺陷检测和质量控制
- **安全监控**: 智能监控系统和异常检测
- **自动化生产**: 机器人视觉引导和控制

### 商业应用
- **零售分析**: 客流统计和行为分析
- **智能交通**: 车辆检测和交通监控
- **医疗影像**: 医学图像分析和辅助诊断

### 教育研究
- **算法验证**: 深度学习模型效果验证
- **原型开发**: 快速原型构建和测试
- **学术研究**: 计算机视觉算法研究平台

## 🔧 快速开始

### 安装环境
```bash
# 克隆项目
git clone https://github.com/your-repo/hailo_tools.git
cd hailo_tools

# 安装依赖
pip install -e .

# 验证安装
hailo-toolbox --version
```

### 基础使用
```bash
# 运行目标检测
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source video.mp4

# 实时摄像头推理
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source 0 --show

# 批量图像处理
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source images/ --save
```

### 自定义扩展
```python
# 注册自定义后处理函数
from hailo_tools.utils.registry import register_callback

@register_callback("custom_detection")
def custom_detection_callback(outputs, frame, model_info):
    """
    Custom detection postprocessing function
    
    Args:
        outputs: Model inference outputs
        frame: Input frame
        model_info: Model metadata
    
    Returns:
        Processed frame with visualizations
    """
    # 实现自定义后处理逻辑
    processed_frame = apply_custom_processing(outputs, frame)
    return processed_frame
```

## 📊 性能特性

### 高性能推理
- **硬件加速**: 充分利用 Hailo AI 处理器的并行计算能力
- **内存优化**: 智能内存管理，减少内存占用和拷贝开销
- **批处理支持**: 支持批量推理以提升整体吞吐量

### 并发处理
- **多线程架构**: 输入读取、推理处理、结果输出独立线程
- **异步处理**: 异步 I/O 操作避免阻塞等待
- **负载均衡**: 智能任务调度和负载分配

### 实时性能
- **低延迟**: 优化的推理流水线，最小化端到端延迟
- **高帧率**: 支持高帧率视频流的实时处理
- **自适应调节**: 根据硬件性能自动调节处理参数

## 🌐 平台兼容性

### 操作系统支持
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **Windows**: Windows 10/11 (64-bit)
- **嵌入式 Linux**: 支持 ARM 架构的嵌入式平台

### 硬件要求
- **CPU**: x86_64 或 ARM64 架构
- **内存**: 最小 4GB RAM，推荐 8GB+
- **存储**: 最小 10GB 可用空间
- **Hailo 设备**: Hailo-8 或 Hailo-15 AI 处理器

## 📖 文档导航

- [快速入门指南](GET_STAR.md) - 新手入门教程
- [开发文档](DEV.md) - 详细开发指南  
- [输入源文档](SOURCE.md) - 输入源使用说明
- [推理教程](INFERENCE.md) - 推理功能详解
- [模型转换](CONVERT.md) - 模型转换指南

## 🤝 贡献指南

我们欢迎社区贡献！请参考以下方式参与项目：

1. **提交 Issue**: 报告 Bug 或提出功能请求
2. **代码贡献**: Fork 项目并提交 Pull Request
3. **文档完善**: 改进文档和示例代码
4. **测试用例**: 添加测试用例和性能基准

## 📄 开源协议

本项目采用 [MIT License](../LICENSE) 开源协议，允许自由使用、修改和分发。

## 📞 技术支持

- **GitHub Issues**: [项目 Issues 页面](https://github.com/your-repo/hailo_tools/issues)
- **邮件支持**: your.email@example.com
- **技术文档**: [在线文档站点](https://your-docs-site.com)

---

*Hailo Tools - 让 AI 推理更简单、更高效！*