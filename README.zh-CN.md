# YOLO-By-C++

[English](README.md) | **简体中文**

> 山东大学 控制科学与工程学院 2024 年《面向对象的程序设计》课程作业

基于 **OpenCV DNN + C++** 的 YOLO 目标检测示例，包含两个 Visual Studio 工程：

- **YOLO by C++ using pictures** —— 对**静态图片**做目标检测
- **YOLO by C++ using video** —— 调用**电脑摄像头**做实时视频目标检测

支持 YOLOv3 / YOLOv4 / YOLOv4-tiny / YOLObile / YOLO-Fastest 五种模型，纯 CPU 推理，无需 CUDA。
推理与后处理（NMS、置信度过滤、耗时统计）全部由 OpenCV `dnn` 模块完成。

运行方式：打开对应 `.sln` 编译后直接运行——图片工程会检测 `person.jpg` 并弹窗显示结果，
视频工程会打开本机摄像头做实时检测，窗口内按任意键退出。

> 仓库不含演示视频与模型权重，克隆后请按「快速开始 → 放入模型权重」准备模型。

---

## 目录结构

```
YOLO-By-C++/
├── YOLO by C++ using pictures/          # 图片检测工程
│   ├── YOLO by OpenCV.sln               # 解决方案
│   └── YOLO by OpenCV/
│       ├── main_yolo.cpp                # 源码入口（YOLO 类实现 + main）
│       ├── yolo.h                       # 网络配置与 YOLO 类声明
│       ├── coco.names                   # COCO 80 类标签
│       ├── bus.jpg / dog.jpg / person.jpg / zidane.jpg   # 测试图片
│       ├── yolov3/yolov3.cfg
│       ├── yolov4/yolov4.cfg, yolov4-tiny.cfg
│       ├── yolobile/csdarknet53s-panet-spp.cfg
│       └── yolo-fastest/yolo-fastest-xl.cfg
│
├── YOLO by C++ using video/             # 摄像头实时检测工程
│   ├── YOLO by C++.sln
│   └── YOLO by C++/
│       ├── YOLO by C++.cpp              # 源码（VideoCapture(0) 打开摄像头）
│       ├── yolo.h
│       ├── coco.names
│       └── （同上 4 个模型的 .cfg）
│
├── archive/                             # 开发过程中的历史版本，仅作参考，不参与编译
├── 项目说明.txt                          # 项目说明（中文）
├── README.md                            # English README（GitHub 默认展示）
└── README.zh-CN.md                      # 中文说明（本文件）
```

> **注意：`.weights` 权重文件未包含在仓库中**（单文件 200MB+，超出 GitHub 100MB 限制）。
> 请按下文「模型权重」一节下载后放入对应目录。

---

## 环境要求

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows 10 / 11 |
| 编译器 | Visual Studio 2022（v143 工具集，需勾选「使用 C++ 的桌面开发」） |
| OpenCV | 4.9.0（`opencv_world` 单库版本，`x64/vc16` 构建） |
| 平台 | 仅支持 **x64**（Win32 配置未配置 OpenCV 路径，请勿使用） |

---

## 快速开始

### 1. 配置 OpenCV 路径

工程中已通过 `$(OpenCV_DIR)` 变量引用 OpenCV，**默认值**为 `D:\OpenCV\opencv\build`。
如果你的 OpenCV 装在别处，设置环境变量即可（无需改工程文件）：

```powershell
# 临时生效（当前终端）
$env:OpenCV_DIR = "C:\opencv\build"

# 永久生效
setx OpenCV_DIR "C:\opencv\build"
```

> `OpenCV_DIR` 指向 OpenCV 安装包的 `build` 目录，其下应有 `include` 和 `x64\vc16\lib`。

### 2. 放入模型权重

从 [Releases](../../releases) 页面下载权重文件，按下面的路径放置（两个工程各放一份）：

```
YOLO by OpenCV/
├── yolov3/yolov3.weights
├── yolov4/yolov4.weights
├── yolov4/yolov4-tiny.weights
├── yolobile/yolobile.weights
└── yolo-fastest/yolo-fastest-xl.weights
```

**只需要用到的模型**即可，不必全部下载。例如只用 YOLO-Fastest，就只放
`yolo-fastest/yolo-fastest-xl.weights`。

### 3. 编译运行

用 VS2022 打开 `.sln`，选择 `Release | x64`，按 `F5` 运行。

⚠️ 运行时若提示找不到 `opencv_world490.dll` / `opencv_world490d.dll`，
把 `%OpenCV_DIR%\x64\vc16\bin` 加入系统 `PATH`，或把对应 dll 复制到 `.exe` 同目录。

### 4. 更换推理模型

模型在 `yolo.h` 的 `yolo_nets[4]` 数组中定义，源码中通过下标选用：

```cpp
YOLO yolo_model(yolo_nets[2]);   // 0=yolov3  1=yolov4  2=yolo-fastest  3=yolobile
```

| 下标 | 模型 | 输入尺寸 | 权重体积 | 速度 | 精度 |
|---|---|---|---|---|---|
| 0 | YOLOv3 | 416×416 | 236 MB | 慢 | 高 |
| 1 | YOLOv4 | 608×608 | 245 MB | 最慢 | 最高 |
| 2 | YOLO-Fastest-XL | 320×320 | 3.5 MB | **最快** | 较低 |
| 3 | YOLObile | 320×320 | 245 MB | 快 | 较高 |

监控摄像头等实时场景建议用 `2`（默认值），单张图片追求精度可用 `1`。

---

## 两个工程的区别

| | 图片工程 | 视频工程 |
|---|---|---|
| 输入 | `imread("person.jpg")` | `VideoCapture(0)` 摄像头 |
| 输出 | 弹窗显示检测结果 | 逐帧实时显示 |
| 源码 | `main_yolo.cpp` | `YOLO by C++.cpp` |
| 换测试图 | 改 `main()` 里的 `imgpath` | 无需修改 |

两个工程的 `yolo.h` 与检测逻辑完全一致，只是 `main()` 中的输入源不同。

### 视频工程：摄像头序号

默认打开 **0 号摄像头**。若机器有多个摄像头或默认索引不对，可用命令行参数指定：

```bat
:: 项目属性 → 调试 → 命令参数 里填 1，或直接命令行运行
"YOLO by C++.exe" 1
```

程序内部会按 `MSMF → DirectShow → 自动` 三种后端、`0~5` 号设备依次尝试，
成功后会打印实际使用的序号与后端，例如 `Camera opened: index=0, backend=DirectShow`。

---

## 模型权重获取

**方式一（推荐）**：本仓库 [Releases](../../releases) 页面的 `weights` 附件。

**方式二**：官方原始发布渠道

- YOLOv3 / YOLOv4：<https://github.com/AlexeyAB/darknet#pre-trained-models>
- YOLO-Fastest：<https://github.com/dog-qiuqiu/Yolo-Fastest>
- YOLObile：<https://github.com/Coobird/YOLObile>

上游教程提供的百度网盘打包（4 个模型文件夹，提取码 `imgu`）：
<https://pan.baidu.com/s/1Kcw-VhuDTRzCtVkaNEDOBg>

---

## 常见问题

**Q：运行报 `global obsensor_uvc_stream_channel.cpp cv::obsensor::getStreamChannelGroup Camera index out of range` + `Cannot open camera`**
OpenCV 4.9 在 Windows 上的典型问题：MSMF 后端打不开默认摄像头，日志里的 `obsensor` 只是它在盲探设备，不是真的没摄像头。
本工程已做自动兜底（依次尝试 MSMF / DirectShow / 自动 后端，并扫描 0~5 号设备），若仍失败请依次检查：

1. 摄像头被占用——相机 App、微信、腾讯会议、浏览器网页等会独占设备；
2. `设置 → 隐私和安全性 → 相机` 中允许「桌面应用访问相机」；
3. 设备管理器 → 摄像头，确认设备存在且未被禁用；
4. 手动指定序号：`"YOLO by C++.exe" 1`。

**Q：编译报 `无法打开源文件 "opencv2/dnn.hpp"`**
`OpenCV_DIR` 没设对。确认该变量指向 `...\opencv\build`，且该目录下存在 `include\opencv2\dnn.hpp`。

**Q：编译报 `LNK2019 无法解析的外部符号`**
链接库不匹配：Debug 配置应链接 `opencv_world490d.lib`，Release 配置应链接 `opencv_world490.lib`（本工程已配好）。

**Q：编译通过但运行闪退**
多半是权重文件路径不对或缺失。`yolo.h` 中的路径是**相对路径**（如 `yolov4/yolov4.weights`），
因此需要在「项目属性 → 调试 → 工作目录」为项目目录，或保证从项目目录启动程序。

**Q：中文注释在 VS 里显示乱码**
源码统一为 UTF-8 with BOM。若已用 GBK 打开，在 VS 中「文件 → 高级保存选项」改回 UTF-8 即可。

---

## 来源与致谢

`yolo.h`、`main_yolo.cpp` 的核心实现来自网络公开教程 `yolov34-cpp-opencv-dnn`
（包含 YOLOv3 / YOLOv4 / YOLO-Fastest / YOLObile 的 C++ 与 Python 两种实现），
本项目在其基础上完成了工程整理、绝对路径可移植化改造、源码编码统一，
并区分出「图片检测」与「摄像头实时检测」两个独立工程。

模型权重版权归各 YOLO 原作者所有，仅用于学习与研究。
