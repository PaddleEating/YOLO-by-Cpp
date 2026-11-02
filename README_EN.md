# YOLO-By-C++

[简体中文](README.md) | **English**

Real-time object detection with **YOLO + OpenCV DNN in C++**, shipped as two Visual Studio projects:

- **YOLO by C++ using pictures** — object detection on **static images**
- **YOLO by C++ using video** — **webcam** object detection on a live video stream

Five model variants are supported: YOLOv3 / YOLOv4 / YOLOv4-tiny / YOLObile / YOLO-Fastest.
Everything runs on CPU — no CUDA required. Inference and post-processing (NMS, confidence
thresholding, timing) are handled entirely by the OpenCV `dnn` module.

How to run: build the `.sln` and launch it. The pictures project detects `person.jpg` and
shows the result in a window; the video project opens your webcam and runs detection live —
press any key while the window is focused to quit.

> **The repository ships without demo video or model weights.** After cloning, follow
> "Quick Start → 2. Add the model weights" to prepare the models.

---

## Repository layout

```
YOLO-By-C++/
├── YOLO by C++ using pictures/          # Image detection project
│   ├── YOLO by OpenCV.sln               # Solution
│   └── YOLO by OpenCV/
│       ├── main_yolo.cpp                # Entry point (YOLO class impl + main)
│       ├── yolo.h                       # Net config & YOLO class declaration
│       ├── coco.names                   # COCO 80 class labels
│       ├── bus.jpg / dog.jpg / person.jpg / zidane.jpg   # Sample images
│       ├── yolov3/yolov3.cfg
│       ├── yolov4/yolov4.cfg, yolov4-tiny.cfg
│       ├── yolobile/csdarknet53s-panet-spp.cfg
│       └── yolo-fastest/yolo-fastest-xl.cfg
│
├── YOLO by C++ using video/             # Webcam detection project
│   ├── YOLO by C++.sln
│   └── YOLO by C++/
│       ├── YOLO by C++.cpp              # Source (opens camera via VideoCapture)
│       ├── yolo.h
│       ├── coco.names
│       └── (the same 4 model .cfg files)
│
├── archive/                             # Historical versions, reference only, not compiled
├── 项目说明.txt                          # Project notes (Chinese)
├── README.md                            # Chinese README
└── README_EN.md                         # English README (this file)
```

> **Note: `.weights` files are not tracked in this repository** — each is 200 MB+, beyond
> GitHub's 100 MB per-file limit. Download them from the Releases page and drop them into
> the directories listed below.

---

## Requirements

| Item | Version |
|---|---|
| OS | Windows 10 / 11 |
| Compiler | Visual Studio 2022 (v143 toolset, "Desktop development with C++" workload) |
| OpenCV | 4.9.0 (`opencv_world` single-lib build, `x64/vc16`) |
| Platform | **x64 only** (the Win32 configurations have no OpenCV paths configured — do not use them) |

---

## Quick start

### 1. Point the project at your OpenCV

The projects reference OpenCV through the `$(OpenCV_DIR)` variable, whose **default** value is
`D:\OpenCV\opencv\build`. If yours lives elsewhere, just set an environment variable — no need to
touch the project files:

```powershell
# Current session only
$env:OpenCV_DIR = "C:\opencv\build"

# Persist for the user account
setx OpenCV_DIR "C:\opencv\build"
```

> `OpenCV_DIR` must point to the `build` directory of your OpenCV package, which should contain
> `include` and `x64\vc16\lib`.

### 2. Add the model weights

Download the weights from the [Releases](../../releases) page and place them as follows
(one copy per project — the two projects do not share files):

```
YOLO by OpenCV/
├── yolov3/yolov3.weights
├── yolov4/yolov4.weights
├── yolov4/yolov4-tiny.weights
├── yolobile/yolobile.weights
└── yolo-fastest/yolo-fastest-xl.weights
```

You only need the models you actually use. If you only want YOLO-Fastest, place just
`yolo-fastest/yolo-fastest-xl.weights`.

### 3. Build and run

Open the `.sln` in VS2022, select `Release | x64`, press `F5`.

⚠️ If launching complains about a missing `opencv_world490.dll` / `opencv_world490d.dll`, add
`%OpenCV_DIR%\x64\vc16\bin` to your system `PATH`, or copy the matching DLL next to the `.exe`.

### 4. Switch the inference model

Models are declared in the `yolo_nets[4]` array in `yolo.h` and selected by index in the source:

```cpp
YOLO yolo_model(yolo_nets[2]);   // 0=yolov3  1=yolov4  2=yolo-fastest  3=yolobile
```

| Index | Model | Input size | Weights | Speed | Accuracy |
|---|---|---|---|---|---|
| 0 | YOLOv3 | 416×416 | 236 MB | Slow | High |
| 1 | YOLOv4 | 608×608 | 245 MB | Slowest | Highest |
| 2 | YOLO-Fastest-XL | 320×320 | 3.5 MB | **Fastest** | Lower |
| 3 | YOLObile | 320×320 | 245 MB | Fast | High |

Use `2` (the default) for real-time camera streams; use `1` when you need maximum accuracy
on a single image.

---

## Pictures vs. video project

| | Pictures project | Video project |
|---|---|---|
| Input | `imread("person.jpg")` | Webcam via `VideoCapture` |
| Output | Single result window | Frame-by-frame live display |
| Source file | `main_yolo.cpp` | `YOLO by C++.cpp` |
| Change the test image | Edit `imgpath` in `main()` | Not applicable |

Both projects share an identical `yolo.h` and detection logic; only the input source in
`main()` differs.

### Video project: choosing a camera

The default is **camera index 0**. If your machine has multiple cameras or the default index is
wrong, pass an index on the command line:

```bat
:: Set it under Project Properties -> Debugging -> Command Arguments, or run directly
"YOLO by C++.exe" 1
```

Internally the program tries the `MSMF → DirectShow → Auto` backends against devices `0`–`5`
in turn, and prints the combination that actually worked, e.g.
`Camera opened: index=0, backend=DirectShow`.

---

## Where to get the weights

**Option 1 (recommended):** the `weights` attachments on this repository's [Releases](../../releases) page.

**Option 2:** the upstream publishers

- YOLOv3 / YOLOv4: <https://github.com/AlexeyAB/darknet#pre-trained-models>
- YOLO-Fastest: <https://github.com/dog-qiuqiu/Yolo-Fastest>
- YOLObile: <https://github.com/Coobird/YOLObile>

The original tutorial also distributes all four model folders as one Baidu Pan archive
(code `imgu`): <https://pan.baidu.com/s/1Kcw-VhuDTRzCtVkaNEDOBg>

---

## Troubleshooting

**Q: Runtime error `cv::obsensor::getStreamChannelGroup Camera index out of range` followed by `Cannot open camera`**

A well-known OpenCV 4.9 issue on Windows: the MSMF backend fails to open the default camera. The
`obsensor` lines in the log are just the backend probing for devices — they do **not** mean no
camera is present. This project already falls back automatically (MSMF / DirectShow / Auto backends,
scanning devices 0–5). If it still fails, check in order:

1. The camera is busy — the Camera app, WeChat, Tencent Meeting, a browser tab, etc. can hold it exclusively;
2. `Settings → Privacy & security → Camera` allows "Let desktop apps access your camera";
3. Device Manager → Cameras: the device exists and is not disabled;
4. Try an explicit index: `"YOLO by C++.exe" 1`.

**Q: Build error `cannot open source file "opencv2/dnn.hpp"`**

`OpenCV_DIR` is wrong. Make sure it points to `...\opencv\build` and that
`include\opencv2\dnn.hpp` exists inside it.

**Q: Build error `LNK2019 unresolved external symbol`**

Library mismatch: the Debug configuration must link `opencv_world490d.lib`, Release must link
`opencv_world490.lib` (already configured in this project).

**Q: Builds fine but crashes immediately at runtime**

Usually a wrong or missing weights path. Paths in `yolo.h` are **relative**
(e.g. `yolov4/yolov4.weights`), so the working directory must be the project directory —
it already is when you press F5 in Visual Studio.

**Q: Chinese comments show up as garbled text in Visual Studio**

All sources are UTF-8 with BOM. If a file was opened as GBK, use `File → Advanced Save Options`
and switch it back to UTF-8.

---

## Credits

The core of `yolo.h` and `main_yolo.cpp` comes from the public tutorial project
`yolov34-cpp-opencv-dnn` (C++ and Python implementations of YOLOv3 / YOLOv4 / YOLO-Fastest /
YOLObile). This repository builds on it by cleaning up the project structure, removing
hard-coded absolute paths, normalizing source encoding, and splitting the code into two
self-contained projects (images / live camera).

Model weights remain the property of their respective YOLO authors and are used here for
learning and research purposes only.
