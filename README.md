
# Real-Time Facial Emotion Detection using YOLOv8 + System Performance Monitoring

This project implements a real-time facial emotion detection system using **YOLOv8** combined with **system performance monitoring tools** such as `psutil`, `torch`, and `OpenCV`. It runs on both **CPU and GPU** environments and compares their performance in terms of **FPS, CPU usage, memory usage, and GPU memory (if applicable)**.

## 🔍 Project Overview

This project was developed as part of an Operating Systems coursework to:
- Demonstrate real-time image processing using YOLOv8.
- Compare system performance on different hardware setups (Mac M2, Colab GPU, and Windows with NVIDIA GPU).
- Log and analyze key metrics like CPU%, memory%, GPU memory, and FPS.
- Design modular, maintainable, and well-documented code with robust error handling.

---

## 📁 Project Structure

```bash
📦 OS-RealTime-YOLOv8
├── real_time_webcam.py        # Real-time detection using webcam
├── colab_video_csv_logger.py  # Offline video processing on Colab and save stats to CSV
├── runs/                      # YOLO training logs and weights (output folder)
├── weights/                   # Pretrained weights (e.g., best.pt from Roboflow)
├── diagrams/
│   └── architecture.png       # Architecture overview image
├── README.md                  # You're here!


---

## 🧠 Architecture Diagram
```
+---------------+         +---------------------+         +------------------+
|   Webcam /    |  --->   |  YOLOv8 Inference   |  --->   |  Annotated Frame |
|   Video File  |         |     (CPU / GPU)     |         |   + Performance  |
+---------------+         +---------------------+         +------------------+
                                     |
                                     v
                           +----------------------+
                           |  System Stats (CSV)  |
                           +----------------------+
```
- The diagram shows the data flow from **camera/video input → YOLOv8 model → annotated output**.
- System stats are gathered using `psutil` and overlaid on the output or logged in CSV.

---

## 🖥️ Code 1: Real-Time Detection from Webcam (`real_time_webcam.py`)

### Description
- Uses OpenCV to open webcam.
- Runs YOLOv8 model on each frame.
- Shows FPS, CPU usage, RAM usage, and GPU memory (if available).
- Displays the annotated webcam output in a live window.

### Key Features
- Real-time emotion detection.
- Monitors system performance.
- Clean exit using `'q'` key.

### Sample Output

```
🔧 Running on: cuda
CPU: 45.2% RAM: 58.7% FPS: 24.1 GPU Mem: 212.3 MB
```
![image](https://github.com/user-attachments/assets/8b4835b5-d929-40e8-8349-872bd1fcbf11)

### Code Highlights

```python
model = YOLO("runs/detect/train/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_percent = psutil.cpu_percent()
fps = 1 / (time.time() - start_time)
```

---

## 🧪 Code 2: Offline Video Analysis on Google Colab (`colab_video_csv_logger.py`)

### Description
- Loads pre-recorded video (e.g., `.MOV`, `.mp4`).
- Processes each frame with YOLOv8.
- Measures FPS, CPU%, RAM%, and GPU memory.
- Saves stats into a CSV file for benchmarking.

### Use Case
- Helpful for environments where live webcam is not available (like Google Colab).
- Useful for running performance comparisons between CPU and GPU.

### Output
- Two CSV files:
  - `stats_cpu_YYYYMMDD_HHMMSS.csv`
  - `stats_gpu_YYYYMMDD_HHMMSS.csv`

### Code Snippet

```python
writer.writerow(["Frame", "FPS", "CPU_Usage", "RAM_Usage", "GPU_Memory_MB"])
gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
```

---

## 🚀 How to Run

### 🧑‍💻 Real-Time Webcam (Desktop)

```bash
pip install ultralytics opencv-python psutil torch
python real_time_webcam.py
```

Make sure `best.pt` is in the `weights/` folder.

### ☁️ On Google Colab

1. Upload your video and model weights to Google Drive.
2. Mount Google Drive in Colab.
3. Copy the `colab_video_csv_logger.py` code and run it.
4. CSV files will be downloaded after execution.

---

## ⚙️ System Requirements

| Component | Required |
|----------|----------|
| Python   | 3.8+     |
| Torch    | 2.0+     |
| OpenCV   | ✅       |
| GPU (Optional) | NVIDIA CUDA for acceleration |
| OS       | Windows, macOS (M1/M2), or Linux |

---

## 📊 Performance Comparison Summary

| Platform           | FPS   | CPU%   | RAM%   | GPU Mem |
|-------------------|-------|--------|--------|----------|
| Mac M2 (CPU only) | 5–10  | High   | Medium | ❌ Not available |
| Google Colab (CPU)| 10–12 | Medium | Medium | ❌ |
| Google Colab (GPU)| 20–30 | Low    | Low    | ✅ 100–300MB |
| Windows + NVIDIA  | 25–35 | Low    | Low    | ✅ CUDA |

---

## 💡 Trade-Offs

- **YOLOv8** was chosen over traditional ML models like Random Forest for superior real-time performance.
- **Colab** was used for GPU testing since macOS lacks CUDA.
- **Real-time webcam detection** only feasible on local systems with GPU access.

---

## 🧼 Clean Code Practices

- Functions are modularized and separated (e.g., video reading, YOLO inference, stats logging).
- Error handling for missing frames or webcam failure.
- Code is commented thoroughly for educational readability.

---

## 📌 Future Improvements

- Add head pose and facial landmark detection.
- Automatically optimize image brightness/contrast for low light.
- Expand dataset with more emotions and edge cases.

---
