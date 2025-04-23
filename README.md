# Real-Time Facial Emotion Detection using YOLOv8 + System Performance Monitoring

This project implements a real-time facial emotion detection system using **YOLOv8** combined with **system performance monitoring tools** such as `psutil`, `torch`, and `OpenCV`. It runs on both **CPU and GPU** environments and compares their performance in terms of **FPS, CPU usage, memory usage, and GPU memory (if applicable)**.

## 🔍 Project Overview

This project was developed as part of an Operating Systems coursework to:
- Demonstrate real-time image processing using YOLOv8.
- Compare system performance on different hardware setups (Mac M2, Colab GPU, and Windows with NVIDIA GPU).
- Log and analyze key metrics like CPU%, memory%, GPU memory, and FPS.
- Design modular, maintainable, and well-documented code with robust error handling.

---

## 🧠 Architecture Diagram
```bash
+---------------+         +---------------------+         +------------------+
|   Webcam /    |  --->   |  YOLOv8 Inference   |  --->   |  Annotated Frame |
|   Video File  |         |     (CPU / GPU)     |  <---   |   + Performance  |
+---------------+         +---------------------+         +------------------+
                                     |
                                     v
                   +-----------------------------------------+
                   |  System Stats (CSV) and display output  |
                   +-----------------------------------------+
```
- The diagram shows the data flow from **camera/video input → YOLOv8 model → annotated output**.
- System stats are gathered using `psutil` and overlaid on the output or logged in CSV.

---

## 🖥️ Code 1: Real-Time Detection from Webcam (`main.py`)

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
#### Convert to RGB for YOLO
```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

#### Run YOLO prediction
```python
results = model.predict(rgb_frame, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
```

#### Annotate detections
```python
annotated_frame = results[0].plot()
bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
```

#### System stats
```python
cpu_percent = psutil.cpu_percent()
ram_percent = psutil.virtual_memory().percent
fps = 1 / (time.time() - start_time)
```

#### Apple GPU or GPU CUDA (if any)
```python
if torch.cuda.is_available():
  gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
else:
  gpu_mem = 0  # macOS Metal GPU is not shown via PyTorch
```

---

## 🧪 Code 2: Offline Video Analysis on Google Colab (`Create_CSV_from_CPU_and_GPU.ipynb`)

### Description
- Loads pre-recorded video (e.g., `.MOV`, `.mp4`).
- Processes each frame with YOLOv8.
- Measures FPS, CPU%, RAM%, and GPU memory.
- Saves stats into a CSV file for benchmarking.

### Use Case
- Helpful for environments where live webcam is not available (like Google Colab).
- Useful for running performance comparisons between CPU and GPU.

### Code Upload file
```python
uploaded = files.upload()
```

### Code to call CPU working
```python
cpu_csv = run_yolo_on_video("cpu", max_frames=100)
files.download(cpu_csv)
```

### Code to call GPU working
##### Click Runtime > Change runtime type > GPU
```python
gpu_csv = run_yolo_on_video("GPU", max_frames=100)
files.download(gpu_csv)
```

### Output
- Two CSV files:
  - `stats_cpu_YYYYMMDD_HHMMSS.csv`
  - `stats_gpu_YYYYMMDD_HHMMSS.csv`

### FBS Comparison between CPU and GPU, running on Google Colab
**The graph shows that the GPU performs better.**
![FPS on CPU and GPU Usage Over Time](https://github.com/user-attachments/assets/455e0893-9cb6-4245-8f3b-826179fa3de4)

---

## 🚀 How to Run

### 🧑‍💻 Real-Time Webcam main.py on Desktop or vscode

```bash
pip install opencv-python
pip install torch
pip install psutil
pip install ultralytics

```
Make sure have `best.pt`

### ☁️ On Google Colab

1. Upload your video and model weights to Google Drive.
2. Mount Google Drive in Colab.
3. Paste the `Create_CSV_from_CPU_and_GPU.ipynb` code and run it.
4. CSV files will be downloaded after execution.

---

## 📊 Performance Comparison Summary and Use in projects

| Platform           | FPS   | CPU%   | RAM%   | GPU Mem |
|-------------------|-------|--------|--------|----------|
| Mac M2 (CPU only) | 5–10  | High   | Medium | ❌ Not available |
| Google Colab (CPU)| 10–12 | Medium | Medium | ❌ |
| Google Colab (GPU)| 20–30 | Low    | Low    | ✅ 100–300MB |
| Windows + NVIDIA  | 25–35 | Low    | Low    | ✅ CUDA |


