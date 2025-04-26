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
+------------------------+       +-------------------------+
|  Face Dataset (Images) | ----> | YOLOv8 Training Script  |
+------------------------+       +-------------------------+
                                         |
                                         v
                              +------------------------+
                              | Trained YOLOv8 Model   |
                              +------------------------+
                                         |
                                         v
  +---------------------+       +---------------------------------+
  | Video Input Source  | ----> | Real-Time Detection Pipeline    |
  | (Webcam/Stream)     |       | (OpenCV + YOLOv8 + Ultralytics) |
  +---------------------+       +---------------------------------+
                                         |
                      +------------------+--------------------+
                      |                                       |
                      v                                       v
     +-----------------------------+          +------------------------------+
     | Draw Bounding Boxes on Feed|           | Monitor System Stats in Real |
     | (Face Detection + FPS)     |           | Time (CPU, GPU, MEM, FPS)    |
     +-----------------------------+          +------------------------------+
                      |                                       |
                      +------------------+--------------------+
                                         |
                                         v
                   +---------------------------------------------+
                   | Display UI / Dashboard / System Stats (CSV) |
                   +---------------------------------------------+

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

---
## 🔥🔥 Start with `main.py` , This code is real time detection
---
## 1. 🛠️ Requirements

Make sure you install the following Python packages:

```bash
pip install ultralytics
pip install opencv-python
pip install psutil
torch (PyTorch) must be installed too.
```

👉 **Note:**  
- If you want GPU acceleration, make sure your PyTorch is installed with CUDA support.
- If using Google Colab, macOS, or Windows, check your hardware.

---

## 2. 📂 Project Structure

- `best.pt` — Your trained YOLOv8 model.
- `OS project/runs/detect/train/weights/best.pt` — Path to the model file.
- Your webcam or external camera device connected to the computer.

---

## 3. 📜 How the Code Works (Step-by-Step)

### Step 1: Import Required Libraries
```python
import cv2, torch, psutil, time
from ultralytics import YOLO
```
These libraries are used for:
- `cv2`: Capture and display video frames.
- `torch`: Run YOLO model on CPU or GPU.
- `psutil`: Monitor CPU and RAM usage.
- `time`: Measure FPS (Frames Per Second).
- `ultralytics`: YOLOv8 model operations.

---

### Step 2: Load the YOLOv8 Model
```python
model = YOLO("best.pt")
```
You load your **custom-trained** or **pretrained** YOLOv8 model.

---

### Step 3: Set Device (CPU or GPU)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Running on: {device}")
```
- If GPU is available (`cuda`), it will use GPU.
- Otherwise, fallback to CPU.

---

### Step 4: Open the Webcam
```python
cap = cv2.VideoCapture(1)
```
- `0` for default webcam.
- `1`, `2`, etc., for external cameras.

---

### Step 5: Start the Real-Time Loop
Inside the loop:

1. **Capture a Frame**  
```python
ret, frame = cap.read()
```
If the frame is not grabbed successfully, the program will print a warning.

---

2. **Convert the Frame**
```python
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rgb_frame = cv2.merge([gray_frame, gray_frame, gray_frame])
```
✅ **Optimization:**
- Convert BGR to **Grayscale** (1 channel → less memory and faster).
- Duplicate into 3 channels (needed because YOLOv8 expects RGB 3-channel input).

---
   
3. **Run YOLO Detection**
```python
results = model.predict(rgb_frame, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
```
- YOLOv8 detects objects on each frame.
- The results include bounding boxes and class predictions.

---

4. **Annotate the Frame**
```python
annotated_frame = results[0].plot()
bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
```
- Draws bounding boxes on detected objects.
- Converts back to BGR for OpenCV display.

---

5. **Monitor and Display System Statistics**
```python
cpu_percent = psutil.cpu_percent()
ram_percent = psutil.virtual_memory().percent
fps = 1 / (time.time() - start_time)
gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
```
Displays:
- CPU usage (%)
- RAM usage (%)
- Frames per second (FPS)
- GPU memory usage (MB)

Overlay the statistics onto the frame using `cv2.putText`.

---

6. **Show the Video Feed**
```python
cv2.imshow("YOLOv8 - CPU/GPU Monitor", bgr_frame)
```
- Shows the live video with detection and system stats.
- Press `q` to quit the program.

---

7. **Cleanup After Exit**
```python
cap.release()
cv2.destroyAllWindows()
```
- Releases the camera.
- Closes all OpenCV windows.

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


