# 📘 Tutorial: Real-Time YOLOv8 Detection with CPU/GPU Monitoring (with OpenCV Grayscale Optimization)

This tutorial explains **step-by-step** how to use the provided Python script for **real-time video detection** with **YOLOv8**, while **monitoring system performance** (CPU, RAM, and GPU usage).  
We also apply **grayscale optimization** to save memory and speed up processing.

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

## 4. 🎯 Key Features of this Project

- **Real-time YOLOv8 detection** on live webcam feed.
- **Dynamic hardware detection**: Runs on CPU or GPU automatically.
- **System monitoring**: CPU%, RAM%, GPU memory, and real-time FPS.
- **Grayscale optimization**: Saves memory, reduces CPU load, and increases FPS.

---

## 5. 🔥 Important Tips

- If you experience lag or high CPU usage, lower webcam resolution or increase grayscale optimization.
- Ensure your webcam permission is granted (especially on macOS).
- If running on GPU, always install correct PyTorch with CUDA version.
- If using Google Colab, change the "Runtime" → "Change runtime type" → Hardware accelerator to GPU.

---

## 6. 🚀 Possible Improvements

- Add automatic device detection for TPU.
- Log system stats to CSV file for deeper performance analysis.
- Try running YOLOv8n (nano version) for even faster results with a lighter model.
- Add real-time plotting of CPU, RAM, GPU trends using matplotlib.

---

# ✅ Conclusion

This script is a complete mini real-time AI application combining:
- **YOLOv8 detection**
- **Resource optimization with Grayscale frames**
- **Real-time CPU/GPU monitoring**

It can serve as a strong foundation for future real-world projects like fall detection, emotion recognition, surveillance, etc.

---

## 🔥🔥 Second code is `Create_CSV_from_CPU_and_GPU.ipynb` , This code for create the csv file of resource usage.

---
# 📘 Tutorial: Batch YOLOv8 Video Inference with CPU/GPU Resource Monitoring

This guide explains how to use the provided script to **analyze a video file frame-by-frame** using **YOLOv8**, while **logging system performance metrics** (CPU, RAM, GPU memory) to a CSV file for later analysis.

---

## 1. 🛠️ Requirements

Install the necessary Python libraries:

```bash
pip install ultralytics
pip install opencv-python
pip install psutil
torch (PyTorch) must be installed.
```

> **Tip:** If you want GPU acceleration, install PyTorch with CUDA support.

---

## 2. 📂 Project Setup

You should have:
- A **trained YOLOv8 model** (`best.pt` or similar).
- A **video file** (e.g., `.mp4`, `.mov`) you want to analyze.

Paths used in the code:
```python
model_path = "/content/drive/MyDrive/Colab Notebooks/OS project/runs/detect/train/weights/best.pt"
video_path = "/content/drive/MyDrive/Colab Notebooks/OS project/J fall.MOV"
```

> **Make sure** to adjust the paths to your own files if different.

---

## 3. 📜 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
import cv2, torch, psutil, time, csv
from datetime import datetime
from ultralytics import YOLO
```
- `cv2` for video reading.
- `torch` for model loading and device handling.
- `psutil` to monitor CPU and RAM.
- `csv` and `datetime` to log results into a file.

---

### Step 2: Load the YOLOv8 Model
```python
model = YOLO(model_path)
```
Loads your trained YOLOv8 model into memory.

---

### Step 3: Define the `run_yolo_on_video` Function

This function processes the video, frame by frame.

Parameters:
- `device_str`: `"cpu"` or `"GPU"`.
- `max_frames`: maximum number of frames to analyze.

---

Inside the function:

#### a) Set Device
```python
device = device_str if device_str == "cpu" else 0
```
- `"cpu"` keeps it on CPU.
- Anything else (like `"GPU"`) moves processing to CUDA GPU device `0`.

---

#### b) Open Video
```python
cap = cv2.VideoCapture(video_path)
```
Reads video file frame-by-frame.

---

#### c) Create a CSV File for Logging
```python
csv_filename = f"stats_{device_str}_{timestamp}.csv"
```
The script automatically saves a new CSV with the device name and timestamp in the filename.

---

#### d) Process Video Frames
While frames are available **and** `frame_count < max_frames`:

1. **Read frame**
   ```python
   ret, frame = cap.read()
   ```

2. **(Optional) Grayscale Optimization**  
   (commented out but available for future use):
   ```python
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   rgb_frame = cv2.merge([gray, gray, gray])
   ```
   ✅ This would **reduce memory and speed up processing** if uncommented.

3. **Convert to RGB**  
   (YOLO expects RGB input):
   ```python
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   ```

4. **Run YOLOv8 prediction**
   ```python
   _ = model.predict(rgb_frame, device=device, verbose=False)
   ```

5. **Measure system performance**
   ```python
   cpu = psutil.cpu_percent()
   ram = psutil.virtual_memory().percent
   fps = 1 / (time.time() - start)
   gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
   ```

6. **Write data to CSV**
   ```python
   writer.writerow([frame_count + 1, round(fps, 2), cpu, ram, round(gpu_mem, 2)])
   ```

---

#### e) Close Everything
```python
cap.release()
```
Releases the video file after all frames are processed.

---

### Step 4: Run Inference on CPU and GPU
** Don't forget to change the Runtime type**
**Click Runtime > Change runtime type > CPU**
```python
cpu_csv = run_yolo_on_video("cpu", max_frames=100)
files.download(cpu_csv)
```
**Click Runtime > Change runtime type > GPU**
```python
gpu_csv = run_yolo_on_video("GPU", max_frames=100)
files.download(gpu_csv)
```
✅ **Results:**
- One CSV file for CPU processing.
- One CSV file for GPU processing.
- Both are automatically downloaded after the run finishes.

---

## 4. 🎯 Key Features

- **Batch video analysis**: No need to watch real-time.
- **Frame-by-frame system monitoring**: CPU usage, RAM usage, FPS, GPU memory.
- **Automatic CSV logging**: Easy for later data analysis.
- **Device flexibility**: Choose CPU or GPU at runtime.
- **Grayscale optimization option**: Use less memory and compute if needed.

---

## 5. 🔥 Important Notes

- For best performance, use GPU if available.
- If you experience out-of-memory errors, either:
  - Lower frame size (resize input).
  - Enable grayscale mode to lighten frame load.
- If processing speed is critical, limit `max_frames` to smaller values during testing.
- Make sure `files.download()` is used in environments like **Google Colab** (won't work in normal local Python scripts).

---

## 6. 🚀 Advanced Ideas for Improvement

- Compare CPU vs GPU performance by plotting graphs.
- Automatically switch to grayscale for slow devices.
- Extend code to record detection results (bounding boxes, classes) along with system stats.

---

# ✅ Conclusion

This script offers a **lightweight**, **modular**, and **scalable** solution for analyzing how YOLOv8 runs across different hardware devices, making it perfect for optimizing real-world AI deployments where resources matter.

---

Would you also like me to **generate a version where Grayscale is automatically ON** for both CPU and GPU runs for maximum performance? 🚀  
(Like a **Phase 3 optimization version**!)  
It would make your report even stronger! 🔥
  
Let me know!