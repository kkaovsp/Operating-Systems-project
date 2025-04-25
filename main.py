import cv2
import torch
import psutil
import time
from ultralytics import YOLO

# Load YOLOv8 model (choose one: yolov8n.pt or yolov8s.pt)
model = YOLO("OS project/runs/detect/train/weights/best.pt")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Running on: {device}")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Convert to RGB for YOLO
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale for YOLO
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rgb_frame = cv2.merge([gray_frame, gray_frame, gray_frame])

    # Run YOLO prediction
    results = model.predict(rgb_frame, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
    
    # Annotate detections
    annotated_frame = results[0].plot()
    bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # System stats
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    fps = 1 / (time.time() - start_time)
    
    # Apple GPU or CUDA (if any)
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        gpu_mem = 0  # macOS Metal GPU is not shown via PyTorch

    # Display stats on frame
    cv2.putText(bgr_frame, f"CPU: {cpu_percent:.1f}%  RAM: {ram_percent:.1f}%", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(bgr_frame, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(bgr_frame, f"GPU Mem: {gpu_mem:.1f} MB", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Show window
    cv2.imshow("YOLOv8 - CPU/GPU Monitor", bgr_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
