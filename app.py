import torch
import gradio as gr
import os
import time
import shutil
import glob
import csv
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Inference + save detection result
def detect_mines(image):
    # Cleanup previous detection folder to avoid exp2, exp3...
    if os.path.exists("runs/detect"):
        shutil.rmtree("runs/detect")

    # Run YOLOv5
    results = model(image)
    results.save()

    # Wait for image to save
    time.sleep(0.3)

    # Find the latest exp folder
    exp_folders = glob.glob('runs/detect/exp*')
    if not exp_folders:
        return None, "❌ Error: No detection output found."

    latest_exp = sorted(exp_folders, key=os.path.getmtime)[-1]
    yolo_output = os.path.join(latest_exp, 'image0.jpg')

    if not os.path.exists(yolo_output):
        return None, "❌ Error: Output image missing."

    # Make sure static/ exists
    os.makedirs("static", exist_ok=True)

    # Generate unique result filename
    timestamp = int(time.time() * 1000)
    result_path = f'static/result_{timestamp}.jpg'
    shutil.copy(yolo_output, result_path)

    # Build detection summary
    detections = results.pandas().xyxy[0]
    if len(detections) == 0:
        summary = "✅ No SATM or SAPEM landmines detected in the image."
    else:
        summary = "🚨 Landmine(s) detected:\n"
        for idx, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            summary += f"- {label.upper()} ({conf*100:.1f}%)\n"

    # Prepare log file
    log_file = 'detection_log.csv'
    log_exists = os.path.exists(log_file)

    # Log each detection to CSV
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not log_exists:
            writer.writerow(["timestamp", "class", "confidence"])

        # Timestamp for log
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if len(detections) == 0:
            writer.writerow([timestamp_str, "None", "-"])
        else:
            for idx, row in detections.iterrows():
                label = row['name']
                conf = f"{row['confidence']:.4f}"
                writer.writerow([timestamp_str, label.upper(), conf])

    return result_path, summary

# 🧹 Cleanup function
def clear_output():
    folders = ['static', 'runs/detect']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    return None, None, None

def serve_log():
    log_path = "detection_log.csv"
    if os.path.exists(log_path):
        return log_path
    else:
        return None

# Clean folders before starting app
for folder in ['static', 'runs/detect']:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Remove detection log file
if os.path.exists("detection_log.csv"):
    os.remove("detection_log.csv")
    print("[Startup Cleanup] Removed detection_log.csv")

# Gradio UI using Blocks (to support multiple functions)
with gr.Blocks(title="Chinese Landmine Detector") as demo:
    gr.Markdown("## Chinese Landmine Detector (SATM / SAPEM)")
    gr.Markdown("Upload an image. The model will detect SATM or SAPEM landmines.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            detect_btn = gr.Button("🔍 Detect")
            clear_btn = gr.Button("🧹 Clear")

        with gr.Column():
            image_output = gr.Image(type="filepath", label="Detection Result")
            summary_output = gr.Textbox(label="Detection Summary")
            log_download = gr.File(label="Download Detection Log")
            download_btn = gr.Button("📝️ Generate Log")

    # Events
    detect_btn.click(fn=detect_mines, inputs=image_input, outputs=[image_output, summary_output])
    clear_btn.click(
        fn=clear_output,
        inputs=[],
        outputs=[image_input, image_output, summary_output]
    )
    download_btn.click(fn=serve_log, inputs=[], outputs=log_download)

# Launch app
demo.launch()