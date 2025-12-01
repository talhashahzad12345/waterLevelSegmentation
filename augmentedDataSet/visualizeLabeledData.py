from ultralytics import YOLO
import cv2
import yaml
import os

# load any YOLO model (just for plotting)
model = YOLO("yolov8n.pt")

# load your dataset YAML
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

# train folder paths from YAML
img_dir = data['train']          # example: '../train/images'
label_dir = img_dir.replace("images", "labels")

# output folder
out_dir = "viz_images"
os.makedirs(out_dir, exist_ok=True)

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + ".txt")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Skipped unreadable:", img_path)
        continue

    h, w, _ = img.shape

    # read YOLO format labels
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, w_rel, h_rel = map(float, line.split())
                
                x1 = int((x - w_rel/2) * w)
                y1 = int((y - h_rel/2) * h)
                x2 = int((x + w_rel/2) * w)
                y2 = int((y + h_rel/2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(out_dir, img_name), img)

print("Done! Check folder:", out_dir)
