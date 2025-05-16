# B I O M E T R I A
# Lukas Patrnciak
# AIS ID: 92320
# xpatrnciak@stuba.sk


# KNIZNICE
import os
import random
import shutil
import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ultralytics import YOLO


# KONSTANTY
IMG_DIR = "Dataset/images"
LBL_DIR = "Dataset/labels"
YOLO_DIR = "YOLO"
IOU_THRESHOLD = 0.7


# FUNKCIE
def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    images = [fe for fe in os.listdir(image_dir) if fe.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    train_split = int(train_ratio * len(images))
    val_split = int((train_ratio + val_ratio) * len(images))
    sets = {
        'train': images[:train_split],
        'val': images[train_split:val_split],
        'test': images[val_split:]}

    for set_name, file_list in sets.items():
        img_out = os.path.join(output_dir, 'images', set_name)
        lbl_out = os.path.join(output_dir, 'labels', set_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for file in file_list:
            img_src = os.path.join(image_dir, file)
            lbl_src = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
            shutil.copy(img_src, img_out)

            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, lbl_out)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return interArea / (areaA + areaB - interArea + 1e-6)

def get_yolo_annotation(file_path, img_shape):
    h_img, w_img = img_shape[:2]
    boxes = []

    if not os.path.exists(file_path):
        return boxes

    with open(file_path, 'r') as fle:
        for line in fle:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            local_x1 = int((x_center - width / 2) * w_img)
            local_y1 = int((y_center - height / 2) * h_img)
            local_x2 = int((x_center + width / 2) * w_img)
            local_y2 = int((y_center + height / 2) * h_img)
            boxes.append((local_x1, local_y1, local_x2, local_y2))

    return boxes

def evaluate_detector(model):
    local_TP, local_FP, local_FN, local_TN = 0, 0, 0, 0
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)

        if img is None:
            continue

        gt_boxes = get_yolo_annotation(label_path, img.shape)
        result = model.predict(source=img, conf=0.25, iou=0.7, verbose=False)[0]

        pred_boxes = []

        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            pred_boxes.append((x1, y1, x2, y2))

        matched = set()
        tp_local, fp_local = 0, 0

        for pred in pred_boxes:
            matched_flag = False

            for i, gt in enumerate(gt_boxes):
                if i in matched:
                    continue

                if iou(pred, gt) > IOU_THRESHOLD:
                    matched.add(i)
                    matched_flag = True
                    break

            if matched_flag:
                tp_local += 1
            else:
                fp_local += 1

        fn_local = len(gt_boxes) - len(matched)

        local_TP += tp_local
        local_FP += fp_local
        local_FN += fn_local

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            local_TN += 1

    local_precision = local_TP / (local_TP + local_FP + 1e-6)
    local_recall = local_TP / (local_TP + local_FN + 1e-6)
    local_f1 = 2 * (local_precision * local_recall) / (local_precision + local_recall + 1e-6)

    return local_precision, local_recall, local_f1, local_TP, local_FP, local_FN, local_TN

def main():
    # SUBOR YOLO_DATA.YAML
    yaml_content = """path: YOLO
train: images/train
val: images/val
test: images/test
names:
    0: face
"""

    with open("yolo_data.yaml", "w") as f:
        f.write(yaml_content)

    # ROZDELENIE A TRENING YOLO
    split_dataset(IMG_DIR, LBL_DIR, YOLO_DIR)
    model = YOLO("yolo11n.pt")
    model.train(data="yolo_data.yaml", epochs=10, imgsz=640, batch=32)


    # VYHODNOTENIE YOLO DETEKTORA (YOLO METRIKY)
    # results = model.val(data="yolo_data.yaml", split="test")
    # metrics = results.box

    # print("\n=== YOLO METRIKY Z VAL() ===")
    # print(f"Precision:      {metrics.p:.3f}")
    # print(f"Recall:         {metrics.r:.3f}")
    # print(f"F1 Score:       {metrics.f1:.3f}")


    # VYHODNOTENIE YOLO DETEKTORA (VLASTNE METRIKY)
    # Haar, MediaPipe
    precisions = [0.29, 0.35]
    recalls = [0.21, 0.15]
    f1s = [0.25, 0.21]
    precision, recall, f1_score, TP, FP, FN, TN = evaluate_detector(model)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1_score)

    print("\n=== VYHODNOTENIE YOLO DETEKTORA (IoU ≥ 0.7) ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1_score:.3f}")

    conf_matrix = np.array([[TP, FN], [FP, TN]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Positive", "Negative"],
                yticklabels=["Positive", "Negative"])
    plt.title("Konfúzna matica")
    plt.xlabel("Predikcia")
    plt.ylabel("Skutočnosť")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    detectors = ['Haar', 'MediaPipe', 'YOLO']

    x = range(len(detectors))

    plt.bar(x, precisions, width=0.25, label='Presnosť')
    plt.bar([p + 0.25 for p in x], recalls, width=0.25, label='Recall')
    plt.bar([p + 0.50 for p in x], f1s, width=0.25, label='F1-skóre')

    plt.xticks([p + 0.25 for p in x], detectors)
    plt.ylabel('Skóre')
    plt.title('Porovnanie detektorov')
    plt.legend()
    plt.show()


# SPUSTENIE
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Pre kompatibilitu s .exe
    main()