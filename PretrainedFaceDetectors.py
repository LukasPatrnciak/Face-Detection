# B I O M E T R I A
# Lukas Patrnciak
# AIS ID: 92320
# xpatrnciak@stuba.sk


# KNIZNICE
import cv2
from cv2 import data
import mediapipe as mp
import matplotlib.pyplot as plt
import os


# KONSTANTY
IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.5  # MediaPipe confidence
USE_PREPROCESSING = True


# FUNKCIE
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

def detect_faces_mediapipe(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    local_results = mp_face_detection.process(rgb_image)

    if local_results.detections:
        for detection in local_results.detections:
            bbox = detection.location_data.relative_bounding_box
            img_h, img_w, _ = image.shape
            x = int(bbox.xmin * img_w)
            y = int(bbox.ymin * img_h)
            w = int(bbox.width * img_w)
            h = int(bbox.height * img_h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def draw_yolo_annotation(image, labels_path):
    if not os.path.exists(labels_path):
        return image

    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, x_center, y_center, width, height = map(float, parts)

            h_img, w_img, _ = image.shape
            x_center *= w_img
            y_center *= h_img
            width *= w_img
            height *= h_img

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

def create_collage(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for ax, image in zip(axes.flatten(), images):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def preprocess_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return interArea / (areaA + areaB - interArea + 1e-6)

def get_haar_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

def get_mediapipe_boxes(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    local_results = mp_face_detection.process(rgb_image)
    h, w, _ = image.shape
    boxes = []

    if local_results.detections:
        for detection in local_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            boxes.append((x1, y1, x2, y2))

    return boxes

def get_yolo_annotation(file_path, img_shape):
    h_img, w_img = img_shape[:2]
    boxes = []

    if not os.path.exists(file_path):
        return boxes

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            x1 = int((x_center - width / 2) * w_img)
            y1 = int((y_center - height / 2) * h_img)
            x2 = int((x_center + width / 2) * w_img)
            y2 = int((y_center + height / 2) * h_img)
            boxes.append((x1, y1, x2, y2))

    return boxes

def evaluate_detector(detector_fn, image_path, use_preprocessing):
    TP, FP, FN = 0, 0, 0
    local_errors_per_image = {}

    for local_path in image_path:
        local_img = cv2.imread(local_path)

        if local_img is None:
            continue

        if use_preprocessing:
            local_img  = preprocess_image(local_img)

        filename = os.path.splitext(os.path.basename(local_path))[0] + ".txt"
        label_path = os.path.join(labels_folder, filename)
        gt_boxes = get_yolo_annotation(label_path, local_img.shape)
        pred_boxes = detector_fn(local_img)
        matched = set()
        tp_local, fp_local, fn_local = 0, 0, 0

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

        TP += tp_local
        FP += fp_local
        FN += fn_local

        local_errors_per_image[local_path] = fp_local + fn_local

    local_precision = TP / (TP + FP + 1e-6)
    local_recall = TP / (TP + FN + 1e-6)
    local_f1 = 2 * (local_precision * local_recall) / (local_precision + local_recall + 1e-6)

    return local_precision, local_recall, local_f1, local_errors_per_image



# NASTRENOVANE MODELY
face_cascade = cv2.CascadeClassifier(data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# SPRACOVANIE OBRAZKOV
base_folder = "Dataset"
images_folder = os.path.join(base_folder, "images")
labels_folder = os.path.join(base_folder, "labels")
image_paths = []
original_images = []
haar_results = []
mediapipe_results = []
annotation_results = []

for process_filename in os.listdir(images_folder):
    if process_filename.lower().endswith(('png', 'jpg', 'jpeg')):
        label_filename = os.path.splitext(process_filename)[0] + ".txt"
        lbl_path = os.path.join(labels_folder, label_filename)

        if os.path.exists(lbl_path):
            img_path = os.path.join(images_folder, process_filename)
            image_paths.append(img_path)

for process_path in image_paths:
    process_img = cv2.imread(process_path)

    if process_img is not None:
        original_images.append(process_img)

selected_images =  original_images[:6]

for j, process_img in enumerate(selected_images):
    process_filename = os.path.splitext(os.path.basename(image_paths[j]))[0] + ".txt"
    process_label_path = os.path.join(labels_folder, process_filename)

    # haar_results.append(detect_faces_haar(process_img.copy()))
    # mediapipe_results.append(detect_faces_mediapipe(process_img.copy()))
    annotation_results.append(draw_yolo_annotation(process_img.copy(), process_label_path))

create_collage(annotation_results, 2, 3)


# VYHODNOTENIE
print("\n=== VYHODNOTENIE DETEKTOROV ===")

results = []
all_errors = {}

for detector, name in [(get_haar_boxes, "Haar Cascade"), (get_mediapipe_boxes, "MediaPipe")]:
    for preprocess in [False, True]:
        label = f"{name} | {'predspracované' if preprocess else 'nepredspracované'}"

        precision, recall, f1, errors_per_image = evaluate_detector(detector, image_paths, preprocess)

        results.append({
            'label': label,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(f"\n{label}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")

        for path, err_count in errors_per_image.items():
            all_errors[path] = all_errors.get(path, 0) + err_count

best_result = max(results, key=lambda a: a['f1'])

print("\nNAJLEPŠÍ MODEL A NASTAVENIE")
print(f"{best_result['label']} (F1: {best_result['f1']:.3f})")

print("\nVPLYV PREDSPRACOVANIA")

for name in ["Haar Cascade", "MediaPipe"]:
    pre, raw = None, None

    for r in results:
        if r['label'] == f"{name} | predspracované":
            pre = r

        elif r['label'] == f"{name} | nepredspracované":
            raw = r

    if pre and raw:
        diff = pre['f1'] - raw['f1']
        zmena = "zlepšilo" if diff > 0 else "zhoršilo"

        print(f"{name}: Predspracovanie {zmena} výkon (ΔF1 = {diff:.3f})")

worst_imgs = sorted(all_errors.items(), key=lambda x: x[1], reverse=True)[:6]
worst_img_objs = []

for img_path, err in worst_imgs:
    img = cv2.imread(img_path)

    if img is not None:
        txt_path = os.path.join(labels_folder, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        img = draw_yolo_annotation(img, txt_path)
        img = detect_faces_haar(img)
        img = detect_faces_mediapipe(img)
        worst_img_objs.append(img)

print("\nNAJŤAŽŠIE OBRÁZKY NA DETEKCIU")

for p, err in worst_imgs:
    print(f"{os.path.basename(p)} – Chýb/nesprávnych detekcií: {err}")

create_collage(worst_img_objs, 2, 3)