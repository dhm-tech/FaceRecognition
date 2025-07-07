import cv2
import os
import numpy as np
import argparse
from pathlib import Path

# إعدادات المجلدات
DATASET_DIR = Path("dataset")
MODEL_FILE = Path("trainer.yml")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# لجمع صور الوجه
def capture_faces(name: str, num_samples: int = 50, cam_index: int = 0):
    person_dir = DATASET_DIR / name
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(cam_index)
    count = 0
    print(f"[INFO] Capturing faces for '{name}'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't read from camera.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            count += 1
            cv2.imwrite(str(person_dir / f"{count}.png"), face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capturing...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {count} images saved in {person_dir}")

# تجهيز البيانات للتدريب
def load_images_and_labels():
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for person_dir in DATASET_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = current_id
            current_id += 1
        label_id = label_map[name]
        for img_path in person_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_id)
    return faces, np.array(labels), label_map

# تدريب النموذج
def train_lbph():
    faces, labels, label_map = load_images_and_labels()
    if not faces:
        raise RuntimeError("No face data found.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.write(str(MODEL_FILE))
    np.save(MODEL_FILE.with_suffix('.labels.npy'), label_map)
    print(f"[INFO] Model trained and saved as {MODEL_FILE}")

# تشغيل التعرف
def recognize_faces(cam_index: int = 0, conf_threshold: float = 60):
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Train first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    label_map = np.load(MODEL_FILE.with_suffix('.labels.npy'), allow_pickle=True).item()
    inv_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(cam_index)
    print("[INFO] Starting face recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_img)
            name = inv_map.get(label_id, "Unknown")
            text = f"{name} ({confidence:.1f})" if confidence < conf_threshold else "Unknown"
            color = (0, 255, 0) if confidence < conf_threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------- Main ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["capture", "train", "recognize"])
    parser.add_argument("--name", help="Person's name for capture")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(DATASET_DIR)

    if args.mode == "capture":
        if not args.name:
            parser.error("--name is required when using capture mode.")
        capture_faces(args.name, args.samples, args.cam)
    elif args.mode == "train":
        train_lbph()
    elif args.mode == "recognize":
        recognize_faces(args.cam)
