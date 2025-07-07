# 👤 OpenCV Face Recognition (Python)

A straightforward Python project that trains on **pre-collected face images** and then recognises those faces live from webcam.

---

## 🎯 Project Goal  

Learn how to:
- Organise a face-image dataset for machine learning  
- Train an **LBPH** (Local Binary Patterns Histogram) model with OpenCV  
- Run real-time face recognition through a webcam feed  
- Interpret model confidence scores and tweak thresholds  
- Package and document a small AI project for university submission

---

## 🧱 What We Did

We wrote a single Python script, **`face_recognition_project.py`**, that can:

1. **Train** – load every image found in `datasets`, build an LBPH model, and save it to `trainer.yml`.  
2. **Recognise** – open the default webcam, detect faces with a Haar cascade, and predict each face’s identity using the trained model.

During training the script also saves a **label map** (`labels.npy`) so the model’s numeric IDs map back to human-readable names.

---

## 🛠️ Tools & Technologies  

- 🧠 OpenCV (LBPH Face Recognizer)  
- 🧮 NumPy  
- 💻 Visual Studio Code  
- 🔀 Git & GitHub  
- 🎥 Webcam  
- 🖥️ Tested on Windows & Ubuntu

---

## 🧪 How It Works

1. **Dataset loading**  
   The script scans every sub-folder in `datasets/`. Each folder name becomes a class label; every image inside is read in grayscale and added to the training set.  

2. **Model Training**  
   `cv2.face.LBPHFaceRecognizer_create()` learns texture patterns unique to each face. After training, the model is written to `trainer.yml`; the label map is dumped to `labels.npy`.

3. **Real-time Recognition**  
   The webcam stream is converted to gray. Faces are detected with the Haar cascade. Each detected face is cropped, resized automatically by OpenCV, and passed to `recognizer.predict()`.  
   - **Confidence < 60 →** accepted and the person’s name is drawn in green.  
   - **Confidence ≥ 60 →** treated as **Unknown** and drawn in red.  

4. **Threshold Tuning**  
   Edit the line `if confidence < 60:` to a higher number (e.g. 80) if recognition is too strict, or a lower number if it is too lenient.

---

## 🗄️ Minimum Dataset Example

> **Tip:** 3-5 clear photos per person (different angles & lighting) are usually enough for a demo.


Add more people by simply creating new folders at the same level.

---

## 🚀 How to Run Locally

```bash
# 1) Clone the repo
git clone https://github.com/dhm-tech/FaceRecognition.git
cd FaceRecognition

# 2) Install requirements
pip install -r requirements.txt

# 3) Train on your images
python face_recognition_project.py --mode train

# 4) Start live recognition
python face_recognition_project.py --mode recognize
```

---

## 👤 Author
> Designed by: [Abdulrahman Qutah]  
> Date: [7 Jul 2025]
