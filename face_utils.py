import cv2
import os
import numpy as np

# =========================
# BASE PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DIR = os.path.join(BASE_DIR, "data", "faces")
HAAR_DIR = os.path.join(BASE_DIR, "data", "haarcascade")

os.makedirs(FACE_DIR, exist_ok=True)


# =========================
# HAAR CASCADE
# =========================

CASCADE_PATH = os.path.join(
    HAAR_DIR,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError(
        f"Could not load Haar Cascade file: {CASCADE_PATH}"
    )


# =========================
# FACE MODEL PATHS
# =========================

MODEL_PATH = os.path.join(
    FACE_DIR,
    "face_model.yml"
)

LABELS_PATH = os.path.join(
    FACE_DIR,
    "labels.txt"
)


# =========================
# FACE DETECTION
# =========================

def detect_face(image_bytes):
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    face = gray[
        y:y + h,
        x:x + w
    ]

    return cv2.resize(
        face,
        (200, 200)
    )


# =========================
# ENROLL USER
# =========================

def enroll_user(username, image_bytes):

    face = detect_face(image_bytes)

    if face is None:
        return (
            False,
            "No face detected. Look directly at the camera."
        )

    user_dir = os.path.join(
        FACE_DIR,
        username
    )

    os.makedirs(
        user_dir,
        exist_ok=True
    )

    existing = len([
        x
        for x in os.listdir(user_dir)
        if x.lower().endswith(".jpg")
    ])

    filename = os.path.join(
        user_dir,
        f"{existing}.jpg"
    )

    cv2.imwrite(
        filename,
        face
    )

    train_model()

    return (
        True,
        "Face enrolled successfully."
    )


# =========================
# TRAIN MODEL
# =========================

def train_model():

    faces = []
    labels = []
    label_names = []

    current_label = 0

    for username in sorted(
        os.listdir(FACE_DIR)
    ):

        user_dir = os.path.join(
            FACE_DIR,
            username
        )

        if not os.path.isdir(user_dir):
            continue

        label_names.append(username)

        for filename in os.listdir(user_dir):

            path = os.path.join(
                user_dir,
                filename
            )

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                faces.append(image)
                labels.append(current_label)

        current_label += 1

    if not faces:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        faces,
        np.array(labels)
    )

    recognizer.write(
        MODEL_PATH
    )

    with open(
        LABELS_PATH,
        "w"
    ) as f:

        for name in label_names:
            f.write(name + "\n")

    return True


# =========================
# VERIFY USER
# =========================

def verify_user(username, image_bytes):

    if not os.path.exists(MODEL_PATH):
        return False, 999

    face = detect_face(image_bytes)

    if face is None:
        return False, 999

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(
        MODEL_PATH
    )

    label, confidence = recognizer.predict(
        face
    )

    with open(
        LABELS_PATH,
        "r"
    ) as f:

        names = [
            x.strip()
            for x in f.readlines()
        ]

    if label >= len(names):
        return False, confidence

    predicted_user = names[label]

    # LBPH:
    # lower confidence = better match
    matched = (
        predicted_user == username
        and confidence < 70
    )

    return matched, confidence