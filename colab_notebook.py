# ============================================================
# 🤟 نظام التعرف على لغة الإشارة العربية — KArSL
# Google Colab — نسخة محدثة (MediaPipe Tasks API)
# ============================================================


# ======================== CELL 1 ========================
# تثبيت + تحميل موديلات MediaPipe
# شغّل الخلية دي مرة واحدة بس ثم اعمل Restart session
# ========================================================

# !pip install mediapipe opencv-python-headless openpyxl --quiet
# !wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# !wget -q -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task


# ======================== CELL 2 ========================
# الإعدادات + ربط Drive
# ========================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
import glob
import json
import time
from sklearn.model_selection import train_test_split

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import tensorflow as tf
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"TensorFlow: {tf.__version__}")
print(f"MediaPipe: {mp.__version__}")

# مسارات
DRIVE_DATA_PATH = '/content/drive/MyDrive'  # ← عدّل حسب مكان الداتا عندك
WORK_DIR = '/content/karsl_project'
EXTRACTED_PATH = os.path.join(WORK_DIR, 'extracted_landmarks')
PROCESSED_PATH = os.path.join(WORK_DIR, 'processed')
MODEL_PATH = os.path.join(WORK_DIR, 'model')

os.makedirs(EXTRACTED_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

SEQUENCE_LENGTH = 30
FEATURES = 258  # 63 + 63 + 132

# البحث عن الفيديوهات
def find_sign_folders(base_path):
    sign_folders = []
    for root, dirs, files in os.walk(base_path):
        mp4s = [f for f in files if f.lower().endswith(('.mp4', '.avi'))]
        if mp4s:
            sign_id = os.path.basename(root)
            sign_folders.append((sign_id, root, len(mp4s)))
    sign_folders.sort(key=lambda x: x[0])
    return sign_folders

print("\n🔍 جاري البحث عن فيديوهات KArSL...")
sign_folders = find_sign_folders(DRIVE_DATA_PATH)
print(f"✅ تم العثور على {len(sign_folders)} إشارة")
for sf in sign_folders[:5]:
    print(f"   {sf[0]}: {sf[2]} فيديو")
if len(sign_folders) > 5:
    print(f"   ... و{len(sign_folders) - 5} إشارة أخرى")


# ======================== CELL 3 ========================
# الخطوة ١: استخراج Landmarks — MediaPipe Tasks API
# ========================================================

# إعداد MediaPipe Landmarkers
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='/content/hand_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='/content/pose_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.4,
    min_tracking_confidence=0.4
)


def extract_from_video(video_path):
    """
    يستخرج landmarks من فيديو واحد
    الناتج: مصفوفة (T, 258)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    keypoints_seq = []

    # إنشاء landmarkers جديدة لكل فيديو (عشان الـ timestamp يبدأ من صفر)
    hand_lm = vision.HandLandmarker.create_from_options(hand_options)
    pose_lm = vision.PoseLandmarker.create_from_options(pose_options)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * (1000 / fps))

        # استخراج اليد
        try:
            hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
        except:
            hand_result = None

        # استخراج الجسم
        try:
            pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
        except:
            pose_result = None

        # ===== تحويل لمصفوفة =====
        # يد يمين (21 × 3 = 63)
        rh = np.zeros(63)
        # يد شمال (21 × 3 = 63)
        lh = np.zeros(63)

        if hand_result and hand_result.hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(
                zip(hand_result.hand_landmarks, hand_result.handedness)):
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
                label = handedness[0].category_name.lower()
                if label == 'right':
                    rh = hand_data
                elif label == 'left':
                    lh = hand_data

        # جسم (33 × 4 = 132)
        pose = np.zeros(132)
        if pose_result and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in landmarks]).flatten()

        keypoints = np.concatenate([rh, lh, pose])
        keypoints_seq.append(keypoints)
        frame_idx += 1

    cap.release()
    hand_lm.close()
    pose_lm.close()

    return np.array(keypoints_seq) if keypoints_seq else None


# ===== شغّل الاستخراج =====
print("🦴 بدء استخراج Landmarks...")
start = time.time()
total_done = 0
total_errors = 0

for idx, (sign_id, sign_path, num_vids) in enumerate(sign_folders):
    save_dir = os.path.join(EXTRACTED_PATH, sign_id)

    # تخطي لو اتعمل قبل كده
    if os.path.exists(save_dir):
        existing = len(glob.glob(os.path.join(save_dir, '*.npy')))
        if existing >= num_vids * 0.8:
            print(f"⏭️ [{idx+1}/{len(sign_folders)}] {sign_id}: موجود ({existing} ملف)")
            total_done += existing
            continue

    os.makedirs(save_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(sign_path, '*.mp4')) +
                     glob.glob(os.path.join(sign_path, '*.avi')))

    success = 0
    for vi, vf in enumerate(videos):
        try:
            kp = extract_from_video(vf)
            if kp is not None and len(kp) > 0:
                np.save(os.path.join(save_dir, f"{vi:03d}.npy"), kp)
                success += 1
        except Exception as e:
            total_errors += 1
            if vi == 0:
                print(f"    ❌ خطأ: {e}")

    total_done += success
    elapsed = time.time() - start
    eta = (elapsed / (idx + 1)) * (len(sign_folders) - idx - 1)
    print(f"✅ [{idx+1}/{len(sign_folders)}] {sign_id}: {success}/{len(videos)} | "
          f"إجمالي: {total_done} | متبقي: {eta/60:.0f} دقيقة")

print(f"\n🎉 اكتمل! {total_done} فيديو | أخطاء: {total_errors} | "
      f"الوقت: {(time.time()-start)/60:.1f} دقيقة")


# ======================== CELL 4 ========================
# الخطوة ٢: تجهيز البيانات
# ========================================================

def normalize_sequence(seq, target_len=SEQUENCE_LENGTH):
    if len(seq) == 0:
        return np.zeros((target_len, FEATURES))
    if len(seq) > target_len:
        indices = np.linspace(0, len(seq)-1, target_len, dtype=int)
        return seq[indices]
    elif len(seq) < target_len:
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.vstack([seq, pad])
    return seq

X_all, y_all, class_names = [], [], []

sign_dirs = sorted([d for d in os.listdir(EXTRACTED_PATH)
                     if os.path.isdir(os.path.join(EXTRACTED_PATH, d))])

print(f"📦 {len(sign_dirs)} إشارة جاري تجهيزها...")

for class_idx, sign_id in enumerate(sign_dirs):
    sign_path = os.path.join(EXTRACTED_PATH, sign_id)
    npy_files = sorted(glob.glob(os.path.join(sign_path, '*.npy')))
    class_names.append(sign_id)

    for nf in npy_files:
        seq = np.load(nf)
        normalized = normalize_sequence(seq)
        X_all.append(normalized)
        y_all.append(class_idx)

    if (class_idx + 1) % 10 == 0:
        print(f"   {class_idx+1}/{len(sign_dirs)}...")

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
print(f"\n📊 X={X_all.shape}, y={y_all.shape}, فئات={len(class_names)}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

np.save(os.path.join(PROCESSED_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(PROCESSED_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(PROCESSED_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_PATH, 'y_test.npy'), y_test)
with open(os.path.join(PROCESSED_PATH, 'class_names.json'), 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("✅ تم!")


# ======================== CELL 5 ========================
# الخطوة ٣: بناء نموذج Transformer
# ========================================================

from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pos = np.arange(seq_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ff_dim, self.dropout_rate = d_model, num_heads, ff_dim, dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.do1(self.att(x, x), training=training)
        x = self.ln1(x + attn)
        ffn = self.do2(self.ffn(x), training=training)
        return self.ln2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads,
                        "ff_dim": self.ff_dim, "dropout": self.dropout_rate})
        return config


def build_model(num_classes, seq_len=30, features=258,
                d_model=256, num_heads=8, ff_dim=512, num_layers=4, dropout=0.3):
    inputs = layers.Input(shape=(seq_len, features))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)
    x = layers.Dropout(dropout)(x)
    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f"transformer_{i}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

num_classes = len(class_names)
model = build_model(num_classes)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ======================== CELL 6 ========================
# الخطوة ٤: التدريب
# ========================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_PATH, 'best_model.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=32,
    callbacks=callbacks, verbose=1
)


# ======================== CELL 7 ========================
# الخطوة ٥: التقييم + الحفظ
# ========================================================

import matplotlib.pyplot as plt

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 دقة الاختبار: {test_acc * 100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, 'training_curves.png'), dpi=150)
plt.show()

DRIVE_SAVE = '/content/drive/MyDrive/KArSL_Model'
os.makedirs(DRIVE_SAVE, exist_ok=True)
model.save(os.path.join(DRIVE_SAVE, 'final_model.keras'))
with open(os.path.join(DRIVE_SAVE, 'class_names.json'), 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(os.path.join(DRIVE_SAVE, 'training_history.json'), 'w') as f:
    json.dump(history_dict, f)
print(f"✅ تم الحفظ على Drive: {DRIVE_SAVE}")


# ======================== CELL 8 ========================
# اختبار سريع
# ========================================================

import random
print("\n🧪 اختبار عشوائي:")
for _ in range(5):
    idx = random.randint(0, len(X_test) - 1)
    pred = model.predict(X_test[idx:idx+1], verbose=0)
    pred_label = np.argmax(pred)
    true_label = y_test[idx]
    conf = pred[0][pred_label] * 100
    print(f"  الحقيقي: {class_names[true_label]} | التوقع: {class_names[pred_label]} | "
          f"الثقة: {conf:.1f}% | {'✅' if true_label == pred_label else '❌'}")
