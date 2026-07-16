"""
الخطوة ٢: تجهيز البيانات
يقرأ ملفات .npy المستخرجة ويحولها لشكل جاهز للتدريب
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

# ===== الإعدادات =====
EXTRACTED_PATH = os.path.join('..', 'data', 'extracted_landmarks')
PROCESSED_PATH = os.path.join('..', 'data', 'processed')
SEQUENCE_LENGTH = 30   # عدد الفريمات الثابت لكل فيديو
FEATURES_PER_FRAME = 258  # 63 + 63 + 132

def normalize_sequence(seq, target_len=SEQUENCE_LENGTH):
    """
    توحيد طول الفيديو لـ 30 فريم:
    - لو أطول: نأخذ فريمات موزعة بالتساوي
    - لو أقصر: نضيف أصفار في النهاية
    """
    if len(seq) == 0:
        return np.zeros((target_len, FEATURES_PER_FRAME))
    
    if len(seq) > target_len:
        # أخد فريمات موزعة بالتساوي
        indices = np.linspace(0, len(seq) - 1, target_len, dtype=int)
        return seq[indices]
    elif len(seq) < target_len:
        # padding بأصفار
        pad = np.zeros((target_len - len(seq), seq.shape[1]))
        return np.vstack([seq, pad])
    else:
        return seq

def augment_sequence(seq):
    """
    Data Augmentation — نكبّر الداتا بتنويعات:
    1. إضافة ضوضاء خفيفة
    2. تسريع/تبطيء
    3. عكس (mirror)
    """
    augmented = []
    
    # ١. ضوضاء خفيفة
    noise = seq + np.random.normal(0, 0.005, seq.shape)
    augmented.append(noise)
    
    # ٢. تبطيء (أخد كل فريم تاني وعمل interpolation)
    slow = np.repeat(seq, 2, axis=0)
    augmented.append(normalize_sequence(slow))
    
    return augmented

def load_labels_mapping(labels_path):
    """
    يقرأ ملف Labels ويربط كل رقم إشارة باسمها العربي
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(labels_path)
        ws = wb.active
        mapping = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is not None:
                sign_id = str(row[0]).zfill(4)  # "1" → "0001"
                sign_name = str(row[1]) if row[1] else f"sign_{sign_id}"
                mapping[sign_id] = sign_name
        return mapping
    except ImportError:
        print("⚠️ مكتبة openpyxl غير مثبتة. شغّل: pip install openpyxl")
        print("  سنستخدم أرقام الفولدرات كأسماء...")
        return {}
    except Exception as e:
        print(f"⚠️ خطأ في قراءة Labels: {e}")
        return {}

def main():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # ===== قراءة أسماء الإشارات =====
    labels_file = os.path.join('..', 'data', 'karsl_videos', 'KARSL-502_Labels.xlsx')
    label_mapping = load_labels_mapping(labels_file)
    
    # ===== تجميع كل البيانات =====
    X_all = []  # المدخلات
    y_all = []  # التصنيفات
    class_names = []  # أسماء الإشارات
    
    if not os.path.exists(EXTRACTED_PATH):
        print(f"❌ مسار البيانات المستخرجة غير موجود: {EXTRACTED_PATH}")
        print("   شغّل extractor.py الأول!")
        return
    
    sign_folders = sorted([f for f in os.listdir(EXTRACTED_PATH) 
                          if os.path.isdir(os.path.join(EXTRACTED_PATH, f))])
    
    if not sign_folders:
        print("❌ لا توجد إشارات مستخرجة. شغّل extractor.py الأول!")
        return
    
    print(f"📦 عدد الإشارات: {len(sign_folders)}")
    
    for class_idx, sign_folder in enumerate(sign_folders):
        sign_path = os.path.join(EXTRACTED_PATH, sign_folder)
        npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        
        # اسم الإشارة بالعربي (من Labels) أو رقمها
        sign_name = label_mapping.get(sign_folder, sign_folder)
        class_names.append(sign_name)
        
        for npy_file in npy_files:
            filepath = os.path.join(sign_path, npy_file)
            seq = np.load(filepath)
            
            # توحيد الطول
            normalized = normalize_sequence(seq)
            X_all.append(normalized)
            y_all.append(class_idx)
            
            # Data Augmentation (اختياري — نفعّله لو الداتا قليلة)
            # for aug_seq in augment_sequence(normalized):
            #     X_all.append(aug_seq)
            #     y_all.append(class_idx)
        
        print(f"  ✅ {sign_name} ({sign_folder}): {len(npy_files)} عينة")
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    
    print(f"\n📊 شكل البيانات الكلي: X={X_all.shape}, y={y_all.shape}")
    print(f"   عدد الفئات: {len(class_names)}")
    
    # ===== تقسيم البيانات =====
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📂 التقسيم:")
    print(f"   Train: {X_train.shape[0]} عينة")
    print(f"   Validation: {X_val.shape[0]} عينة")
    print(f"   Test: {X_test.shape[0]} عينة")
    
    # ===== الحفظ =====
    np.save(os.path.join(PROCESSED_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_PATH, 'y_test.npy'), y_test)
    
    # حفظ أسماء الإشارات
    with open(os.path.join(PROCESSED_PATH, 'class_names.json'), 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ تم الحفظ في: {PROCESSED_PATH}")
    print(f"   ملفات: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")
    print(f"   أسماء الإشارات: class_names.json")

if __name__ == '__main__':
    main()
