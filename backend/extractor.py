"""
الخطوة ١: استخراج الـ Landmarks من فيديوهات KArSL
محدّث ليتوافق مع هيكل الفولدرات الفعلي:
  train/0171-0190/0171/*.mp4
"""
import os
import sys
import cv2
import numpy as np
import glob
import time

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
except ImportError:
    print("❌ مكتبة mediapipe غير مثبتة!")
    print("   شغّل: pip install mediapipe")
    sys.exit(1)


def mediapipe_detection(image, model):
    """تحويل الألوان + تشغيل MediaPipe"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results


def extract_keypoints(results):
    """
    استخراج 258 feature من كل فريم:
    - يد يمين: 21 نقطة × 3 = 63
    - يد شمال: 21 نقطة × 3 = 63
    - جسم: 33 نقطة × 4 = 132
    """
    rh = np.array([[res.x, res.y, res.z] for res in 
                    results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(63)
    
    lh = np.array([[res.x, res.y, res.z] for res in 
                    results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(63)
    
    # Use WORLD landmarks (meters, centered at hip) for much better depth/position data
    # Fall back to regular pose_landmarks if world landmarks unavailable
    if results.pose_world_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                          results.pose_world_landmarks.landmark]).flatten()
    elif results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                          results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(132)
    
    return np.concatenate([rh, lh, pose])


def process_video(video_path):
    """يقرأ فيديو ويستخرج landmarks من كل فريم"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠️ فشل فتح: {os.path.basename(video_path)}")
        return None
    
    keypoints_seq = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = mediapipe_detection(frame, holistic)
            kp = extract_keypoints(results)
            keypoints_seq.append(kp)
    
    cap.release()
    
    if len(keypoints_seq) == 0:
        return None
    
    return np.array(keypoints_seq)


def main():
    # ===== المسارات =====
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'karsl_videos', 'train')
    SAVE_PATH = os.path.join(BASE_DIR, '..', 'data', 'extracted_landmarks')
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ مسار البيانات غير موجود: {DATA_PATH}")
        return
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # ===== اكتشاف هيكل الفولدرات =====
    # الهيكل: train/0171-0190/0171/*.mp4
    all_sign_folders = []
    
    for range_folder in sorted(os.listdir(DATA_PATH)):
        range_path = os.path.join(DATA_PATH, range_folder)
        if not os.path.isdir(range_path):
            continue
        
        for sign_folder in sorted(os.listdir(range_path)):
            sign_path = os.path.join(range_path, sign_folder)
            if os.path.isdir(sign_path):
                all_sign_folders.append((sign_folder, sign_path))
    
    if not all_sign_folders:
        # ممكن الهيكل يكون مباشر: train/0171/*.mp4
        for sign_folder in sorted(os.listdir(DATA_PATH)):
            sign_path = os.path.join(DATA_PATH, sign_folder)
            if os.path.isdir(sign_path):
                all_sign_folders.append((sign_folder, sign_path))
    
    total_signs = len(all_sign_folders)
    print(f"🔥 تم العثور على {total_signs} إشارة")
    print(f"   مسار الحفظ: {SAVE_PATH}")
    print("=" * 60)
    
    total_videos = 0
    total_errors = 0
    start_time = time.time()
    
    for sign_idx, (sign_id, sign_path) in enumerate(all_sign_folders):
        # مجلد الحفظ لهذه الإشارة
        save_sign_path = os.path.join(SAVE_PATH, sign_id)
        
        # تخطي لو تمت معالجتها من قبل
        if os.path.exists(save_sign_path) and len(os.listdir(save_sign_path)) > 0:
            existing = len([f for f in os.listdir(save_sign_path) if f.endswith('.npy')])
            print(f"⏭️  [{sign_idx+1}/{total_signs}] الإشارة {sign_id}: تمت من قبل ({existing} ملف)")
            total_videos += existing
            continue
        
        os.makedirs(save_sign_path, exist_ok=True)
        
        # جمع الفيديوهات
        videos = sorted(glob.glob(os.path.join(sign_path, '*.mp4')) + 
                        glob.glob(os.path.join(sign_path, '*.avi')))
        
        print(f"\n📂 [{sign_idx+1}/{total_signs}] الإشارة {sign_id}: {len(videos)} فيديو")
        
        sign_success = 0
        for vid_idx, video_file in enumerate(videos):
            try:
                keypoints = process_video(video_file)
                
                if keypoints is not None and len(keypoints) > 0:
                    save_name = os.path.join(save_sign_path, f"{vid_idx:03d}.npy")
                    np.save(save_name, keypoints)
                    sign_success += 1
                    
                    # تحديث بسيط كل 10 فيديوهات
                    if (vid_idx + 1) % 10 == 0:
                        print(f"    ✅ {vid_idx+1}/{len(videos)} فيديو تم")
                else:
                    total_errors += 1
                    
            except Exception as e:
                print(f"    ❌ خطأ في فيديو {vid_idx}: {e}")
                total_errors += 1
        
        total_videos += sign_success
        elapsed = time.time() - start_time
        avg_per_sign = elapsed / (sign_idx + 1)
        remaining = avg_per_sign * (total_signs - sign_idx - 1)
        
        print(f"    ✅ {sign_success}/{len(videos)} | إجمالي: {total_videos} | متبقي: {remaining/60:.0f} دقيقة")
    
    # ===== ملخص =====
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ اكتمل الاستخراج!")
    print(f"   إجمالي الفيديوهات المعالجة: {total_videos}")
    print(f"   الأخطاء: {total_errors}")
    print(f"   الوقت: {elapsed/60:.1f} دقيقة")
    print(f"   الحفظ في: {SAVE_PATH}")


if __name__ == '__main__':
    main()
