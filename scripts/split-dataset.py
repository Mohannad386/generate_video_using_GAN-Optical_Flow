import os
import shutil
import random
import argparse

def split_dataset(video_dir, motion_dir, output_video_dir, output_motion_dir, split_ratio=0.1, seed=42):
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_motion_dir, exist_ok=True)

    video_files = os.listdir(video_dir)
    random.seed(seed)
    random.shuffle(video_files)

    #num_to_move = int(len(video_files) * split_ratio)
    random.shuffle(video_files)

    # خذ 10% كـ validation/test
    val_files = video_files[:112]

    for vid_name in val_files:
        # نقل الفيديو
        shutil.move(os.path.join(video_dir, vid_name),
                    os.path.join(output_video_dir, vid_name))
        
        # اسم ملف الحركة المقابل
        motion_name = os.path.splitext(vid_name)[0] + '.pt'
        motion_path = os.path.join(motion_dir, motion_name)
        if os.path.exists(motion_path):
            shutil.move(motion_path, os.path.join(output_motion_dir, motion_name))
        else:
            print(f"[⚠️] ملف الحركة غير موجود: {motion_name}")

    print(f"✅ تم نقل {len(val_files)} فيديو وحقولهم الحركية إلى مجلد التحقق.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="iPER/train", help="مجلد الفيديوهات الأصلية")
    parser.add_argument("--motion_dir", default="iPER/train motion fields", help="مجلد حقول الحركة الأصلية")
    parser.add_argument("--output_video_dir", default="iPER/val", help="المجلد الجديد للفيديوهات")
    parser.add_argument("--output_motion_dir", default="iPER/val motion fields", help="المجلد الجديد لحقول الحركة")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="نسبة البيانات التي ستُنقل")
    parser.add_argument("--seed", type=int, default=42, help="قيمة seed لضمان العشوائية القابلة للتكرار")

    args = parser.parse_args()

    split_dataset(
        video_dir=args.video_dir,
        motion_dir=args.motion_dir,
        output_video_dir=args.output_video_dir,
        output_motion_dir=args.output_motion_dir,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
