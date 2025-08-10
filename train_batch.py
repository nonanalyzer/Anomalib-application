import os
import subprocess
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="amd", help="name of test data directory")
    args = parser.parse_args()

    base_dir = args.input
    error_dir_name = "error_dataset_dir"
    num_error_per_class = 5  # 每类选取图片数量，可自行调整

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            print(f'training on class: {class_name}')

            error_dataset_dir = os.path.join(base_dir, error_dir_name)
            if os.path.exists(error_dataset_dir):
                shutil.rmtree(error_dataset_dir)
            os.makedirs(error_dataset_dir)
            # 从其他类别中选取图片
            for other_class in os.listdir(base_dir):
                other_class_path = os.path.join(base_dir, other_class)
                if other_class != class_name and os.path.isdir(other_class_path):
                    imgs = [f for f in os.listdir(other_class_path) if os.path.isfile(os.path.join(other_class_path, f))]
                    import random
                    if len(imgs) > 0:
                        selected_imgs = random.sample(imgs, min(num_error_per_class, len(imgs)))
                        for img in selected_imgs:
                            src_img = os.path.join(other_class_path, img)
                            dst_img = os.path.join(error_dataset_dir, f"{other_class}_{img}")
                            try:
                                shutil.copy2(src_img, dst_img)
                            except Exception as e:
                                print(f"Error copying {src_img} to {dst_img}: {e}")
            
            if os.name == 'posix':
                cmd = f"CUDA_VISIBLE_DEVICES=0 python train_single.py --name {class_name} --input {class_path} --error {error_dataset_dir}"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                cmd = f"python train_single.py --name {class_name} --input {class_path} --error {error_dataset_dir}"
            subprocess.run(cmd, shell=True)

    # 全部训练完成后删除 error_dataset_dir
    if os.path.exists(os.path.join(base_dir, error_dir_name)):
        shutil.rmtree(os.path.join(base_dir, error_dir_name))