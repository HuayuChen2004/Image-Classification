import os
import cv2
import argparse

def read_jpg_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"The file {file_path} is not a valid image.")
    return image

def read_jpg_files_in_directory(directory_path, num_files=None):
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The path {directory_path} is not a directory.")
    jpg_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    if num_files is not None:
        jpg_files = jpg_files[:num_files]
    images = []
    for file in jpg_files:
        file_path = os.path.join(directory_path, file)
        image = read_jpg_file(file_path)
        images.append(image)
    label = directory_path.split('/')[-1]
    return images, label

def resize_images(images, size=(384, 384)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, size)
        resized_images.append(resized_img)
    return resized_images

def letterbox_resize_images(images, size=(384, 384), color=(0, 0, 0)):
    target_w, target_h = size
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_top, pad_bottom,
            pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )
        resized.append(img_padded)
    return resized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the resized images")
    parser.add_argument("--num_files", type=int, default=None, help="Number of files to read from the directory")
    parser.add_argument("--size", type=int, nargs=2, default=(384, 384), help="Size to resize the images to")
    parser.add_argument("--color", type=int, nargs=3, default=(0, 0, 0), help="Color for padding")
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    for sub in os.listdir(data_dir):
        path = os.path.join(data_dir, sub)
        if not os.path.isdir(path):
            continue
        images, _ = read_jpg_files_in_directory(path)
        images = letterbox_resize_images(images, size=args.size, color=tuple(args.color))
        # save the resized images
        save_path = os.path.join(save_dir, sub)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, img in enumerate(images):            
            save_file = os.path.join(save_path, f"{i}.jpg")
            cv2.imwrite(save_file, img)
    print(f"Resized images saved to {save_dir}")