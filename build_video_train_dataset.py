import os
import random
import shutil
import cv2
import numpy as np

# ==============================================================================
# SCRIPT CONFIGURATION AND FUNCTIONS
# ==============================================================================

def apply_video_augmentations(frame, h_flip, v_flip, brightness_factor):
    """
    Applies selected augmentations to a single video frame.

    Args:
        frame (np.array): The input video frame.
        h_flip (bool): Whether to apply horizontal flip.
        v_flip (bool): Whether to apply vertical flip.
        brightness_factor (float): Factor to adjust brightness. 1.0 means no change.

    Returns:
        np.array: The augmented frame.
    """
    if h_flip:
        frame = cv2.flip(frame, 1)
    if v_flip:
        frame = cv2.flip(frame, 0)

    # Adjust brightness using the HSV color space
    if brightness_factor != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Apply brightness factor and clip values to the valid [0, 255] range
        v = cv2.multiply(v, np.array([brightness_factor]))
        v = np.clip(v, 0, 255)
        # Merge channels and convert back to BGR
        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return frame


def augment_and_save_video(original_path, output_path):
    """
    Reads a video, applies random augmentations, and saves it to a new file.

    Args:
        original_path (str): The path to the source video.
        output_path (str): The path where the augmented video will be saved.
    """
    # Decide on random augmentations for the entire video
    h_flip = random.random() > 0.5
    v_flip = random.random() > 0.5  # Note: Vertical flip might not be suitable for all datasets
    brightness_factor = random.uniform(0.7, 1.3)

    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {original_path}")
        return

    # Get video properties to create a new video with the same settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 output

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        augmented_frame = apply_video_augmentations(frame, h_flip, v_flip, brightness_factor)
        out.write(augmented_frame)

    cap.release()
    out.release()


def split_video_dataset_balanced(input_dir, output_root, train_ratio=0.7):
    """
    Splits and augments a video dataset, ensuring a balanced distribution across classes.

    Args:
        input_dir (str): Path to the root directory of the original dataset.
        output_root (str): Path to the output directory where 'train' and 'val' folders will be created.
        train_ratio (float): The proportion of the dataset to allocate to the training set.
    """
    train_dir = os.path.join(output_root, "train")
    test_dir = os.path.join(output_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Total classes to process: {len(class_names)}\n")

    for class_idx, class_name in enumerate(class_names):
        print(f"--- Processing Class {class_idx + 1}/{len(class_names)}: {class_name} ---")
        class_path = os.path.join(input_dir, class_name)

        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        file_list = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in video_extensions]
        random.shuffle(file_list)
        file_count = len(file_list)

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        print(f"Found {file_count} videos in class '{class_name}'.")

        # Scenario 1: Very large class
        if file_count > 3000:
            print("Large class detected. Capping train set and augmenting test set if needed.")
            train_files = random.sample(file_list, 3000)
            test_files = list(set(file_list) - set(train_files))
            for f in train_files: shutil.copy(os.path.join(class_path, f), os.path.join(train_class_dir, f))
            for f in test_files: shutil.copy(os.path.join(class_path, f), os.path.join(test_class_dir, f))

            if len(test_files) < 900:
                augment_needed = 900 - len(test_files)
                print(f"Test set insufficient. Augmenting {augment_needed} videos for test set...")
                for i in range(augment_needed):
                    base_video = random.choice(test_files)
                    file_name, file_ext = os.path.splitext(base_video)
                    new_name = f"{file_name}_aug_test_{i}{file_ext}"
                    augment_and_save_video(os.path.join(class_path, base_video), os.path.join(test_class_dir, new_name))

        # Scenario 2: Very small class
        elif 1 <= file_count <= 100:
            print("Small class detected. Augmenting to create more train and test samples.")
            for f in file_list: shutil.copy(os.path.join(class_path, f), os.path.join(train_class_dir, f))

            min_train_samples = 50
            if file_count < min_train_samples:
                augment_needed = min_train_samples - file_count
                print(f"Training set insufficient. Augmenting {augment_needed} videos for training set...")
                for i in range(augment_needed):
                    base_video = random.choice(file_list)
                    file_name, file_ext = os.path.splitext(base_video)
                    new_name = f"{file_name}_aug_train_{i}{file_ext}"
                    augment_and_save_video(os.path.join(class_path, base_video), os.path.join(train_class_dir, new_name))

            min_test_samples = 50
            print(f"Creating {min_test_samples} augmented videos for the test set...")
            for i in range(min_test_samples):
                base_video = random.choice(file_list)
                file_name, file_ext = os.path.splitext(base_video)
                new_name = f"{file_name}_aug_test_{i}{file_ext}"
                augment_and_save_video(os.path.join(class_path, base_video), os.path.join(test_class_dir, new_name))

        # Scenario 3: Medium class
        elif 100 < file_count <= 3000:
            print("Medium class detected. Splitting into train/test sets.")
            split_idx = int(file_count * train_ratio)
            train_files = file_list[:split_idx]
            test_files = file_list[split_idx:]
            for f in train_files: shutil.copy(os.path.join(class_path, f), os.path.join(train_class_dir, f))
            for f in test_files: shutil.copy(os.path.join(class_path, f), os.path.join(test_class_dir, f))
            print(f"Split dataset: {len(train_files)} for train, {len(test_files)} for test.")

        print(f"--- Finished processing class: {class_name} ---\n")

    print("✅ All processing complete.")


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- IMPORTANT: YOU MUST CHANGE THESE PATHS ---
    #
    # Set the path to the folder containing your raw video class folders (e.g., "Cat", "Dog").
    # For Windows, it might look like: "C:/Users/YourUser/Desktop/MyVideos"
    # For Mac/Linux, it might look like: "/home/user/my_videos"
    #
    input_directory = "/path/to/your/raw_video_dataset"

    #
    # Set the path where you want the new "train" and "val" folders to be created.
    # For Windows, it might look like: "C:/Users/YourUser/Desktop/ProcessedVideos"
    # For Mac/Linux, it might look like: "/home/user/processed_videos"
    #
    output_directory = "/path/to/your/processed_dataset"


    # --- Run the main function ---
    # No need to edit anything below this line.
    split_video_dataset_balanced(
        input_dir=input_directory,
        output_root=output_directory,
        train_ratio=0.7
    )
