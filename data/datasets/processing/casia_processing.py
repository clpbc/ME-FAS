import os, cv2, math
from tqdm import tqdm

def extract_frames(video_path, frame_ratios):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame indices to extract
    frames_index = [math.floor(total_frames * ratio) for ratio in frame_ratios]

    frames = []

    for frame_index in frames_index :
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames


def get_video_path(root_path, ends_list):
    video_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(tuple(ends_list)):
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def casia_transfer_img(original_video_path, output_dir):
    for video_path in tqdm(original_video_path):
        directory, filename = os.path.split(video_path)
        parts = directory.split(os.sep)

        new_folder_path = os.path.join(output_dir, 'train' if parts[-2] == 'train_release' else 'test', 'real' if filename.startswith(('1', '2', 'HR_1')) else 'fake')
        os.makedirs(new_folder_path, exist_ok=True)

        frames = extract_frames(video_path, (1 / 4, 3 / 4))
        base_filename = f"{parts[-1]}_{filename.split('.')[0]}"
        if len(frames) == 2:
            cv2.imwrite(os.path.join(new_folder_path, f"{base_filename}_frame1.jpg"), frames[1])
        else :
            print(f"{base_filename}_frame1.png Error")

        if len(frames) >= 1:
            cv2.imwrite(os.path.join(new_folder_path, f"{base_filename}_frame0.jpg"), frames[0])
        else :
            print(f"{base_filename}_frame0.png Error")

if __name__ == '__main__':
    original_video_path = get_video_path(r"/media/disk2/cailvpan/FaceAnti-spoofing/CASIA-FASD", ['.avi'])

    casia_transfer_img(original_video_path, r'/home/wanghaowei/cailvpan/frame/casia')
