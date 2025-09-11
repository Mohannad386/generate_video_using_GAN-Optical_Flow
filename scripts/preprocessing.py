import torchvision
from torchvision.io import read_video
import torch
import cv2
import numpy as np
import os
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", default=r"iPER/iPER", help="path to dataset")
    parser.add_argument("--new_dataset_path",
                        default="iPER/train", help="path to new directory")
    parser.add_argument("--image_size", default=128, help="size of image")
    parser.add_argument("--seperate_video", default=True,
                        help="True if we want to seperate each video to new videos each of length frames_per_vid and False otherwise")
    parser.add_argument("--frames_per_video", default=200,
                        help="number of frames of each seperated video")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")

    opt = parser.parse_args()

    directory = opt.dataset_path
    new_directory = opt.new_dataset_path
    size = opt.image_size
    seperate_video = opt.seperate_video
    frames_per_video = opt.frames_per_video
    device = opt.device

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    # Resizing video frames

    for name in os.listdir(directory):
        vid_s, _, _ = read_video(os.path.join(
            directory, name), output_format="TCHW", pts_unit='sec')
        vid = torch.empty(vid_s.shape[0], size, size, 3)

        for i in range(vid.shape[0]):
            vid[i] = torchvision.transforms.functional.resize(
                vid_s[i], size=[size, size], antialias=False).permute(1, 2, 0)

        num_frames, height, width, channels = vid.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20

        if seperate_video:
            # Seperating each video to smaller videos of equal lenghts

            for i in range(num_frames // frames_per_video):
                out = cv2.VideoWriter(os.path.join(new_directory, name.split(
                    '.')[0] + '_' + str(i) + '.mp4'), fourcc, fps, (width, height))
                for j in range(frames_per_video):
                    frame = vid[i * frames_per_video + j]
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        image = frame.numpy().astype(np.uint8)
                    else:
                        print(f"[⚠️] Skipping invalid frame with shape {frame.shape}")
                        continue
                    out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


                out.release()

        else:
            out = cv2.VideoWriter(os.path.join(
                new_directory, name), fourcc, fps, (width, height))

            for i in range(num_frames):
                frame = vid[i]
                image = frame.numpy().astype(np.uint8)
                out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            out.release()
