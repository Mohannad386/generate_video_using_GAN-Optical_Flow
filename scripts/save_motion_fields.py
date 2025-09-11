from torchvision.io import read_video
import torch
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torch import nn
import os
from argparse import ArgumentParser
from utils import get_bounding_boxes
from utils import crop_motion_fields
from utils import preprocess

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", default="iPER/train", help="path to dataset")
    parser.add_argument("--motion_fields_path", default=r"iPER/train motion fields",
                        help="path to motion fields if available")
    parser.add_argument("--image_size", default=192, help="size of image")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")

    opt = parser.parse_args()

    dataset_path = opt.dataset_path
    motion_fields_path = opt.motion_fields_path
    image_size = opt.image_size
    device = opt.device

    if not os.path.exists(motion_fields_path):
        os.mkdir(motion_fields_path)

    detector = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                       progress=False).to(device)

    detector = detector.eval()
    model = model.eval()

    model = nn.DataParallel(model)
    detector = nn.DataParallel(detector)

    names = []
    for name in os.listdir(dataset_path):
        names.append(name)

    names = np.array(names)

    m_names = []

    for name in os.listdir(motion_fields_path):
        m_names.append(name)

    m_names = np.array(m_names)

    step = 20
    vid = 1
    for name in names:
        if name.split('.')[0] + '.pt' in m_names:
            continue

        frames = read_video(os.path.join(dataset_path, name),
                            output_format="TCHW", pts_unit='sec')[0]
        if frames.shape[0] < 2:
            print(f"[⚠️] Skipping {name} — too few frames: {frames.shape[0]}")
            continue
            
        dense_motion_field = torch.empty(
            frames.shape[0] - 1, 2, image_size, image_size)
        for j in range(0, len(frames) - 1, step):
            with torch.no_grad():
                img1_batch = frames[j: min(j + step, len(frames) - 1)]
                img2_batch = frames[j + 1: min(j + step + 1, len(frames))]

                img1_batch, img2_batch = preprocess(
                    img1_batch, img2_batch, image_size)

                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)

                field = model(img1_batch, img2_batch)[:][-1]
                #edit1
                if field.shape[1:] != (2, image_size, image_size):
                    print(f"[⚠️] Invalid field shape in {name}: {field.shape}")
                    continue


                boxes = get_bounding_boxes(img2_batch, detector, image_size)
                cropped_field = crop_motion_fields(field, boxes)

                dense_motion_field[j: min(
                    j + step, len(frames) - 1)] = cropped_field

        torch.save(dense_motion_field, os.path.join(
            motion_fields_path, name.split('.')[0] + '.pt'))
        print(vid)
        vid += 1
        print(name)
