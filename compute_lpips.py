import lpips
from torchvision.io import read_video
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import os
import re
import csv
import traceback
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import copy
from utils import preprocess
from utils import get_bounding_boxes
from utils import motion_field_resize
from networks.generator import Generator
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", default="iPER/val", help="path to dataset")
    parser.add_argument("--motion_fields_path", default=r"iPER/val motion fields",
                        help="path to motion fields if available")
    parser.add_argument("--ready_batch", default=True,
                        help="True if motion fields are already extracted and stored in motion_fields_path")
    parser.add_argument("--checkpoint", default="checkpoints/g_checkpoint381.pth",
                        help="path to checkpoint to restore")
    parser.add_argument("--preloaded_videos", default=False,
                        help="load all dataset to RAM, all videos must be of the same length")
    parser.add_argument("--image_size", default=128, help="size of image")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")

    opt = parser.parse_args()

    directory = opt.dataset_path
    motion_fields_path = opt.motion_fields_path
    ready_batch = opt.ready_batch
    checkpoint = opt.checkpoint
    preloaded_videos = opt.preloaded_videos
    image_size = opt.image_size
    device = opt.device

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    detector = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device='cuda')
    detector = detector.eval()

    model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                       progress=False).to(device)
    model = model.eval()

    g = Generator(num_channels_image=4, num_channels_motion=3, block_expansion=64,
                  max_features=512, num_down_blocks=2, num_bottleneck_blocks=6).to(device)
    g.load_state_dict(torch.load(checkpoint))

    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    loss = 0.0

    if preloaded_videos:
        names = []
        for name in os.listdir(directory):
            names.append(name)

        names = np.array(names)

        num_frames_per_video = read_video(os.path.join(
            directory, names[0]), output_format="TCHW", pts_unit='sec')[0].shape[0]
        all_videos = torch.empty(
            (len(names), num_frames_per_video, 3, image_size, image_size)).to(torch.uint8)

        for i in range(len(names)):
            all_videos[i] = read_video(os.path.join(
                directory, names[i]), output_format="TCHW", pts_unit='sec')[0].to(torch.uint8)

        n_frames = 0
        step = 8

        torch.cuda.manual_seed(88)
        image_noise = torch.randn(
            (step, 1, image_size, image_size), device=device)

        torch.cuda.manual_seed(57)
        motion_noise = torch.randn(
            (step, 1, image_size, image_size), device=device)

        for vid in range(0, len(names), step):
            frames = all_videos[vid: vid + step]
            with torch.no_grad():
                source, _ = transforms(
                    frames[:, 0], torch.zeros_like(frames[:, 0]))
                source = (source.to(device) + 1) / 2

                for i in range(len(frames[0]) - 1):
                    try:
                        img1_batch = frames[:, i]
                        img2_batch = frames[:, i + 1]
                        img1_batch, img2_batch = preprocess(
                            img1_batch, img2_batch, 128)
                        img1_batch = img1_batch.to(device)
                        img2_batch = img2_batch.to(device)

                        dense_motion_field = model(
                            img1_batch, img2_batch)[:][-1]

                        boxes1 = get_bounding_boxes(img1_batch, detector)
                        boxes2 = get_bounding_boxes(source, detector)

                        field = torch.zeros_like(dense_motion_field)
                        new_dense_motion_field = torch.zeros_like(
                            dense_motion_field)

                        for b in range(step):
                            box1 = boxes1[b]
                            box2 = boxes2[b]

                            field[b, :, box1[1]: box1[3], box1[0]: box1[2]
                                  ] = dense_motion_field[b, :, box1[1]: box1[3], box1[0]: box1[2]]
                            new_size = [(box2[3] - box2[1]).item(),
                                        (box2[2] - box2[0]).item()]

                            new_dense_motion_field[b, :, box2[1]: box2[3], box2[0]: box2[2]] = motion_field_resize(
                                field[b, :, box1[1]: box1[3], box1[0]: box1[2]], new_size, device)

                        input_g = (torch.cat((source, image_noise), dim=1), torch.cat(
                            (new_dense_motion_field, motion_noise), dim=1))
                        frame = g(*input_g)

                        source = copy.deepcopy(frame)

                        pair_loss = loss_fn_vgg(
                            frame.to(device), img2_batch.to(device))
                        loss += torch.sum(pair_loss)
                        n_frames += step
                    except Exception as e:
                        print(f"❌ Exception while processing video index {vid}, frame {i}: {e}")
                        traceback.print_exc()
                print('Frames: ', n_frames)
                print(loss / n_frames)

    else:
        names = []
        for name in os.listdir(directory):
            names.append(name)

        names = np.array(names)

        torch.cuda.manual_seed(88)
        image_noise = torch.randn(
            (8, 1, image_size, image_size), device=device)

        torch.cuda.manual_seed(57)
        motion_noise = torch.randn(
            (8, 1, image_size, image_size), device=device)

        n_frames = 0
        step = 1

        torch.cuda.manual_seed(88)
        image_noise = torch.randn(
            (step, 1, image_size, image_size), device=device)

        torch.cuda.manual_seed(57)
        motion_noise = torch.randn(
            (step, 1, image_size, image_size), device=device)

        for vid in range(0, len(names), step):
            frames = read_video(os.path.join(
                directory, names[vid]), output_format="TCHW", pts_unit='sec')[0].unsqueeze(0)
            with torch.no_grad():
                source, _ = transforms(
                    frames[:, 0], torch.zeros_like(frames[:, 0]))
                source = (source.to(device) + 1) / 2

                for i in range(len(frames)):
                    try:
                        img1_batch = frames[:, i]
                        img2_batch = frames[:, i + 1]
                        img1_batch, img2_batch = preprocess(
                            img1_batch, img2_batch, 128)
                        img1_batch = img1_batch.to(device)
                        img2_batch = img2_batch.to(device)

                        dense_motion_field = model(
                            img1_batch, img2_batch)[:][-1]

                        boxes1 = get_bounding_boxes(img1_batch, detector)
                        boxes2 = get_bounding_boxes(source, detector)

                        field = torch.zeros_like(dense_motion_field)
                        new_dense_motion_field = torch.zeros_like(
                            dense_motion_field)

                        for b in range(step):
                            box1 = boxes1[b]
                            box2 = boxes2[b]

                            field[b, :, box1[1]: box1[3], box1[0]: box1[2]
                                  ] = dense_motion_field[b, :, box1[1]: box1[3], box1[0]: box1[2]]
                            new_size = [(box2[3] - box2[1]).item(),
                                        (box2[2] - box2[0]).item()]

                            new_dense_motion_field[b, :, box2[1]: box2[3], box2[0]: box2[2]] = motion_field_resize(
                                field[b, :, box1[1]: box1[3], box1[0]: box1[2]], new_size, device)

                        input_g = (torch.cat((source, image_noise), dim=1), torch.cat(
                            (new_dense_motion_field, motion_noise), dim=1))
                        frame = g(*input_g)

                        source = copy.deepcopy(frame)

                        pair_loss = loss_fn_vgg(
                            frame.to(device), img2_batch.to(device))
                        loss += torch.sum(pair_loss)
                        n_frames += step
                    except:
                        print('Error')
                print('Frames: ', n_frames)
                print(loss / n_frames)
                

    print('Total frames: ', n_frames)
    print('Total Loss: ', loss)
    print('LPIPS: ', loss / n_frames)

    #edit 1

    results_csv = "lpips_results.csv"
    header = ['Checkpoint', 'LPIPS', 'Total_Loss', 'Frames']
    
    checkpoint_number = re.findall(r'\d+', checkpoint)
    checkpoint_number = int(checkpoint_number[0]) if checkpoint_number else -1

    # إذا لم يكن الملف موجودًا، نضيف رأس الجدول
    file_exists = os.path.isfile(results_csv)

    with open(results_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([checkpoint_number, float(loss / n_frames), float(loss), int(n_frames)])

    print(f"✅ تم حفظ النتيجة في {results_csv}")
