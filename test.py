from torchvision.io import read_video
from torchvision.io import read_image
import os
import cv2
import torch
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import copy
from argparse import ArgumentParser
from networks.generator import Generator
from utils.utils import get_bounding_boxes
from utils.utils import motion_field_resize
from utils.utils import preprocess

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

source_image = config["Test"]["source_image"]
driven_video = config["Test"]["driving_video"]
output_video = config["Test"]["output_video"]
checkpoint = config["Test"]["checkpoint"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--source_image", default="Test/s.jpg",
                        help="path to source image")
    parser.add_argument("--driving_video", default="Test/d.mp4",
                        help="path to driving video")
    parser.add_argument("--checkpoint", default="checkpoints/g_checkpoint375.pth",
                        help="path to generator checkpoint to use to make video")
    parser.add_argument("--image_size", default=128, help="size of image")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")

    opt = parser.parse_args()

    source_image = opt.source_image
    driving_video = opt.driving_video
    checkpoint = opt.checkpoint
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
    g.load_state_dict(torch.load(checkpoint, weights_only=True))

    frames, _, _ = read_video(
        driving_video, output_format="TCHW", pts_unit='sec')

    torch.cuda.manual_seed(88)
    image_noise = torch.randn((1, 1, image_size, image_size), device=device)

    torch.cuda.manual_seed(57)
    motion_noise = torch.randn((1, 1, image_size, image_size), device=device)

    with torch.no_grad():
        source = read_image(source_image)
        source, _ = transforms(source, source)
        source = (source.to(device) + 1) / 2
        source = source.unsqueeze(0)
        vid_frames = [source]

        try:
            for i in range(len(frames) - 1):
                img1_batch = frames[i].unsqueeze(0)
                img2_batch = frames[i + 1].unsqueeze(0)
                img1_batch, img2_batch = preprocess(
                    img1_batch, img2_batch, image_size)
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)

                dense_motion_field = model(img1_batch, img2_batch)[:][-1]

                box1 = get_bounding_boxes(img1_batch, detector)[0]
                box2 = get_bounding_boxes(source, detector)[0]

                field = torch.zeros_like(dense_motion_field)
                field[0, :, box1[1]: box1[3], box1[0]: box1[2]
                      ] = dense_motion_field[0, :, box1[1]: box1[3], box1[0]: box1[2]]

                new_size = [(box2[3] - box2[1]).item(),
                            (box2[2] - box2[0]).item()]

                new_dense_motion_field = torch.zeros_like(dense_motion_field)

                new_dense_motion_field[0, :, box2[1]: box2[3], box2[0]: box2[2]] = motion_field_resize(
                    field[0, :, box1[1]: box1[3], box1[0]: box1[2]], new_size, device)

                input_g = (torch.cat((source, image_noise), dim=1), torch.cat(
                    (new_dense_motion_field, motion_noise), dim=1))
                frame = g(*input_g)
                vid_frames.append(frame.detach().to('cpu'))

                source = copy.deepcopy(frame)

        except:
            pass

    print(len(vid_frames))

    shape = vid_frames[0].shape
    vid_frames[0] = vid_frames[0].to('cpu')
    vid = torch.Tensor(len(vid_frames), *shape)
    #vid = torch.cat(vid_frames, out=vid).permute(0, 2, 3, 1)
    vid_frames = [frame.squeeze(1) if frame.dim() == 5 else frame for frame in vid_frames]  # إزالة البعد الزائد فقط إذا كان موجودًا
    vid = torch.cat(vid_frames, dim=0).permute(0, 2, 3, 1)


    num_frames, height, width, channels = vid.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec to use
    fps = 20  # Specify the frame rate of the output video
    out = cv2.VideoWriter('Test/result.mp4', fourcc, fps, (width, height))

    # Loop over the frames in the tensor, convert each frame to an image,
    # and write the image to the output video
    for i in range(num_frames):
        #     print(vid[i].min())
        frame = vid[i] * 255
        image = frame.numpy().astype(np.uint8)
        out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Release the VideoWriter object and close the output file
    out.release()
