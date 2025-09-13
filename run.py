import os
import sys
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import os
import torch.optim as optim
import torch
from torch import nn
from argparse import ArgumentParser
from utils.utils import get_logger
from networks.discriminator import Discriminator
from networks.generator import Generator
from train import train
from utils.utils import get_best_checkpoint

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_default = config["data"]["train_videos"]
motion_fields_path_default = config["data"]["train_motion_fields"]
log_dir_default = config["logs"]
checkpoint_default = config["checkpoints"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", default=dataset_path_default, help="path to dataset")
    parser.add_argument("--motion_fields_path", default=motion_fields_path_default,
                        help="path to motion fields if available")
    parser.add_argument("--log_dir", default=log_dir_default, help="path to log")
    parser.add_argument("--checkpoint", default=checkpoint_default,
                        help="path to save or restore checkpoints")
    parser.add_argument("--preloaded_videos", default=True,
                        help="load all dataset to RAM, all videos must be of the same length")
    parser.add_argument("--image_size", default=128, help="size of image")
    parser.add_argument("--sample_from_same_video", default=5,
                        help="number of times we sample from the same video in the same iteration")
    parser.add_argument("--batch_size", default=4,
                        help="number of random source images taken from the same video")
    parser.add_argument("--num_frames", default=5,
                        help="number of sequence frams to train generator to reconstruct")
    parser.add_argument("--ready_batch", default=True,
                        help="True if motion fields are already extracted and stored in motion_fields_path")
    parser.add_argument("--num_epochs", default=500, help="Number of epochs")
    parser.add_argument("--g_lr", default=0.00001,
                        help="generator learning rate")
    parser.add_argument("--d_lr", default=0.00001,
                        help="discriminator learning rate")
    parser.add_argument("--g_betas", default=(0.7, 0.999),
                        help="generator betas for adam optimizer")
    parser.add_argument("--d_betas", default=(0.7, 0.999),
                        help="discriminator betas for adam optimizer")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--checkpoint_freq", default=1,
                        help="frequency of checkpoints saving")
    parser.add_argument("--show_freq", default=100,
                        help="frequency of saving loss and resulting images")

    opt = parser.parse_args()

    dataset_path = opt.dataset_path
    motion_fields_path = opt.motion_fields_path
    log_dir = opt.log_dir
    checkpoint = opt.checkpoint
    preloaded_videos = opt.preloaded_videos
    image_size = opt.image_size
    sample_from_same_video = opt.sample_from_same_video
    batch_size = opt.batch_size
    num_frames = opt.num_frames
    ready_batch = opt.ready_batch
    num_epochs = opt.num_epochs
    g_lr = opt.g_lr
    d_lr = opt.d_lr
    g_betas = opt.g_betas
    d_betas = opt.d_betas
    checkpoint_freq = opt.checkpoint_freq
    show_freq = opt.show_freq
    device = opt.device

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if 'log.log' in os.listdir(log_dir):
        with open(os.path.join(log_dir, 'log.log'), "r") as f:
            log_lines = f.readlines()

        # Process the log lines to extract the information you need
        for line in log_lines:
            if "Epoch" in line:
                start_epoch = int(line.split(":")[-1].strip())
                if start_epoch != 0:
                    start_epoch += 1
            elif "Iteration" in line:
                iteration = int(line.split(":")[-1].strip())
                if iteration != 0:
                    iteration += 1

    else:
        start_epoch = 0
        iteration = 0

    logger = get_logger("log", os.path.join("log", "log.log"))


    d = Discriminator(num_channels=8, block_expansion=32,
                      num_blocks=4, max_features=512, sn=True).to(device)
    d.train()
    g = Generator(num_channels_image=4, num_channels_motion=3, block_expansion=64,
                  max_features=512, num_down_blocks=2, num_bottleneck_blocks=6).to(device)
    g.train()

    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    if len(os.listdir(checkpoint)) != 0:
        m = get_best_checkpoint(checkpoint)
        g.load_state_dict(torch.load(os.path.join(
            checkpoint, 'g_checkpoint' + str(m) + '.pth')))
        d.load_state_dict(torch.load(os.path.join(
            checkpoint, 'd_checkpoint' + str(m) + '.pth')))
        print("Checkpoints loaded successfully: " + str(m))

    g = nn.DataParallel(g)
    d = nn.DataParallel(d)

    d_opt = optim.Adam(d.parameters(), lr=g_lr, betas=g_betas)
    g_opt = optim.Adam(g.parameters(), lr=d_lr, betas=d_betas)

    print("Started training at epoch: " + str(start_epoch))

    train(start_epoch,
          num_epochs,
          iteration,
          d,
          g,
          sample_from_same_video,
          batch_size,
          num_frames,
          ready_batch,
          d_opt,
          g_opt,
          dataset_path,
          motion_fields_path,
          logger,
          preloaded_videos,
          image_size,
          show_freq,
          checkpoint_freq,
          device)
