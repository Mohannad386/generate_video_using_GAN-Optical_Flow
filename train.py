from torchvision.io import read_video
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.utils import flow_to_image
from torch import nn
import os
import csv
from perceptual_loss import VGGLoss
from utils.utils import get_batch
from utils.utils import get_ready_batch
from utils.utils import make_video

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

loss_file_default = config["logs_losses"]

def train(start_epoch,
          num_epochs,
          iteration,
          discriminator,
          generator,
          sample_from_same_video,
          batch_size,
          num_frames,
          ready_batch,
          d_opt,
          g_opt,
          dataset_directory,
          motion_fields_directory,
          logger,
          preloaded_videos,
          image_size,
          show_freq,
          checkpoints_freq,
          device):
    """
    Trains the generator and discriminator.
    Args:
        start_epoch (int): Number of start epoch, 0 if not training from a checkpoint.
        num_epochs (int): Number of training epochs.
        iteration (int): Number of training iterations, 0 if not training from a checkpoint.
        discriminator (PyTorch model): The discriminator to be trained.
        generator (PyTorch model): The generator to be trained.
        sample_from_same_video (int): Number of times to sample from the same video each iteration.
        batch_size (int): Number of training sequences.
        num_frames (int): Length of sequence (number of frames in each seequence).
        ready_batch (bool): True if motion fields are already stored.
        d_opt (PyTorch optimizer): Discriminator optimizer.
        g_opt (PyTorch optimizer): Generator optimizer.
        dataset_directory (string): Dataset directory.
        motion_fields_directory (string): Motion fields directory if already stored.
        logger (log): Log file.
        preloaded_videos (bool): True if videos of same length and can be put in RAM.
        image_size (int): Image size.
        show_freq (int): Frequency of saving loss and resulting images.
        checkpoints_freq (int): Frequency of checkpoints saving.
        device (string): Type of device, cpu or cuda.
    """
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    if not ready_batch:
        detector = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                           progress=False).to(device)

        detector = detector.eval()
        model = model.eval()

        model = nn.DataParallel(model)
        detector = nn.DataParallel(detector)

    g_avg_loss = 0
    d_avg_loss = 0
    div = 0

    g_his = []
    d_his = []

    #edit 1 
    log_path = loss_file_default
    # تأكد من عدم إعادة الكتابة
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "g_loss1", "g_loss2", "g_loss3", "g_loss_total", "d_loss"])

    disc_loss = nn.BCELoss()
    gen_loss = nn.MSELoss()
    perceptual_loss = VGGLoss(device)

    names = []
    for name in os.listdir(dataset_directory):
        names.append(name)

    names = np.array(names, dtype= object)
    print(len(names))
    print(names[:5])

    if ready_batch:
        m_names = []
        for name in os.listdir(motion_fields_directory):
            m_names.append(name)

        m_names = np.array(m_names, dtype=object)
        print(len(m_names))
        print(m_names[:5])
        
    print(preloaded_videos)
    if preloaded_videos:
        num_frames_per_video = read_video(os.path.join(
            dataset_directory, names[0]), output_format="TCHW", pts_unit='sec')[0].shape[0]
        all_videos = torch.empty(
            (len(names), num_frames_per_video, 3, image_size, image_size)).to(torch.uint8)

        for i in range(len(names)):
            all_videos[i] = read_video(os.path.join(
                dataset_directory, names[i]), output_format="TCHW", pts_unit='sec')[0].to(torch.uint8)
        print(len(all_videos))
    for epoch in range(start_epoch, num_epochs):
        indices = torch.randperm(len(names))

        for vid in range(0, len(names)):
          if preloaded_videos:
              videos = all_videos[indices[vid: min(
                  vid + batch_size, len(all_videos))]]
          else:
            current_video = read_video(os.path.join(dataset_directory, names[vid]), output_format="TCHW", pts_unit='sec')[0].unsqueeze(0)
            videos = current_video.to(device)

          if ready_batch:
            motion_fields = []
            for idx in indices[vid: min(vid + batch_size, len(m_names))]:
                name = m_names[idx]
                path = os.path.join(motion_fields_directory, name)
                if not os.path.exists(path):
                    print(f"[⚠️] File not found: {path}")
                    continue  # أو raise Error("Missing file")
                motion_fields.append(torch.load(path))
            motion_fields = torch.cat(
                motion_fields).reshape(-1, 2, image_size, image_size)

          for i in range(sample_from_same_video):
                # Discriminator Training

                if ready_batch:
                    source_images, img1_batch, img2_batch, dense_motion_field_batch = get_ready_batch(batch_size,
                                                                                                      num_frames,
                                                                                                      videos,
                                                                                                      motion_fields,
                                                                                                      image_size,
                                                                                                      device)
                else:
                    source_images, img1_batch, img2_batch, dense_motion_field_batch = get_batch(batch_size,
                                                                                                num_frames,
                                                                                                videos,
                                                                                                model,
                                                                                                detector,
                                                                                                image_size,
                                                                                                device)


                source_images.requires_grad = True
                img1_batch.requires_grad = True
                img2_batch.requires_grad = True
                dense_motion_field_batch.requires_grad = True

                # Getting predictions on 5 consecutive generated frames
                generated_vids = make_video(
                    generator, source_images, dense_motion_field_batch, batch_size, num_frames, image_size, device, show=False)

                input_d1 = torch.cat((img2_batch, img1_batch, dense_motion_field_batch),
                                     dim=2).reshape(-1, 8, image_size, image_size)
                input_d2 = torch.cat((generated_vids, img1_batch, dense_motion_field_batch),
                                     dim=2).reshape(-1, 8, image_size, image_size)

                fake_predictions = discriminator(input_d2.detach())
                real_predictions = discriminator(input_d1)

                fake_loss = disc_loss(
                    fake_predictions, torch.zeros_like(fake_predictions))
                real_loss = disc_loss(
                    real_predictions, torch.ones_like(real_predictions))

                d_loss = (fake_loss + real_loss) / 2
                d_avg_loss += d_loss.item()

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                # Generator Training
                if ready_batch:
                    source_images, img1_batch, img2_batch, dense_motion_field_batch = get_ready_batch(batch_size,
                                                                                                      num_frames,
                                                                                                      videos,
                                                                                                      motion_fields,
                                                                                                      image_size,
                                                                                                      device)
                else:
                    source_images, img1_batch, img2_batch, dense_motion_field_batch = get_batch(batch_size,
                                                                                                num_frames,
                                                                                                videos,
                                                                                                model,
                                                                                                detector,
                                                                                                image_size,
                                                                                                device)

                # Getting predictions on 5 consecutive generated frames
                generated_vids = make_video(
                    generator, source_images, dense_motion_field_batch, batch_size, num_frames, image_size, device, show=False)

                input_d = torch.cat((generated_vids, img1_batch, dense_motion_field_batch),
                                    dim=2).reshape(-1, 8, image_size, image_size)

                fake_predictions = discriminator(input_d)

                g_loss1 = gen_loss(generated_vids, img2_batch)
                g_loss2 = disc_loss(
                    fake_predictions, torch.ones_like(fake_predictions))
                g_loss3 = perceptual_loss(
                    generated_vids.view(-1, 3, image_size, image_size), img2_batch.view(-1, 3, image_size, image_size))

                g_loss = g_loss1 + g_loss2 + g_loss3
                g_avg_loss += g_loss.item()

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                div += 1

          d_his.append(d_avg_loss / div)
          g_his.append(g_avg_loss / div)

          if iteration % show_freq == 0:
              print(iteration)
              print('Discriminator Loss: ', d_avg_loss / div)
              print('Generator Loss: ', g_avg_loss / div)
              print(f"g_loss1 (MSE): {g_loss1.item():.4f}")
              print(f"g_loss2 (BCE): {g_loss2.item():.4f}")
              print(f"g_loss3 (VGG): {g_loss3.item():.4f}")
              #print(f"g_loss4 (LPIPS): {g_loss4.item():.4f}")

              #edit 5 save Losses in CSV_file
              
              with open(log_path, mode='a', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow([
                      iteration,
                      g_loss1.item(),
                      g_loss2.item(),
                      g_loss3.item(),
                      g_loss.item(),
                      d_loss.item()
                  ])

              for b in range(batch_size):
                  fig = plt.figure(figsize=(12, 6))

                  # --- الصف العلوي: الرسم البياني للخسائر ---
                  ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
                  ax1.plot(g_his, label="g_loss", color='blue')
                  ax1.plot(d_his, label="d_loss", color='orange')
                  ax1.set_title("Training Losses")
                  ax1.set_xlabel("Iteration")
                  ax1.set_ylabel("Loss")
                  ax1.legend()
                  ax1.grid(True)

                  # --- الصف السفلي: الصور ---
                  ax2 = plt.subplot2grid((2, 4), (1, 0))
                  img1 = generated_vids[b, -1].detach().to('cpu')
                  ax2.imshow(img1.permute(1, 2, 0))
                  ax2.set_title("Generated")
                  ax2.axis('off')

                  ax3 = plt.subplot2grid((2, 4), (1, 1))
                  ax3.imshow(img1_batch[b, -1].detach().permute(1, 2, 0).to('cpu'))
                  ax3.set_title("Input 1")
                  ax3.axis('off')

                  ax4 = plt.subplot2grid((2, 4), (1, 2))
                  ax4.imshow(img2_batch[b, -1].detach().permute(1, 2, 0).to('cpu'))
                  ax4.set_title("Target")
                  ax4.axis('off')

                  ax5 = plt.subplot2grid((2, 4), (1, 3))
                  ax5.imshow(flow_to_image(dense_motion_field_batch[b, -1]).detach().permute(1, 2, 0).to('cpu'))
                  ax5.set_title("Motion")
                  ax5.axis('off')

                  plt.tight_layout()
                  plt.savefig(f'log/loss_{iteration}_{b}.png')
                  plt.close()
          iteration += 1

        if epoch % checkpoints_freq == 0:
            logger.info(f"Epoch: {epoch}\nIteration: {iteration}")
            torch.save(generator.module.state_dict(
            ), 'checkpoints/g_checkpoint' + str(epoch // checkpoints_freq) + '.pth')
            torch.save(discriminator.module.state_dict(
            ), 'checkpoints/d_checkpoint' + str(epoch // checkpoints_freq) + '.pth')
