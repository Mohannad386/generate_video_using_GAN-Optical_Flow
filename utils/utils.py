import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import Raft_Large_Weights
import os
import torchvision
import logging


def preprocess(img1_batch, img2_batch, size):
    """
    Resizes and transforms images to become in range between -1 and 1.
    Args:
        img1_batch (ndarray): Batch of images.
        img2_batch (ndarray): Batch of images.
        size (int): New image size.
    Returns:
        tuple: (preprocessed img_batch1, preprocessed img_batch2).
    """

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    img1_batch = torchvision.transforms.functional.resize(
        img1_batch, size=size, antialias=False)
    img2_batch = torchvision.transforms.functional.resize(
        img2_batch, size=size, antialias=False)
    return transforms(img1_batch, img2_batch)

def get_bounding_boxes(imgs, detector):
    """
    Finds the ground truth boxes of people in images.
    Args:
        imgs (ndarray): Batch of images to find people in.
        detector (PyTorch model): Human detector.
    Returns:
        list: Bounding boxes for people in imgs.
    """

    predictions = detector(imgs)
    boxes = []
    for prediction in predictions:
        boxes.append(torch.tensor(
            [prediction['boxes'][0][i].to(torch.uint8).item() for i in range(4)]))

    return boxes


def crop_motion_fields(motion_fields, boxes, image_size=128, device='cuda'):
    """
    Crops the motion field to contain only the movement of a person based on bounding boxes.
    Args:
        motions_fields (ndarray): Motion fields between two image batches.
        boxes (list): Bounding boxes of peopel.
        image_size (int): Size of motioon_fields.
        device (string): Type of device, cpu or cuda.
    Returns:
        ndarray: Cropped motion fields.
    """

    mask = torch.zeros(
        (len(motion_fields), 2, image_size, image_size)).to(device)

    for i in range(len(motion_fields)):
        box = boxes[i]
        mask[i, :, box[1]: box[3], box[0]: box[2]
             ] = motion_fields[i, :, box[1]: box[3], box[0]: box[2]]

    return mask


def motion_field_resize(field, size, device):
    """
    Resizes motion field based on the size of a bounding box for a person.
    Args:
        field (ndarray): Motion field to be resized.
        size (tuple): (length, width) of the new size.
        device (string): Type of device, cpu or cuda.
    Returns:
        ndarray: resized field.
    """

    field = torch.cat(
        [field, torch.ones(1, *field.shape[1:]).to(device)], axis=0)
    field = torchvision.transforms.functional.resize(
        field, size=size, antialias=False)
    return field[0:2, :, :].unsqueeze(0)


def make_video(g, sources, dense_motion_field_batch, batch_size, num_frames, image_size=128, device='cuda', show=False):
    """
    Makes a sequence of generated frames.
    Args:
        g (PyTorch model): Generator model.
        sources (ndarray): Initial images.
        dense_motion_field_batch (ndarray): Motion fields.
        batch_size (int): Number of initial images and sequences to be generated.
        num_frames (int): Length of sequence (number of frames to be generated in each seequence).
        image_size (int): Size of motioon_fields.
        device (string): Type of device, cpu or cuda.
        show (bool): To show generated frames or not (used for debugging purposes).
    Returns:
        ndarray: batch_size sequences of generated images of length num_frames.
    """

    vids_frames = torch.Tensor(
        batch_size, num_frames, *sources.shape[1:]).to(device)
    vids_frames[0: batch_size, 0] = sources

    for i in range(num_frames):
        noise_image = torch.randn(
            batch_size, 1, image_size, image_size).to(device)

        noise_motion = torch.randn(
            batch_size, 1, image_size, image_size).to(device)

        input_g = (torch.cat((sources, noise_image), dim=1),
                   torch.cat((dense_motion_field_batch[0: batch_size, i], noise_motion), dim=1))
        frames = g(*input_g)
        vids_frames[0: batch_size, i] = frames

        if show:
            plt.imshow(sources[0].detach().to('cpu').permute(1, 2, 0))
            plt.show()

        sources = frames

    return vids_frames


def get_logger(name, log_file=None):
    """
    Creats or gets a log file.
    Args:
        name (string): Name of log file.
        log_file (string): Name of log folder.
    Returns:
        log: The created or already available log.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file:
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_best_checkpoint(path):
    """
    Finds the best checkpoint based on its number, all checkpoints must have onle one number (sequence of several digits).
    Args:
        path (string): Directory of checkpoints.
    Returns:
        int: The number of best checkpoint.
    """
    import os

    # تأكد أن المجلد موجود وليس فارغًا
    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"⚠️ المجلد '{checkpoint_dir}' غير موجود.")
        return 0

    checkpoint_files = [f for f in os.listdir(path) if f.startswith("g_checkpoint") and f.endswith(".pth")]

    if not checkpoint_files:
        print(f"⚠️ لم يتم العثور على أي Checkpoints في '{checkpoint_dir}'.")
        return 0  

    checkpoint_numbers = []
    for f in checkpoint_files:
        try:
            num_part = f.split("g_checkpoint")[-1].split(".pth")[0]
            if num_part.strip().isdigit():
                checkpoint_numbers.append(int(num_part.strip()))
        except ValueError:
            continue  

    if not checkpoint_numbers:
        print(f"⚠️ لم يتم العثور على أرقام Checkpoints صحيحة في '{checkpoint_dir}'.")
        return 0  

    best_checkpoint = max(checkpoint_numbers)  # الحصول على أحدث Checkpoint
    print(f"✅ أفضل Checkpoint موجود: {best_checkpoint}")
    return best_checkpoint

    #m = 0
    #for name in os.listdir(path):
       # n = ''
      #  for i in name:
        #    if i.isdigit():
       #         n += i
      #  n = int(n)
     #   m = max(m, n)
    #return m


def get_batch(batch_size, num_frames, video, model, detector, image_size, device):
    """
    Generates the training batch for discriminater and generator.
    Args:
        batch_size (int): Number of initial images.
        num_frames (int): Length of sequence (number of frames in each seequence).
        video (ndarray): The video to get the suquences from.
        model (PyTorch model): Motion detector model.
        detector (PyTorch model): Human detector.
        image_size (int): Size of image.
        device (string): Type of device, cpu or cuda.
    Returns:
        tuple: The training batch (source_images, img1_batch, img2_batch, dense_motion_field_batch)
    """

    with torch.no_grad():
        len_frames = video.shape[1]

        dense_motion_field_batch = torch.empty(
            batch_size, num_frames, 2, image_size, image_size).to(device)

        start = torch.randint(0, len_frames - num_frames - 1, (batch_size,))
        source_images = video[0, start].to(dtype=torch.float32, device=device)

        indices = torch.cat([torch.arange(i, i + num_frames) for i in start])

        img1_batch = video[0, indices]
        img2_batch = video[0, indices + 1]

        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)

        img1_batch, img2_batch = preprocess(img1_batch, img2_batch, image_size)

        dense_motion_field_batch = model(img1_batch, img2_batch)[:][-1]

        boxes = get_bounding_boxes(img2_batch, detector)
        dense_motion_field_batch = crop_motion_fields(
            dense_motion_field_batch, boxes)

        img1_batch = (img1_batch + 1) / 2
        img2_batch = (img2_batch + 1) / 2

        return (source_images,
                img1_batch.reshape(batch_size, num_frames,
                                   3, image_size, image_size),
                img2_batch.reshape(batch_size, num_frames,
                                   3, image_size, image_size),
                dense_motion_field_batch.reshape(
                    batch_size, num_frames, 2, image_size, image_size)
                )


def get_ready_batch(batch_size, num_frames, video, motion_fields, image_size, device):
    """
    Retrieves the training batch for discriminater and generator.
    Args:
        batch_size (int): Number of initial images.
        num_frames (int): Length of sequence (number of frames in each seequence).
        video (ndarray): The video to get the suquences from.
        motion_fields (ndarray): Already stored motion fields of video.
        image_size (int): Size of image.
        device (string): Type of device, cpu or cuda.
    Returns:
        tuple: The training batch (source_images, img1_batch, img2_batch, dense_motion_field_batch)
    """

    with torch.no_grad():
        len_frames = video.shape[1]

        dense_motion_field_batch = torch.empty(
            batch_size, num_frames, 2, image_size, image_size)

        start = torch.randint(0, len_frames - num_frames - 1, (batch_size,))
        source_images = video[0, start].to(dtype=torch.float32, device=device)

        indices = torch.cat([torch.arange(i, i + num_frames) for i in start])

        img1_batch = video[0, indices]
        img2_batch = video[0, indices + 1]

        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)

        img1_batch, img2_batch = preprocess(img1_batch, img2_batch, image_size)

        img1_batch = (img1_batch + 1) / 2
        img2_batch = (img2_batch + 1) / 2

        dense_motion_field_batch = motion_fields[indices].to(device)

        return (source_images,
                img1_batch.reshape(batch_size, num_frames,
                                   3, image_size, image_size),
                img2_batch.reshape(batch_size, num_frames,
                                   3, image_size, image_size),
                dense_motion_field_batch.reshape(
                    batch_size, num_frames, 2, image_size, image_size)
                )