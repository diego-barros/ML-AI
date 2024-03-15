from matplotlib import pyplot as plt
import torch
import torch.utils.tensorboard as tb
import argparse
from homework import dense_transforms


# Assuming the necessary imports for Detector, save_model, and load_detection_data
from .models import Detector, FocalLoss, save_model
from .utils import load_detection_data


def log(logger, imgs, gt_det, det, global_step):
    """
    Log function for TensorBoard visualization.
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)



def train(args):
    from os import path
    model = Detector()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data(args.dataset_path, num_workers=4, batch_size=args.batch_size, transform=transform)

    #tried focal loss - made it worse
    #tried LR scheduler, made it worse as well
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        print('Start of Epoch # %-3d' % (epoch))
        for img, heatmap, bounding_box_size in train_data:
            img, heatmap, bounding_box_size = img.to(device), heatmap.to(device), bounding_box_size.to(device)

            size_w, _ = heatmap.max(dim=1, keepdim=True)

            det, size = model(img)
            
            # Emphasize loss for incorrect object locations and consider confidence
            p_det = torch.sigmoid(det * (1 - 2 * heatmap))

            # Focus size loss on existing objects and calculate average error
            size_loss_val = (size_w * torch.nn.functional.mse_loss(size, bounding_box_size)).mean()

            # Combine losses with a weight for size prediction importance
            loss_val = (loss(det, heatmap) * p_det).mean() / p_det.mean() + size_loss_val * 0.001

            if train_logger is not None and global_step % 2 == 0:
                log(train_logger, img, heatmap, det, global_step)

            if train_logger is not None:
                train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        print('End of Epoch # %-3d' % (epoch))
        save_model(model)

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

        
    parser.add_argument('--transform', default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('--dataset_path', type=str, default='dense_data/train', help='Path to training dataset')
    parser.add_argument('--valid_dataset_path', type=str, default='dense_data/valid', help='Path to validation dataset')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every n epochs')
    parser.add_argument('--continue_training', action='store_true')


    args = parser.parse_args()
    train(args)