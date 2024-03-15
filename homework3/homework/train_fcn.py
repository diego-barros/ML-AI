from homework.early_stopping import EarlyStopping
from .models import FCN, save_model
from .dense_transforms import Compose, Resize, RandomHorizontalFlip, RandomCrop, CenterCrop, Normalize, ColorJitter, RandomResizedCrop, ToTensor
from .utils import DENSE_CLASS_DISTRIBUTION, load_dense_data, accuracy, ConfusionMatrix
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from os import path

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # transformations
    transform = Compose([
        Resize((128, 96)),  # Resize images and labels to a fixed size for consistency
        RandomHorizontalFlip(),  # Randomly flip images and labels horizontally
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Randomly adjust color, brightness, and contrast
        ToTensor(),  # Convert images and labels to PyTorch tensors
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        #RandomCrop(96),
        Normalize(mean=[0.2794, 0.2662, 0.2634], std=[0.1729, 0.1629, 0.1820]),
    ])


    # Initialize model
    model = FCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Define the ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# Adjust weights based on class distribution
    class_weights = torch.tensor([1.0 / x for x in DENSE_CLASS_DISTRIBUTION], dtype=torch.float, device=device)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=6, verbose=True)

    #cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Load your data
    train_data = load_dense_data(args.dataset_path, num_workers=4, batch_size=args.batch_size, transform=transform)
    valid_data = load_dense_data(args.valid_dataset_path, num_workers=4, batch_size=args.batch_size, transform=transform)

    print(f'Training dataset size: {len(train_data)}')
    print(f'Validation dataset size: {len(valid_data)}')

    train_logger = tb.SummaryWriter(log_dir=path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(log_dir=path.join(args.log_dir, 'valid'))

    for epoch in range(args.epochs):
        count = 0
        model.train()
        total_loss = 0
        for imgs, labels in train_data:
            imgs, labels = imgs.to(device), labels.to(device)
            #print( f'Count: {count}')
            #count = count + 1
            optimizer.zero_grad()
            output = model(imgs)
            loss = cross_entropy_loss(output, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log training loss
        avg_loss = total_loss / len(train_data)
        train_logger.add_scalar('loss', avg_loss, epoch)
        
        # Validation
        model.eval()
        total_val_loss = 0
        confusion_matrix = ConfusionMatrix(size=5)  # 5 classes
        with torch.no_grad():
            for imgs, labels in valid_data:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                val_loss = cross_entropy_loss(output, labels.long())
                total_val_loss += val_loss.item()
                preds = torch.argmax(output, dim=1)
                confusion_matrix.add(preds, labels)

        avg_val_loss = total_val_loss / len(valid_data)
        valid_logger.add_scalar('loss', avg_val_loss, epoch)
        valid_logger.add_scalar('accuracy', confusion_matrix.global_accuracy, epoch)
        valid_logger.add_scalar('iou', confusion_matrix.iou, epoch)

        print(f'Epoch {epoch}: Train Loss: {avg_loss}, Val Loss: {avg_val_loss}, Accuracy: {confusion_matrix.global_accuracy}, IoU: {confusion_matrix.iou}')

        # Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        # Step the scheduler on each epoch with the validation loss
        scheduler.step(val_loss)
    
    print(f'Final print Accuracy: {confusion_matrix.global_accuracy}, IoU: {confusion_matrix.iou}')
    print(max(min(confusion_matrix.iou, 0.55) - 0.30, 0) / (0.55 - 0.30))
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dense_data/train', help='Path to training dataset')
    parser.add_argument('--valid_dataset_path', type=str, default='dense_data/valid', help='Path to validation dataset')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every n epochs')

    args = parser.parse_args()
    train(args)
