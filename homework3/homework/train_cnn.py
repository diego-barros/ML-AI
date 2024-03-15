from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, accuracy, LABEL_NAMES
from .early_stopping import EarlyStopping
import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args):
    from os import path
     
    model = CNNClassifier(layers=[32, 64, 128], n_input_channels=3, kernel_size=3)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    #optmizer - from homework1
    optimizer =  torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    #cross entropy initialization, doing it directly in the function throws error
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    cross_entropy = torch.nn.CrossEntropyLoss()

    train_data = load_data(args.dataset_path, batch_size=args.batch_size)
    valid_data = load_data(args.valid_dataset_path, batch_size=args.batch_size)

    global_step = 0
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)

    final_acc_vals, final_vacc_vals = [], []
    #adapted from previous exercise
    for epoch in range(args.epochs):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = cross_entropy(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            #log the training loss with tensorboard
            train_logger.add_scalar('loss', loss_val, global_step)

            #increment
            global_step = global_step + 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        train_logger.add_scalar('accuracy', avg_acc, epoch)
        model.eval()  

        val_loss_vals, vacc_vals = [], []
        with torch.no_grad():
            for img, label in valid_data:
                img, label = img.to(device), label.to(device)
                output = model(img)
                val_loss = cross_entropy(output, label)
                val_loss_vals.append(val_loss.item())
                vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        
        avg_val_loss = sum(val_loss_vals) / len(val_loss_vals)
        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        valid_logger.add_scalar('loss', avg_val_loss, epoch)
        valid_logger.add_scalar('accuracy', avg_vacc, epoch)

        # Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
          
        #log the accuracy
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f},Training Accuracy: {avg_acc:.2f}, Validation Accuracy: {avg_vacc:.2f}')
        final_acc_vals.append(avg_acc)
        final_vacc_vals.append(avg_vacc)

        scheduler.step(val_loss)

    #log final values
    print(f'Epoch: {epoch}, Final Training Accuracy: {sum(final_acc_vals)/args.epochs:.2f}, Final Validation Accuracy: {sum(final_vacc_vals)/args.epochs:.2f}')
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size for training')
    parser.add_argument('--valid_dataset_path', type=str, default='data/valid', help='Path to validation dataset')
    parser.add_argument('--dataset_path', type=str, default='data/train', help='Path to training dataset')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for log')

    args = parser.parse_args()
    train(args)
