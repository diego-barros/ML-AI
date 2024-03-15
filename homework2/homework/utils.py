import csv
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
         
        # classes
        self.label_classes = {'background': 0, 'kart': 1, 'pickup': 2, 'nitro': 3, 'bomb': 4, 'projectile': 5}

        # load the metadata of the dataset
        self.csv_labels = []

        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path, 'labels.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            #no need to skip the first line as DictReader automatically uses it as a header
            for row in reader:
                self.csv_labels.append((row['file'], self.label_classes[row['label']]))

        #transformation - resize to 64,64, 3 channels for RGB are unchanged. ToTensor scales to [0,1]
        self.transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    
        

    def __len__(self):
        #return the length 
        return len(self.csv_labels)

    def __getitem__(self, idx):
        # Retrieve the image and label at the specified index
        file_name, label = self.csv_labels[idx]
        #ensure compatiblity as ToTensor expects RGB images
        image = Image.open(os.path.join(self.dataset_path, file_name)).convert('RGB')
        #image is initally a PIL object, so need to transform it
        image = self.transform(image)
        #convert to integer to facilitate indexing, mem efficiency 
        label = int(label)  
        return image, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
