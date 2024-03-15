import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[32, 64, 128], n_input_channels=3, kernel_size=3):
        
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential()
        
        c = n_input_channels
        for i, l in enumerate(layers):
            self.features.add_module('conv{}'.format(i), nn.Conv2d(c, l, kernel_size, padding=kernel_size//2))
            #normalizing the images to try to increase accuracy as well
            self.features.add_module('batchnorm{}'.format(i), nn.BatchNorm2d(l))
            self.features.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            self.features.add_module('pool{}'.format(i), nn.MaxPool2d(kernel_size=2, stride=2))
            c = l

        # Adaptive pooling to ensure the output from convolutional layers is of the correct size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            #adding dropout to increase accuracy, as it was less than 30%
            nn.Dropout(0.5),
            nn.Linear(layers[-1], 6)  # Assuming the last layer size is connected to 6 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling to flatten the output regardless of input size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
