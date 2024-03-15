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
            self.features.add_module('resblock{}'.format(i), ResidualBlock(l))
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

#new residual block class to create residual connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class FCN(nn.Module):
    def __init__(self, n_classes=5):
        super(FCN, self).__init__()
        
        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),
            ResidualBlock(32),
            
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(64),
            #nn.Dropout(0.3)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(128),
            #nn.Dropout(0.3)
        )
        
        # Decoder layers
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.decoder2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)
        self.decoder3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        original_size = x.size()[2:]  # Capture original input size
          
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        #x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)  # Resize to match input 
        #out = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        return x

    
model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
