from torch import nn

class MolNet(nn.Module):
    """
    This class is created to specify the Neural Network on which vectorized datasets we have created previously
    is trained on, validated and later tested.
    It consist of one input layer, one output layer and multiple hidden layers.
    ...
    """
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(MolNet, self).__init__()
        # Layer definitions
        self.layers = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Forward pass
        return self.layers(x)