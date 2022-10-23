import torch.nn as nn
from utils import SelectiveSearch


class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.conv_layers = nn.Sequential(

        )
        self.rio_layer = None
        self.linear_layers = nn.Sequential(

        )
        self.softmax_layer = nn.Softmax()

    def forward(self, input_):
        feature_map = self.conv_layers(input_)
        region_proposals = SelectiveSearch(feature_map)
        # TODO: classify all the region_proposals and regression the final windows
