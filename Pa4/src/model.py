import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import alexnet, vgg16
from utils import SelectiveSearch


class ObjectDetectionModel(nn.Module):
    def __init__(self, numclass):
        super(ObjectDetectionModel, self).__init__()
        alex = alexnet(pretrained=True)
        self.features = alex.features
        self.roi_layer = MultiScaleRoIAlign(
            # featmap_names=[0],
            # output_size=7,
            # sampling_ratio=2
        )
        self.classifier = alex.classifier
        self.bbox_reg = nn.Linear(numclass, numclass * 4)

    def forward(self, x):
        feature_map = self.features(x)
        # region_proposals = SelectiveSearch(feature_map)
        # TODO: classify all the region_proposals and regression the final windows
