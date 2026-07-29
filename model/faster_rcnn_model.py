import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Number of classes
# 9 obstacle classes + 1 background
NUM_CLASSES = 10


def get_model():

    # Load pretrained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    # Get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace classifier with our own
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model
if __name__ == "__main__":

    model = get_model()

    print(model)

    print("\nModel Loaded Successfully!")