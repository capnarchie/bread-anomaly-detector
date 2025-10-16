# from roboflow import Roboflow

# project = rf.workspace("breadlabels").project("bread-segment-tlsak")
# version = project.version(5)
# dataset = version.download("coco-segmentation")


# from rfdetr import RFDETRNano

# model = RFDETRNano()

# model.train(
#     dataset_dir='./bread-segment-5',
#     epochs=10,
#     batch_size=4,
#     grad_accum_steps=4,
#     lr=1e-4,
#     output_dir='./output'
# )

import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano(num_classes=1, pretrain_weights="output/checkpoint_best_total.pth")

model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = "test_imgs/test_1.png"
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)