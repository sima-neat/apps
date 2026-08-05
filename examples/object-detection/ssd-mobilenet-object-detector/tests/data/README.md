# SSD accuracy reference

`ssd_accuracy_reference.json` contains source-model detections on the ten bundled COCO val2017
test images. Those image IDs are disjoint from both the 128-image post-training quantization
calibration set and the 32-image compiler-validation set used to publish the model matrix.

The V1 reference was decoded from the pinned ONNX Model Zoo SSD-MobileNetV1 source. The V3
reference was decoded from TorchVision `ssdlite320_mobilenet_v3_large` COCO V1. Its BF16 artifacts
retain the exported `[-1,1]` adapter, while the QAT INT8 artifacts expose the ImageNet-normalized
boundary. V2 retains the existing `golden_detections.json` reference produced from TensorFlow
`ssd_mobilenet_v2_coco_2018_03_29`.

The matrix test executes the public Python app for every INT8/BF16 and MLA/non-MLA artifact,
then performs one-to-one same-class matching at IoU 0.45. V3's recall floor accounts for backend
NMS collapsing overlapping source detections; the pinned devkit baseline is 14/20 for BF16 and
15/20 for QAT INT8. The reference is an accuracy fixture, not calibration input.
