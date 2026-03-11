import numpy as np
import cv2
import os
import pyneat

def apply_colormap(segmentation):
    colormap = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
        [128, 128, 0], [0, 128, 0], [0, 128, 128], [0, 0, 128],
        [128, 0, 128], [192, 192, 192], [128, 128, 128], [64, 64, 64],
        [255, 165, 0], [0, 255, 127]
    ], dtype=np.uint8)
    
    color_segmentation = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for class_value, color in enumerate(colormap):
        color_segmentation[segmentation == class_value] = color
    return color_segmentation


def find_first_tensor(sample: pyneat.Sample):
    """Find the first tensor in a sample (handles bundles)."""
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.fields:
        for field in sample.fields:
            t = find_first_tensor(field)
            if t is not None:
                return t
    return None

def tensor_to_numpy(tensor: pyneat.Tensor) -> np.ndarray:
    dtype_map = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")
    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


model_path = 'assets/models/nnunet2d_final_optimized_mpk.tar.gz'
image_path = "assets/images/word_0018/slice_159.png"
output_path = "tmp_output_folder/output_159.png"



opt = pyneat.ModelOptions()
opt.format = "RGB"
model = pyneat.Model(model_path, opt)

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
image_np = np.array(image).astype(np.float32)
image_infer = np.expand_dims(image_np, axis=(0, 1))
input_3ch = np.tile(image_infer, (1, 3, 1, 1))
arr = np.ascontiguousarray(input_3ch, dtype=np.uint8)
print(arr.shape, arr.dtype)
input_tensor = pyneat.Tensor.from_numpy(
                arr, copy=True, image_format=pyneat.PixelFormat.BGR)
out = model.run(input_tensor, timeout_ms=5000)
out_tensor = find_first_tensor(out)
