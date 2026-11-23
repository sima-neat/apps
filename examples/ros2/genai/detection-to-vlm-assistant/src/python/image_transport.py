"""Explicit sensor_msgs/Image packing; no cv_bridge or private message fields."""
from array import array

import cv2
import numpy as np
import pyneat
from sensor_msgs.msg import Image


def image_array(message):
    """Return a read-only-compatible view while the message owns its storage."""
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    count = channels[message.encoding]
    if message.width < 1 or message.height < 1 or message.step < message.width * count:
        raise ValueError("Invalid image geometry/row stride")
    if len(message.data) != message.height * message.step:
        raise ValueError("Image payload length does not match height * step")
    view = np.ndarray((message.height, message.width, count), dtype=np.uint8,
                      buffer=message.data, strides=(message.step, count, 1))
    return view[:, :, 0] if count == 1 else view


def to_message(pixels, stamp, frame_id, encoding="bgr8"):
    from sensor_msgs.msg import Image
    pixels = np.ascontiguousarray(pixels)
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    expected = {"bgr8": (np.dtype("uint8"), 3), "rgb8": (np.dtype("uint8"), 3),
                "mono8": (np.dtype("uint8"), 1), "32FC1": (np.dtype("float32"), 1)}
    if encoding not in expected or (pixels.dtype, channels) != expected[encoding]:
        raise ValueError(f"Pixels do not match {encoding}: {pixels.dtype}, {pixels.shape}")
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = pixels.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = pixels.strides[0]
    data = array("B")
    data.frombytes(memoryview(pixels).cast("B"))
    msg.data = data
    return msg


def header_dict(header):
    return {"stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
            "frame_id": header.frame_id}


def stamp_key(header):
    return (header.frame_id, header.stamp.sec, header.stamp.nanosec)


def planar_array(message):
    """View the documented packed NV12/NV16 wire format without copying."""
    if message.encoding not in ("nv12", "nv16"):
        raise ValueError("Expected nv12 or nv16")
    width, height = message.width, message.height
    if width < 2 or height < 2 or width % 2 or height % 2 or message.step != width:
        raise ValueError("Planar images require even visible geometry and packed rows")
    rows = height * 3 // 2 if message.encoding == "nv12" else height * 2
    if len(message.data) != rows * width:
        raise ValueError("Planar image payload length does not match its visible geometry")
    return np.frombuffer(message.data, np.uint8).reshape(rows, width)


def image_tensor(message, memory=None):
    """Copy ROS pixels to an owned Neat tensor, including explicit Y/UV planes.

    nv12/nv16 use visible Image height and packed planes. This is a documented
    adapter convention, not a claim that every sensor_msgs/Image consumer
    accepts planar encodings. BGR/RGB consumers use the source's bgr8 option.
    """
    import pyneat
    memory = pyneat.TensorMemory.EV74 if memory is None else memory
    if message.encoding in ("bgr8", "rgb8", "mono8"):
        fmt = {"bgr8": pyneat.PixelFormat.BGR, "rgb8": pyneat.PixelFormat.RGB,
               "mono8": pyneat.PixelFormat.GRAY8}[message.encoding]
        return pyneat.Tensor.from_numpy(np.ascontiguousarray(image_array(message)),
                                       copy=True, image_format=fmt, memory=memory)
    if message.encoding not in ("nv12", "nv16"):
        raise ValueError(f"Unsupported Neat input encoding: {message.encoding}")
    width, height = message.width, message.height
    uv_rows = height // 2 if message.encoding == "nv12" else height
    pixels = planar_array(message)
    fmt = pyneat.PixelFormat.NV12 if message.encoding == "nv12" else pyneat.PixelFormat.NV16
    # Establish planar metadata before transferring to EV74: the device copy
    # maps Y/UV and cannot infer their offsets from a two-dimensional array.
    tensor = pyneat.Tensor.from_numpy(pixels, copy=True, image_format=fmt,
                                    memory=pyneat.TensorMemory.CPU)
    planes = []
    for role, rows, offset in ((pyneat.PlaneRole.Y, height, 0),
                               (pyneat.PlaneRole.UV, uv_rows, width * height)):
        plane = pyneat.Plane()
        plane.role, plane.shape = role, [rows, width]
        plane.strides_bytes, plane.byte_offset = [width, 1], offset
        planes.append(plane)
    tensor.shape, tensor.strides_bytes = [height, width], []
    tensor.axis_semantics = [pyneat.TensorAxisSemantic.H, pyneat.TensorAxisSemantic.W]
    tensor.planes = planes
    tensor.read_only = False
    return tensor.cvu() if memory == pyneat.TensorMemory.EV74 else tensor


def sample_tensors(sample):
    """Return model outputs in native order, retaining the owning Sample."""
    if sample.fields:
        return [tensor for field in sample.fields for tensor in sample_tensors(field)]
    if sample.tensors:
        return list(sample.tensors)
    if sample.tensor is not None:
        return [sample.tensor]
    raise ValueError("Inference sample contains no output tensors")


def pack_image(tensor, stamp, frame_id, output_encoding):
    width, height = int(tensor.width()), int(tensor.height())
    if tensor.is_nv12() or tensor.image_format() == pyneat.PixelFormat.NV16:
        nv12 = tensor.is_nv12()
        native_encoding = "nv12" if nv12 else "nv16"
        if output_encoding not in ("native", native_encoding, "bgr8"):
            raise ValueError("Requested planar encoding differs from decoder output")
        uv_rows = height // 2 if nv12 else height
        bgr_output = output_encoding == "bgr8"
        message = Image()
        message.header.stamp, message.header.frame_id = stamp, frame_id
        message.width, message.height = width, height
        message.step = width * 3 if bgr_output else width
        message.encoding = "bgr8" if bgr_output else native_encoding
        # Own the final ROS buffer up front; packing/conversion writes directly into it.
        size = height * width * 3 if bgr_output else (height + uv_rows) * width
        message.data = array("B", [0]) * size
        pixels = np.frombuffer(message.data, np.uint8)
        packed = (np.empty((height + uv_rows, width), np.uint8) if bgr_output
                  else pixels.reshape(height + uv_rows, width))
        payload = tensor.copy_payload_bytes()
        planes = {plane.role: plane for plane in tensor.planes}
        for role, rows, row0 in ((pyneat.PlaneRole.Y, height, 0),
                                 (pyneat.PlaneRole.UV, uv_rows, height)):
            plane = planes[role]
            stride = int(plane.strides_bytes[0]) if plane.strides_bytes else width
            offset = int(plane.byte_offset)
            if stride < width or offset < 0 or offset + (rows - 1) * stride + width > len(payload):
                raise ValueError("Decoder plane extends beyond its backing payload")
            packed[row0:row0 + rows] = np.ndarray((rows, width), np.uint8, buffer=payload,
                                                 offset=offset, strides=(stride, 1))
        if bgr_output:
            bgr = pixels.reshape(height, width, 3)
            if nv12:
                cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12, dst=bgr)
            else:
                cv2.cvtColor(np.dstack([packed[:height], packed[height:]]),
                             cv2.COLOR_YUV2BGR_YUY2, dst=bgr)
        return message
    raise ValueError("Live source expected NV12 or NV16 decoder output")
