"""Pack decoder planes using their actual strides, then convert once to BGR."""
from array import array
import cv2
import numpy as np
import pyneat
from sensor_msgs.msg import Image
from image_transport import to_message

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
