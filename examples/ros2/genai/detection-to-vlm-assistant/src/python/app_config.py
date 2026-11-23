"""Startup YAML and image conversion helpers; no ROS nodes or hidden inference."""
import argparse
import math
from pathlib import Path

import cv2
import yaml
from image_transport import image_array, planar_array


def read_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    # Labels travel with the config; model/output paths are relative to the cwd.
    labels = Path(config['model']['labels']).expanduser()
    config['model']['labels'] = str(labels if labels.is_absolute() else args.config.parent / labels)
    return config, ros_args


def positive(value, name):
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return number


def positive_int(value, name):
    positive(value, name)
    if not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    return value


def image_rgb(image):
    if image.encoding == 'rgb8':
        return image_array(image).copy()
    if image.encoding == 'bgr8':
        return cv2.cvtColor(image_array(image), cv2.COLOR_BGR2RGB)
    if image.encoding == 'nv12':
        return cv2.cvtColor(planar_array(image), cv2.COLOR_YUV2RGB_NV12)
    raise ValueError('VLM input must be rgb8, bgr8 or packed nv12')
