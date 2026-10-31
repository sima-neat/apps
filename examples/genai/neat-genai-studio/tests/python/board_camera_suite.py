"""Board camera (shared/board_camera.py): which /dev node a snapshot reads.
The node passed to the capture tools is the board's own /dev entry, never the
request's string (CodeQL: command-line injection), and no regular expression
runs on request input."""

from __future__ import annotations

import unittest
from unittest import mock

from shared import board_camera


class ResolveCameraNodeTests(unittest.TestCase):
    NODES = ["/dev/video0", "/dev/video16", "/dev/video15m2m"]

    def resolve(self, device):
        with mock.patch.object(board_camera.glob, "glob", return_value=list(self.NODES)):
            return board_camera.resolve_camera_node(device)

    def test_an_existing_node_or_index_resolves_to_the_boards_entry(self):
        self.assertEqual(self.resolve("/dev/video0"), "/dev/video0")
        self.assertEqual(self.resolve("16"), "/dev/video16")

    def test_the_default_is_used_without_a_device(self):
        with mock.patch.dict(board_camera.os.environ, {"NEAT_CAMERA_DEVICE": ""}):
            self.assertEqual(self.resolve(None), board_camera.DEFAULT_DEVICE)

    def test_a_missing_node_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "no camera device /dev/video7"):
            self.resolve("/dev/video7")

    def test_anything_else_is_refused_before_a_command_runs(self):
        for bad in ("/dev/video0; rm -rf /", "/etc/passwd", "--help", "/dev/video" + "0" * 50000 + "x"):
            with self.assertRaises(RuntimeError):
                self.resolve(bad)

    def test_the_capture_command_gets_the_boards_node(self):
        commands = []

        def run(cmd, **_):
            commands.append(cmd)
            raise RuntimeError("no camera in tests")

        with mock.patch.object(board_camera.glob, "glob", return_value=list(self.NODES)), \
                mock.patch.object(board_camera.shutil, "which", side_effect=lambda t: f"/usr/bin/{t}" if t == "ffmpeg" else None), \
                mock.patch.object(board_camera.subprocess, "run", side_effect=run), \
                mock.patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaises(RuntimeError):
                board_camera.capture_camera_frame("/dev/video16")
        self.assertEqual(len(commands), 1)
        self.assertIn("/dev/video16", commands[0])
        self.assertIn(commands[0][commands[0].index("-i") + 1], self.NODES)
