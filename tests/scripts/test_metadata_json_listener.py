import json
import socket
import threading
import time

from tests.utils.metadata_json_listener import MetadataJsonListener


def _payload(metadata_type: str, frame_id: str, timestamp: int) -> bytes:
    data = {"poses": [{}]}
    if metadata_type == "auxiliary-visualization":
        data = {"payload": data}
    return json.dumps(
        {
            "type": metadata_type,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "data": data,
        }
    ).encode()


def test_required_metadata_types_must_share_frame_identity() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with MetadataJsonListener(
        "127.0.0.1",
        port,
        num_ports=1,
        metadata_contracts={
            "pose-estimation": "poses",
            "auxiliary-visualization": "payload.poses",
        },
        metadata_min_counts={
            "pose-estimation": 1,
            "auxiliary-visualization": 1,
        },
    ) as listener:
        def send_messages() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                for payload in (
                    _payload("pose-estimation", "camera0:1", 1000),
                    _payload("auxiliary-visualization", "camera0:2", 1033),
                    _payload("pose-estimation", "camera0:2", 1033),
                ):
                    sender.sendto(payload, ("127.0.0.1", port))
                    time.sleep(0.02)

        thread = threading.Thread(target=send_messages)
        thread.start()
        result = listener.wait_for_messages(1.0)
        thread.join()

    assert result.success
    assert len(result.messages) == 3
    correlated_types = {
        message.metadata_type
        for message in result.messages
        if (message.timestamp_ms, message.frame_id) == (1033, "camera0:2")
    }
    assert correlated_types == {"pose-estimation", "auxiliary-visualization"}
