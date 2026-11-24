from tests.utils.rtp_video import pairs, rtp_timestamps


def test_rtp_timestamps_read_bytes_4_to_8_big_endian() -> None:
    header = bytes([0x80, 0x60, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04]) + bytes(4)

    assert rtp_timestamps([header]) == {0x01020304}


def test_pairs_across_the_2_to_the_32_wrap() -> None:
    # 0 ms is 0 ticks, 45 ticks after an RTP timestamp just below 2^32.
    assert pairs(0, {2**32 - 45})


def test_pairs_rejects_a_frame_more_than_1_ms_away() -> None:
    # 1000 ms is 90,000 ticks, 91 ticks from the frame.
    assert not pairs(1000, {90_000 + 91})


def test_pairs_accepts_a_frame_exactly_1_ms_away() -> None:
    assert pairs(1000, {90_000 + 90})
