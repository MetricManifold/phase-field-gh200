"""Wire-format and damaged-stream checks for the public velocity reader."""
from io import BytesIO
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from velocity_moments import HEADER, frames, read_header, spatial_samples


def stream_bytes() -> bytes:
    header = HEADER.pack(b"PFVMOM4\0", 1, 26, 0.01, 37, 701.0, 100, 2, 17)
    result = bytearray(header)
    for step, samples in [(37, 0), (54, 17), (100, 17), (101, 18), (205, 19)]:
        row = [17, 10.0, 12.0, 0.4, 1.0, 0.5, 0.01, 100.0, step-37]
        row += [0.0]*12 + [samples, 0.03, -0.04, 0.01, 0.02]
        result += struct.pack("<q26d", step, *row)
    return bytes(result)


def consume(raw: bytes):
    stream = BytesIO(raw)
    header = read_header(stream)
    return header, list(frames(stream, header))


valid = stream_bytes()
header, records = consume(valid)
assert len(records) == 5 and records[-1].step == 205
for stride in (1, 3, 25, 100, 997):
    for start in (0, 37, 9000037):
        for dense in (0, 17, 100):
            from dataclasses import replace
            h = replace(header, spatial_stride=stride, start_step=start, dense_start=dense)
            for count in (0, 1, 2, 37, 101, 603):
                expected = sum(n == 0 or n < dense or (start+n) % stride == 0
                               for n in range(count))
                assert spatial_samples(h, count) == expected

bad_stride = bytearray(valid)
struct.pack_into("<I", bad_stride, 40, 0)
bad_count = bytearray(valid)
struct.pack_into("<d", bad_count, HEADER.size + 8 + 21*8, 1)
bad_origin = bytearray(valid)
struct.pack_into("<d", bad_origin, HEADER.size + 8 + 9*8, 1)
for invalid in (valid[:40], valid[:HEADER.size], valid[:-1],
                valid + b"x", bad_stride, bad_count, bad_origin):
    try:
        consume(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged stream accepted")
print("PASS: PFVMOM4 readback, unaligned sample counts, invalid/truncated records")
