"""Row capacity of the decode-shaped kernels.

Every weight kernel here has two shapes: a decode tile for "a verify block" (one CTA row-tile, the
weights streamed once) and a prefill shape for everything wider (BLOCK_M=64 tiles, a transient
dequant + cuBLAS, the blocked head). A single request's verify block is at most 16 rows, so the
switch between them has always been `M <= 16`, and prefill chunk invariance depends on a 17..64-row
call staying on the prefill shape: a short final prefill chunk must not change arithmetic because it
happens to fit a decode tile.

Batched multi-request decode (engine/batch_decode.py) stacks every lane's verify block into one
call so each weight is read once for all of them -- 4 lanes x 6 rows is 24 rows, past the old
switch, and on the prefill shape a dense call would re-dequantize the whole weight per step. It
raises the limit with `wide()` only while its own graphs are captured; the default never changes,
so single-request decode and prefill keep exactly the shapes they had.

The wide decode tiles keep each output element's K order (same BLOCK_K steps, same MMA), so a
row's result does not depend on how many other rows share the call. tools/test_decode_rows.py
checks that against the 16-row tile rather than assuming it.
"""

from contextlib import contextmanager

DEFAULT = 16
WIDE = 64
LIMIT = DEFAULT


def limit() -> int:
    return LIMIT


def block_m(rows: int) -> int:
    """The decode row tile for `rows` <= WIDE: 16 exactly as before, else 32 or 64."""
    if rows <= 16:
        return 16
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    raise ValueError(f"{rows} rows exceeds the {WIDE}-row decode tile")


@contextmanager
def wide(rows: int = WIDE):
    """Treat calls of up to `rows` rows as decode-sized for the duration."""
    global LIMIT
    if not DEFAULT <= rows <= WIDE:
        raise ValueError(f"decode row limit must be in {DEFAULT}..{WIDE}")
    previous = LIMIT
    LIMIT = rows
    try:
        yield
    finally:
        LIMIT = previous
