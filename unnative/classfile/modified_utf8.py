"""Modified UTF-8 (JVM constant pool encoding) and UTF-16 code units.

Vendored from unseed §2; see PROVENANCE.md. Copied, not imported.

Plain `bytes.decode('utf-8', 'replace')` is WRONG for CONSTANT_Utf8: the JVM's
modified UTF-8 encodes NUL as 0xC0 0x80 and supplementary characters as CESU-8
surrogate pairs, both of which 'replace' silently turns into U+FFFD. A protector's
encrypted class/member names are arbitrary char sequences, so they hit exactly
those cases — getting this wrong corrupts the very names that make lifting
possible.
"""

from __future__ import annotations

import struct

from ..errors import UnnativeError


def _combine_surrogates(units, what: str) -> str:
    """UTF-16 code units -> str, joining surrogate pairs.

    Raises on a lone surrogate instead of mangling it: silently substituting a
    replacement character is exactly the failure this tool exists to avoid.
    """
    res = []
    j = 0
    while j < len(units):
        u = units[j]
        if 0xD800 <= u <= 0xDBFF:
            if j + 1 < len(units) and 0xDC00 <= units[j + 1] <= 0xDFFF:
                res.append(chr(0x10000 + ((u - 0xD800) << 10) + (units[j + 1] - 0xDC00)))
                j += 2
                continue
            raise UnnativeError(f'lone high surrogate U+{u:04X} in {what}')
        if 0xDC00 <= u <= 0xDFFF:
            raise UnnativeError(f'lone low surrogate U+{u:04X} in {what}')
        res.append(chr(u))
        j += 1
    return ''.join(res)


def dec_modified_utf8(b: bytes) -> str:
    """Decode CONSTANT_Utf8 exactly as the JVM specifies. Raises on malformed input."""
    units = []
    i, n = 0, len(b)

    def is_cont(k: int) -> bool:
        return k < n and 0x80 <= b[k] < 0xC0

    while i < n:
        c = b[i]
        if c == 0:
            raise UnnativeError(f'raw 0x00 in CONSTANT_Utf8 at byte {i}')
        if c < 0x80:
            units.append(c)
            i += 1
        elif c < 0xC0:
            raise UnnativeError(f'stray continuation byte {c:#x} at {i}')
        elif c < 0xE0:
            if not is_cont(i + 1):
                raise UnnativeError(f'truncated 2-byte form at {i}')
            units.append(((c & 0x1F) << 6) | (b[i + 1] & 0x3F))
            i += 2
        elif c < 0xF0:
            if not (is_cont(i + 1) and is_cont(i + 2)):
                raise UnnativeError(f'truncated 3-byte form at {i}')
            units.append(((c & 0x0F) << 12) | ((b[i + 1] & 0x3F) << 6) | (b[i + 2] & 0x3F))
            i += 3
        else:
            raise UnnativeError(f'4-byte form {c:#x} is not valid modified UTF-8 at {i}')
    return _combine_surrogates(units, 'CONSTANT_Utf8')


def enc_modified_utf8(s: str) -> bytes:
    """Inverse of `dec_modified_utf8`."""
    out = bytearray()
    for ch in s:
        cp = ord(ch)
        if cp == 0:
            out += b'\xc0\x80'                       # NUL is 0xC0 0x80, never 0x00
        elif cp < 0x80:
            out.append(cp)
        elif cp < 0x800:
            out += bytes((0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            out += bytes((0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)))
        else:
            v = cp - 0x10000
            for u in (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)):
                out += bytes((0xE0 | (u >> 12), 0x80 | ((u >> 6) & 0x3F), 0x80 | (u & 0x3F)))
    return bytes(out)


def to_utf16_units(s: str) -> list:
    """Python str -> list of UTF-16 code units (what Java `charAt` indexes)."""
    raw = s.encode('utf-16-be')
    return list(struct.unpack('>%dH' % (len(raw) // 2), raw))


def from_utf16_units(units) -> str:
    """Code units -> Python str. Raises on a lone surrogate."""
    return _combine_surrogates(units, 'plaintext')
