"""ClassFile, ClassSource, and the method-descriptor parser.

Vendored from unseed §4/§5 (plus `parse_desc` from §6); see PROVENANCE.md. Copied,
not imported.

A `ClassFile` parses lazily where it can (CONSTANT_Utf8 entries are decoded on
first use and memoised) and raises `UnnativeError` on anything malformed, never
returning a plausible-looking partial parse.
"""

from __future__ import annotations

import os
import struct
import sys
import zipfile

from ..errors import UnnativeError
from .modified_utf8 import dec_modified_utf8
from .opcodes import decode

# Access flags (JVMS Table 4.1-B / 4.6-A). Only the ones the tool reasons about.
ACC_PUBLIC = 0x0001
ACC_PRIVATE = 0x0002
ACC_PROTECTED = 0x0004
ACC_STATIC = 0x0008
ACC_FINAL = 0x0010
ACC_SYNCHRONIZED = 0x0020
ACC_NATIVE = 0x0100
ACC_ABSTRACT = 0x0400
ACC_SYNTHETIC = 0x1000

STRING_TO_STRING = '(Ljava/lang/String;)Ljava/lang/String;'

_CP_NUMBER = {3: ('int', '>i', 4), 4: ('float', '>f', 4),
              5: ('long', '>q', 8), 6: ('double', '>d', 8)}
_CP_ONE_INDEX = {7: 'class', 8: 'string', 16: 'methodtype', 19: 'module', 20: 'package'}
_CP_TWO_INDEX = {9: 'fieldref', 10: 'methodref', 11: 'ifmethodref',
                 12: 'nameandtype', 17: 'dynamic', 18: 'invokedynamic'}
_CP_REF_KINDS = ('fieldref', 'methodref', 'ifmethodref')


def parse_desc(desc):
    """'(Ljava/lang/Object;JJ)V' -> (['Ljava/lang/Object;','J','J'], 'V')"""
    if not desc.startswith('('):
        raise UnnativeError(f'malformed method descriptor {desc!r}')
    i, args = 1, []
    while i < len(desc) and desc[i] != ')':
        j = i
        while desc[j] == '[':
            j += 1
            if j >= len(desc):
                raise UnnativeError(f'truncated array type in {desc!r}')
        j = desc.index(';', j) + 1 if desc[j] == 'L' else j + 1
        args.append(desc[i:j])
        i = j
    if i >= len(desc) or desc[i] != ')':
        raise UnnativeError(f'unterminated args in {desc!r}')
    return args, desc[i + 1:]


class ClassFile:
    """A parsed classfile.

    `cp[i]` is a (kind, a, b) tuple. `methods`/`fields` are dicts with keys
    access, name, desc, attrs (name -> raw bytes) and attr_offs (name -> absolute
    offset of the attribute body).
    """

    def __init__(self, name: str, data: bytes):
        self.name = name              # how it was addressed (zip entry / path)
        self.data = data
        if data[:4] != b'\xca\xfe\xba\xbe':
            raise UnnativeError(f'{name}: not a classfile (magic {data[:4].hex()})')
        try:
            self.minor, self.major = struct.unpack_from('>HH', data, 4)
            self._parse_cp()
            self._parse_body()
        except (struct.error, IndexError) as ex:
            raise UnnativeError(f'{name}: truncated or malformed classfile ({ex})') from ex

    # -- parsing --
    def _parse_cp(self):
        d = self.data
        count = struct.unpack_from('>H', d, 8)[0]
        self.cp_count = count
        self.cp = [None] * count
        i, p = 1, 10
        while i < count:
            tag = d[p]
            p += 1
            if tag == 1:
                ln = struct.unpack_from('>H', d, p)[0]
                self.cp[i] = ('utf8', d[p + 2:p + 2 + ln], None)   # decoded lazily
                p += 2 + ln
            elif tag in _CP_NUMBER:
                kind, fmt, size = _CP_NUMBER[tag]
                self.cp[i] = (kind, struct.unpack_from(fmt, d, p)[0], None)
                p += size
                if size == 8:
                    i += 1                  # long/double occupy two pool slots
            elif tag in _CP_ONE_INDEX:
                self.cp[i] = (_CP_ONE_INDEX[tag], struct.unpack_from('>H', d, p)[0], None)
                p += 2
            elif tag in _CP_TWO_INDEX:
                a, b = struct.unpack_from('>HH', d, p)
                self.cp[i] = (_CP_TWO_INDEX[tag], a, b)
                p += 4
            elif tag == 15:
                # CONSTANT_MethodHandle: reference_kind u1, reference_index u2.
                # `p` already points past the tag, so the kind is at d[p].
                self.cp[i] = ('methodhandle', d[p], struct.unpack_from('>H', d, p + 1)[0])
                p += 3
            else:
                raise UnnativeError(f'{self.name}: bad CP tag {tag} at index {i}')
            i += 1
        self.cp_end = p

    def _parse_body(self):
        d = self.data
        p = self.cp_end
        self.access, this_i, super_i, n_ifaces = struct.unpack_from('>HHHH', d, p)
        p += 8
        self.this = self.class_name(this_i)
        self.super = self.class_name(super_i) if super_i else None
        self.interfaces = [self.class_name(struct.unpack_from('>H', d, p + 2 * k)[0])
                           for k in range(n_ifaces)]
        p += 2 * n_ifaces
        self.fields, p = self._members(p)
        self.methods, p = self._members(p)
        self.attrs, self.class_attr_offs, p = self._attrs(p)
        self.dotted = self.this.replace('/', '.')
        if p != len(d):
            raise UnnativeError(f'{self.name}: {len(d) - p} trailing byte(s) after class attrs')

    def _members(self, p):
        cnt = struct.unpack_from('>H', self.data, p)[0]
        p += 2
        out = []
        for _ in range(cnt):
            acc, name_i, desc_i = struct.unpack_from('>HHH', self.data, p)
            p += 6
            attrs, offs, p = self._attrs(p)
            out.append({'access': acc, 'name': self.utf8(name_i),
                        'desc': self.utf8(desc_i), 'attrs': attrs, 'attr_offs': offs})
        return out, p

    def _attrs(self, p):
        d = self.data
        cnt = struct.unpack_from('>H', d, p)[0]
        p += 2
        res, offs = {}, {}
        for _ in range(cnt):
            name_i, ln = struct.unpack_from('>HI', d, p)
            p += 6
            aname = self.utf8(name_i)
            if aname in res:
                raise UnnativeError(f'{self.name}: duplicate attribute {aname}')
            res[aname] = d[p:p + ln]
            offs[aname] = p
            p += ln
        return res, offs, p

    # -- constant-pool accessors --
    def _e(self, i):
        if not (0 < i < self.cp_count):
            raise UnnativeError(f'{self.name}: CP index {i} out of range')
        e = self.cp[i]
        if e is None:
            raise UnnativeError(f'{self.name}: CP index {i} is a wide-slot continuation')
        return e

    def _expect(self, i, *kinds):
        """The pool entry at `i`, which must be one of `kinds`."""
        e = self._e(i)
        if e[0] not in kinds:
            raise UnnativeError(f'{self.name}: CP#{i} is {e[0]}, not {"/".join(kinds)}')
        return e

    def utf8(self, i):
        _k, raw, cached = self._expect(i, 'utf8')
        if cached is None:
            cached = dec_modified_utf8(raw)          # decoded on first use, then memoised
            self.cp[i] = ('utf8', raw, cached)
        return cached

    def class_name(self, i):
        return self.utf8(self._expect(i, 'class')[1])

    def string_const(self, i):
        return self.utf8(self._expect(i, 'string')[1])

    def nat(self, i):
        _k, a, b = self._expect(i, 'nameandtype')
        return self.utf8(a), self.utf8(b)

    def ref(self, i):
        """-> (owner_internal, name, desc, kind)"""
        k, a, b = self._expect(i, *_CP_REF_KINDS)
        nm, de = self.nat(b)
        return self.class_name(a), nm, de, k

    def methodhandle(self, i):
        """-> (reference_kind, owner_internal, name, desc)

        Reference kinds 1-4 point at a Fieldref, 5-9 at a (Interface)Methodref.
        An out-of-range kind raises: silently returning '?' is what let a
        constant-pool parsing bug masquerade as "every bootstrap is unknown".
        """
        _k, kind, ref = self._expect(i, 'methodhandle')
        if not 1 <= kind <= 9:
            raise UnnativeError(f'{self.name}: CP#{i} has invalid MethodHandle '
                                f'reference_kind {kind}')
        owner, nm, de, rk = self.ref(ref)
        want_field = kind <= 4
        if want_field != (rk == 'fieldref'):
            raise UnnativeError(f'{self.name}: CP#{i} kind {kind} needs a '
                                f'{"Fieldref" if want_field else "Methodref"}, got {rk}')
        return kind, owner, nm, de

    def indy(self, i):
        _k, a, b = self._expect(i, 'invokedynamic')
        nm, de = self.nat(b)
        return a, nm, de

    def const_value(self, i):
        """The value an `ldc`/`ldc_w`/`ldc2_w` of CP#i pushes, as (kind, value)."""
        k, v, _ = self._e(i)
        if k in ('int', 'long', 'float', 'double'):
            return k, v
        if k == 'string':
            return 'str', self.utf8(v)
        if k == 'class':
            return 'class', self.utf8(v)
        if k == 'methodtype':
            return 'mtype', self.utf8(v)
        if k == 'methodhandle':
            return ('mhandle',) + self.methodhandle(i)[1:]
        raise UnnativeError(f'{self.name}: CP#{i} ({k}) is not an ldc-able constant')

    # -- code and attributes --
    def code_of(self, m):
        """-> (bytecode_bytes, meta_dict) for a method, or (None, {})."""
        raw = m['attrs'].get('Code')
        if raw is None:
            return None, {}
        max_stack, max_locals, clen = struct.unpack_from('>HHI', raw, 0)
        body = raw[8:8 + clen]
        p = 8 + clen
        p += 2 + 8 * struct.unpack_from('>H', raw, p)[0]           # skip exception table
        n_attrs = struct.unpack_from('>H', raw, p)[0]
        p += 2
        sub = {}
        for _ in range(n_attrs):
            name_i, ln = struct.unpack_from('>HI', raw, p)
            sub[self.utf8(name_i)] = raw[p + 6:p + 6 + ln]
            p += 6 + ln
        return body, {'sub': sub, 'max_locals': max_locals, 'max_stack': max_stack,
                      'code_len': clen, 'code_abs_off': m['attr_offs']['Code'] + 8}

    def insns(self, m):
        """-> (decoded instructions or None, meta). None means no Code attribute."""
        body, meta = self.code_of(m)
        if body is None:
            return None, meta
        return decode(body), meta

    def bootstrap_methods(self):
        """-> [(method_handle_cp_index, [static_arg_cp_indexes])]"""
        raw = self.attrs.get('BootstrapMethods')
        if raw is None:
            return []
        p, out = 2, []
        for _ in range(struct.unpack_from('>H', raw, 0)[0]):
            mh, na = struct.unpack_from('>HH', raw, p)
            p += 4
            out.append((mh, [struct.unpack_from('>H', raw, p + 2 * k)[0] for k in range(na)]))
            p += 2 * na
        return out

    @property
    def native_methods(self):
        return {(m['name'], m['desc']) for m in self.methods if m['access'] & ACC_NATIVE}


def load_class(name, data):
    return ClassFile(name, data)


class ClassSource:
    """A jar/zip, or a directory tree of .class files, addressed uniformly."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.is_zip = os.path.isfile(self.path) and zipfile.is_zipfile(self.path)
        if self.is_zip:
            self.zf = zipfile.ZipFile(self.path)
            self.entries = [n for n in self.zf.namelist() if n.endswith('.class')]
        else:
            if not os.path.isdir(self.path):
                raise UnnativeError(f'{self.path}: not a jar/zip or a directory')
            self.entries = sorted(
                os.path.relpath(os.path.join(root, f), self.path)
                for root, _dirs, files in os.walk(self.path)
                for f in files if f.endswith('.class'))

    def read(self, entry):
        if self.is_zip:
            return self.zf.read(entry)
        with open(os.path.join(self.path, entry), 'rb') as f:
            return f.read()

    def names(self):
        """All entry names, not just .class — jars carry the native blob too."""
        if self.is_zip:
            return list(self.zf.namelist())
        return sorted(
            os.path.relpath(os.path.join(root, f), self.path)
            for root, _dirs, files in os.walk(self.path)
            for f in files)

    def read_name(self, name):
        """Read any entry by name (for non-.class resources)."""
        if self.is_zip:
            return self.zf.read(name)
        with open(os.path.join(self.path, name), 'rb') as f:
            return f.read()

    def entry_for(self, cfs, internal):
        """The entry name under which class `internal` was loaded into `cfs`."""
        e = internal + '.class'
        if e in cfs:
            return e
        for k, cf in cfs.items():
            if cf.this == internal:
                return k
        raise UnnativeError(f'no input entry for {internal}')

    def close(self):
        if self.is_zip:
            self.zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __len__(self):
        return len(self.entries)


def _report_parse_failures(fails):
    print(f'WARNING: {len(fails)} classfile(s) failed to parse:', file=sys.stderr)
    for e, why in fails[:10]:
        print(f'  {e}: {why}', file=sys.stderr)


def load_all(src: ClassSource, quiet=False):
    """-> (ordered dict entry->ClassFile, list of (entry, reason) failures)."""
    out, fails = {}, []
    for e in src.entries:
        try:
            out[e] = load_class(e, src.read(e))
        except UnnativeError as ex:
            fails.append((e, str(ex)))
        except Exception as ex:                      # noqa: BLE001 - report, don't guess
            fails.append((e, f'{type(ex).__name__}: {ex}'))
    if fails and not quiet:
        _report_parse_failures(fails)
    return out, fails
