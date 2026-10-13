import re
H = open("/usr/lib/jvm/default/include/jni.h").read()
slot_re = re.compile(r"^\s*(.*?)\(JNICALL \*([A-Za-z0-9_]+)\)", re.M)
res_re  = re.compile(r"^\s*void \*(reserved\d+);", re.M)

def parse(struct):
    body = re.search(r"struct %s \{(.*?)\n\};" % struct, H, re.S).group(1)
    slots = []
    for line in body.splitlines():
        if res_re.match(line):
            slots.append((None, "void*"))
        else:
            sm = slot_re.match(line)
            if sm:
                slots.append((sm.group(2), sm.group(1).strip()))
    return slots

def emit(var, slots):
    out = [f"{var}: list[tuple[int, str | None, str]] = ["]
    for i, (n, r) in enumerate(slots):
        name = "None" if n is None else f'"{n}"'
        out.append(f"    ({i}, {name}, {r!r}),")
    out.append("]")
    return "\n".join(out)

env = parse("JNINativeInterface_")
vm = parse("JNIInvokeInterface_")

print('"""Canonical JNI function tables, derived from the system <jni.h>.')
print()
print("The slot *order* is ABI: a native library invokes a JNI function as")
print("`(*env)->Fn(env, ...)`, i.e. by its pointer index into the interface struct,")
print("so this ordering is as load-bearing as the opcode table — the emulator places")
print("each Python-serviced stub at exactly its slot. The first four JNIEnv slots and")
print("first three JavaVM slots are reserved (NULL) in the real JVM.")
print()
print("Generated from jni.h by tools/gen_jni_api.py; a drift test")
print("(tests/test_jni_api.py) re-parses the system header and checks this matches.")
print('Do not hand-edit — regenerate instead."""')
print()
print("from __future__ import annotations")
print()
print("# We target 64-bit triples first (plan.md Phase 2: prefer linux-x86_64).")
print("POINTER_SIZE = 8")
print()
print("# (slot_index, function_name_or_None_if_reserved, c_return_type)")
print(emit("JNIENV_SLOTS", env))
print()
print(emit("JAVAVM_SLOTS", vm))
print()
print("JNIENV_INDEX = {name: idx for idx, name, _ in JNIENV_SLOTS if name}")
print("JAVAVM_INDEX = {name: idx for idx, name, _ in JAVAVM_SLOTS if name}")
print()
print("# x86-64 System V return-register classes: integers and pointers come back in")
print("# RAX, jfloat/jdouble in XMM0, void not at all. A Python-serviced stub uses this")
print("# to place its synthesised return value.")
print('_FLOAT_RETURNS = frozenset({"jfloat", "jdouble"})')
print()
print()
print("def return_category(c_type: str) -> str:")
print('    """-> "void" | "float" | "int" (int covers integers and pointers)."""')
print("    t = c_type.strip()")
print('    if t == "void":')
print('        return "void"')
print("    if t in _FLOAT_RETURNS:")
print('        return "float"')
print('    return "int"')
