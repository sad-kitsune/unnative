"""Dynamic-observation back-end (plan.md Phase 5a + Phase 2).

Layers, lowest to highest:

    jni_api.py    the canonical JNI function tables, derived from <jni.h>
    jvm_model.py  the Java-side class/method/field model, from the jar's classes
    loader_elf.py the native-library loader (ELF first; PE/Mach-O to follow)
    jnienv.py     the fake JNIEnv/JavaVM: vtable image + handle table + dispatch
    emu.py        the Unicorn CPU wrapper that runs the library against the above

Only `emu.py` needs Unicorn, and it imports it lazily, so everything else is usable
and testable with the standard library alone.
"""
