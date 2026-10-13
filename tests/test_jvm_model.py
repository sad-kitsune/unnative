"""Tests for the Java-side model used by the fake JNIEnv."""

from __future__ import annotations

import unittest

from _fixtures import INHERIT, NATIVE_WITH_LOADER, JavaFixtureCase

from unnative.native.jvm_model import JvmModel


class JvmModelTests(JavaFixtureCase):
    def test_find_and_native(self):
        jar = self.build_jar(NATIVE_WITH_LOADER)
        model = JvmModel.from_source(jar)
        self.assertIsNotNone(model.find_class("Demo"))
        self.assertIsNone(model.find_class("does/not/Exist"))
        natives = {m.name for m in model.native_methods()}
        self.assertEqual(natives, {"add"})
        add = model.resolve_method("Demo", "add", "(II)I")
        self.assertIsNotNone(add)
        self.assertTrue(add.static)

    def test_resolution_walks_superclass(self):
        jar = self.build_jar(INHERIT)
        model = JvmModel.from_source(jar)
        # getSides is declared on Shape; resolving it against Square must find it
        # and report Shape as the declaring owner.
        ref = model.resolve_method("Square", "getSides", "()I")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.owner, "Shape")
        self.assertFalse(ref.static)
        # A field declared on the superclass resolves the same way.
        fref = model.resolve_field("Square", "sides", "I")
        self.assertIsNotNone(fref)
        self.assertEqual(fref.owner, "Shape")
        # A method declared on the subclass resolves to the subclass.
        area = model.resolve_method("Square", "area", "()D")
        self.assertEqual(area.owner, "Square")
        # Native method is present on the subclass.
        self.assertIn("nativeArea", {m.name for m in model.native_methods()})

    def test_absent_member_is_none(self):
        jar = self.build_jar(INHERIT)
        model = JvmModel.from_source(jar)
        self.assertIsNone(model.resolve_method("Square", "nope", "()V"))


if __name__ == "__main__":
    unittest.main()
