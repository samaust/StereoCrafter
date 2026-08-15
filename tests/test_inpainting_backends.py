import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from stereocrafter.inference.inpainting import (
    available_inpainting_backends,
    base,
    create_inpainter,
    register_inpainting_backend,
)


class FakeInpainter:
    name = "fake"

    def __init__(self, **options):
        self.options = options

    def inpaint(self, frames_warped, frames_mask, **options):
        return torch.zeros_like(frames_warped[..., :3])


class BackendRegistryTests(unittest.TestCase):
    def setUp(self):
        base._REGISTERED_BACKENDS.clear()

    def tearDown(self):
        base._REGISTERED_BACKENDS.clear()

    def test_builtin_backends_are_available_without_eager_imports(self):
        sys.modules.pop("stereocrafter.inference.inpainting.wan_vace", None)
        sys.modules.pop("stereocrafter.inference.inpainting.svd", None)

        self.assertEqual(
            available_inpainting_backends()[:2],
            ("wan_vace", "svd"),
        )
        self.assertNotIn(
            "stereocrafter.inference.inpainting.wan_vace",
            sys.modules,
        )
        self.assertNotIn(
            "stereocrafter.inference.inpainting.svd",
            sys.modules,
        )

    def test_factory_dispatches_builtin_backends(self):
        for backend, module_name in (
            ("wan_vace", "stereocrafter.inference.inpainting.wan_vace"),
            ("svd", "stereocrafter.inference.inpainting.svd"),
        ):
            with self.subTest(backend=backend):
                imported = []

                def fake_import(name):
                    imported.append(name)
                    class_name = base._BUILTIN_BACKENDS[backend][1]
                    return SimpleNamespace(**{class_name: FakeInpainter})

                with patch.object(base, "import_module", fake_import):
                    inpainter = create_inpainter(backend, token="value")

                self.assertEqual(imported, [module_name])
                self.assertEqual(inpainter.options, {"token": "value"})

    def test_factory_defaults_to_wan_vace(self):
        imported = []

        def fake_import(name):
            imported.append(name)
            return SimpleNamespace(WanVaceInpainter=FakeInpainter)

        with patch.object(base, "import_module", fake_import):
            create_inpainter()

        self.assertEqual(
            imported,
            ["stereocrafter.inference.inpainting.wan_vace"],
        )

    def test_application_backend_can_be_registered_and_created(self):
        register_inpainting_backend("third_backend", FakeInpainter)

        inpainter = create_inpainter("third_backend", setting=42)

        self.assertEqual(inpainter.options, {"setting": 42})
        self.assertIn("third_backend", available_inpainting_backends())

    def test_duplicate_backend_requires_overwrite(self):
        register_inpainting_backend("third_backend", FakeInpainter)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_inpainting_backend("third_backend", FakeInpainter)

        register_inpainting_backend(
            "third_backend",
            FakeInpainter,
            overwrite=True,
        )

    def test_unknown_backend_lists_available_choices(self):
        with self.assertRaisesRegex(ValueError, "wan_vace, svd"):
            create_inpainter("missing")

    def test_invalid_backend_names_are_rejected(self):
        for name in ("WanVace", "wan-vace", "", "_"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Backend name"):
                    register_inpainting_backend(name, FakeInpainter)

    def test_factory_rejects_objects_without_backend_contract(self):
        register_inpainting_backend("invalid", lambda: object())

        with self.assertRaisesRegex(TypeError, "does not implement"):
            create_inpainter("invalid")


if __name__ == "__main__":
    unittest.main()
