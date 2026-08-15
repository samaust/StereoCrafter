import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stereocrafter.inference.inpainting.wan_vace import (
    FlowMatchScheduler,
    WanVaceInpainter,
    _load_quantized_transformer,
    _prepare_inpaint_inputs,
    _quantized_cache_path,
    _resolve_model_dtype,
    _validate_video_inputs,
    blend_h,
    blend_v,
    merge_tile_latents,
)


class WanVaceHelperTests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.zeros(3, 16, 24, 3)
        self.mask = torch.zeros(3, 16, 24, 1)

    def test_valid_video_inputs(self):
        _validate_video_inputs(
            self.frames,
            self.mask,
            frames_chunk=3,
            frames_overlap=1,
            tile_num=1,
            tile_overlap=0,
            inference_steps=2,
        )

    def test_layout_and_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "layout"):
            _validate_video_inputs(
                self.frames[0],
                self.mask,
                frames_chunk=3,
                frames_overlap=1,
                tile_num=1,
                tile_overlap=0,
                inference_steps=2,
            )
        with self.assertRaisesRegex(ValueError, "share T, H, and W"):
            _validate_video_inputs(
                self.frames,
                self.mask[:, :, :-1],
                frames_chunk=3,
                frames_overlap=1,
                tile_num=1,
                tile_overlap=0,
                inference_steps=2,
            )

    def test_inference_option_validation(self):
        invalid_options = (
            {"frames_chunk": 0},
            {"frames_overlap": 3},
            {"tile_num": 0},
            {"tile_overlap": -1},
            {"inference_steps": 0},
        )
        defaults = {
            "frames_chunk": 3,
            "frames_overlap": 1,
            "tile_num": 1,
            "tile_overlap": 0,
            "inference_steps": 2,
        }
        for changes in invalid_options:
            with self.subTest(changes=changes):
                options = defaults | changes
                with self.assertRaises(ValueError):
                    _validate_video_inputs(self.frames, self.mask, **options)

    def test_full_resolution_inputs_are_staged_on_cpu(self):
        frames, masks = _prepare_inpaint_inputs(self.frames, self.mask)

        self.assertEqual(frames.device.type, "cpu")
        self.assertEqual(masks.device.type, "cpu")
        self.assertEqual(frames.dtype, torch.float32)
        self.assertEqual(masks.dtype, torch.float32)
        self.assertEqual(frames.shape, (1, 3, 3, 16, 24))
        self.assertEqual(masks.shape, (1, 1, 3, 16, 24))

    def test_scheduler_reaches_final_sigma(self):
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=2, shift=1)
        sample = torch.ones(1)
        prediction = torch.ones(1)

        first = scheduler.step(prediction, scheduler.timesteps[0], sample)
        final = scheduler.step(prediction, scheduler.timesteps[1], first)

        self.assertTrue(torch.allclose(first, torch.tensor([0.5])))
        self.assertTrue(torch.allclose(final, torch.tensor([0.0])))

    def test_latent_blending_preserves_shape_and_dtype(self):
        horizontal_a = torch.zeros(1, 1, 1, 2, 4, dtype=torch.float16)
        horizontal_b = torch.ones_like(horizontal_a)
        vertical_a = torch.zeros(1, 1, 1, 4, 2, dtype=torch.float16)
        vertical_b = torch.ones_like(vertical_a)

        horizontal = blend_h(horizontal_a, horizontal_b, 2)
        vertical = blend_v(vertical_a, vertical_b, 2)

        self.assertEqual(horizontal.shape, horizontal_a.shape)
        self.assertEqual(vertical.shape, vertical_a.shape)
        self.assertEqual(horizontal.dtype, torch.float16)
        self.assertEqual(vertical.dtype, torch.float16)


class FakeQuantizedTransformer:
    loads = []
    saves = []

    def __init__(self, source, options):
        self.source = source
        self.options = options
        self.device = None

    @classmethod
    def from_pretrained(cls, source, **options):
        instance = cls(source, options)
        cls.loads.append(instance)
        return instance

    def save_pretrained(self, path, **options):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saves.append((Path(path), options))

    def to(self, device):
        self.device = torch.device(device)
        return self


class WanVaceQuantizationTests(unittest.TestCase):
    def setUp(self):
        FakeQuantizedTransformer.loads.clear()
        FakeQuantizedTransformer.saves.clear()

    def test_precision_dtype_defaults_and_validation(self):
        self.assertEqual(_resolve_model_dtype("fp8", None), torch.bfloat16)
        self.assertEqual(_resolve_model_dtype("fp16", None), torch.float16)
        self.assertEqual(_resolve_model_dtype("bf16", None), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "choose one of"):
            _resolve_model_dtype("missing", None)
        with self.assertRaisesRegex(ValueError, "requires dtype"):
            _resolve_model_dtype("fp16", torch.bfloat16)

    def test_quantized_transformer_is_built_once_then_reused(self):
        with tempfile.TemporaryDirectory() as cache_root:
            with patch.dict(
                _load_quantized_transformer.__globals__,
                {"_torchao_config": lambda _: "quant-config"},
            ):
                first = _load_quantized_transformer(
                    "model/source",
                    "fp8",
                    torch.bfloat16,
                    torch.device("cpu"),
                    cache_root=cache_root,
                    model_class=FakeQuantizedTransformer,
                )
                second = _load_quantized_transformer(
                    "model/source",
                    "fp8",
                    torch.bfloat16,
                    torch.device("cpu"),
                    cache_root=cache_root,
                    model_class=FakeQuantizedTransformer,
                )

            self.assertEqual(len(FakeQuantizedTransformer.saves), 1)
            self.assertEqual(len(FakeQuantizedTransformer.loads), 3)
            self.assertEqual(
                FakeQuantizedTransformer.loads[0].options["quantization_config"],
                "quant-config",
            )
            self.assertTrue(
                all(
                    load.options.get("use_safetensors") is False
                    for load in FakeQuantizedTransformer.loads[1:]
                )
            )
            self.assertEqual(first.device, torch.device("cpu"))
            self.assertEqual(second.device, torch.device("cpu"))

    def test_incompatible_cache_requires_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as cache_root:
            cache_path = _quantized_cache_path(
                "model/source", "int8", torch.bfloat16, cache_root
            )
            cache_path.mkdir(parents=True)
            (cache_path / "stereocrafter_quantization.json").write_text(
                json.dumps({"cache_version": -1}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "rebuild_quantized_cache"):
                _load_quantized_transformer(
                    "model/source",
                    "int8",
                    torch.bfloat16,
                    torch.device("cpu"),
                    cache_root=cache_root,
                    model_class=FakeQuantizedTransformer,
                )

            with patch.dict(
                _load_quantized_transformer.__globals__,
                {"_torchao_config": lambda _: "quant-config"},
            ):
                _load_quantized_transformer(
                    "model/source",
                    "int8",
                    torch.bfloat16,
                    torch.device("cpu"),
                    cache_root=cache_root,
                    rebuild=True,
                    model_class=FakeQuantizedTransformer,
                )
            self.assertEqual(len(FakeQuantizedTransformer.saves), 1)


class FakeComponent:
    def __init__(self, name, locations=None):
        self.name = name
        self.locations = locations if locations is not None else {}
        self.moves = []

    def to(self, device):
        device = torch.device(device)
        self.moves.append(device)
        self.locations[self.name] = device.type
        cuda_models = [
            name for name, location in self.locations.items() if location == "cuda"
        ]
        if "vae" in cuda_models and "transformer" in cuda_models:
            raise AssertionError("VAE and transformer became CUDA-resident together")
        return self


class WanVaceResidencyTests(unittest.TestCase):
    def test_sequential_phases_are_exclusive_and_transformer_moves_once(self):
        locations = {"vae": "cpu", "transformer": "cpu"}
        vae = FakeComponent("vae", locations)
        transformer = FakeComponent("transformer", locations)
        inpainter = WanVaceInpainter.__new__(WanVaceInpainter)
        inpainter.device = torch.device("cuda")
        inpainter.vae_device = torch.device("cuda")
        inpainter.sequential_offload = True
        cuda_mocks = {
            "reset_peak_memory_stats": lambda *_: None,
            "synchronize": lambda *_: None,
            "max_memory_allocated": lambda *_: 1024,
            "empty_cache": lambda: None,
        }
        with (
            patch.multiple(torch.cuda, **cuda_mocks),
            patch("stereocrafter.inference.inpainting.wan_vace.gc.collect"),
        ):
            started = inpainter._load_for_phase("encode", vae, inpainter.vae_device)
            inpainter._finish_phase("encode", vae, started)
            started = inpainter._load_for_phase(
                "denoise", transformer, inpainter.device
            )
            inpainter._finish_phase("denoise", transformer, started)
            started = inpainter._load_for_phase("decode", vae, inpainter.vae_device)
            inpainter._finish_phase("decode", vae, started)

        self.assertEqual(transformer.moves, [torch.device("cuda"), torch.device("cpu")])
        self.assertEqual(
            vae.moves,
            [
                torch.device("cuda"),
                torch.device("cpu"),
                torch.device("cuda"),
                torch.device("cpu"),
            ],
        )
        self.assertEqual(locations, {"vae": "cpu", "transformer": "cpu"})

    def test_resident_and_cpu_only_completion_do_not_offload(self):
        for device in ("cuda", "cpu"):
            with self.subTest(device=device):
                model = FakeComponent("transformer", {"transformer": device})
                inpainter = WanVaceInpainter.__new__(WanVaceInpainter)
                inpainter.device = torch.device(device)
                inpainter.vae_device = torch.device(device)
                inpainter.sequential_offload = False
                cuda_mocks = {
                    "synchronize": lambda *_: None,
                    "max_memory_allocated": lambda *_: 0,
                }
                with patch.multiple(torch.cuda, **cuda_mocks):
                    inpainter._finish_phase("resident", model, 0.0)
                self.assertEqual(model.moves, [])

    def test_cpu_merge_keeps_completed_latents_on_host(self):
        tiles = [torch.full((1, 1, 1, 2, 2), float(i)) for i in range(4)]
        merged = merge_tile_latents(
            tiles,
            tile_num=2,
            tile_overlap=0,
            tile_stride=(2, 2),
            vae_scale_factor_spatial=1,
        )
        self.assertEqual(merged.device.type, "cpu")
        self.assertEqual(merged.shape, (1, 1, 1, 4, 4))


if __name__ == "__main__":
    unittest.main()
