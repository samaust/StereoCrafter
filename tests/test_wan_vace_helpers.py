import unittest

import torch

from stereocrafter.inference.inpainting.wan_vace import (
    FlowMatchScheduler,
    _validate_video_inputs,
    blend_h,
    blend_v,
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


if __name__ == "__main__":
    unittest.main()
