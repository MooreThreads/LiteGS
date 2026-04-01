import unittest

import torch

from litegs.utils import ops
from litegs.utils.ops import backend as backend_module


class OpsArchitectureTests(unittest.TestCase):
    def tearDown(self):
        backend_module.set_default_backend(backend_module.Backend.AUTO)

    def test_create_transform_matrix_matches_expected_formula_for_script_and_cuda(self):
        scale = torch.tensor([[2.0], [3.0], [4.0]], device="cuda", requires_grad=True)
        rot = torch.tensor([[0.5], [0.5], [0.5], [0.5]], device="cuda", requires_grad=True)

        expected = torch.tensor(
            [
                [[0.0], [2.0], [0.0]],
                [[0.0], [0.0], [3.0]],
                [[4.0], [0.0], [0.0]],
            ],
            device="cuda",
        )

        script_output = ops.create_transform_matrix_script(scale, rot)
        cuda_output = ops.create_transform_matrix_cuda(scale, rot)

        self.assertTrue(torch.equal(script_output, expected))
        self.assertTrue(torch.equal(cuda_output, expected))

    def test_create_transform_matrix_backward_matches_expected_for_script_and_cuda(self):
        expected_scale_grad = torch.tensor([[1.0], [1.0], [1.0]], device="cuda")
        expected_rot_grad = torch.tensor([[0.0], [-4.0], [2.0], [2.0]], device="cuda")

        script_scale = torch.tensor([[2.0], [3.0], [4.0]], device="cuda", requires_grad=True)
        script_rot = torch.tensor([[0.5], [0.5], [0.5], [0.5]], device="cuda", requires_grad=True)
        script_output = ops.create_transform_matrix_script(script_scale, script_rot)
        script_output.sum().backward()

        cuda_scale = torch.tensor([[2.0], [3.0], [4.0]], device="cuda", requires_grad=True)
        cuda_rot = torch.tensor([[0.5], [0.5], [0.5], [0.5]], device="cuda", requires_grad=True)
        cuda_output = ops.create_transform_matrix_cuda(cuda_scale, cuda_rot)
        cuda_output.sum().backward()

        self.assertTrue(torch.equal(script_scale.grad, expected_scale_grad))
        self.assertTrue(torch.equal(script_rot.grad, expected_rot_grad))
        self.assertTrue(torch.equal(cuda_scale.grad, expected_scale_grad))
        self.assertTrue(torch.equal(cuda_rot.grad, expected_rot_grad))

    def test_create_rayspace_transform_matrix_matches_expected_formula_for_script_and_cuda(self):
        view_pos = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 10.0], [1.0, 1.0]]],
            device="cuda",
        )
        proj_matrix = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device="cuda",
        )
        output_shape = (10, 20)
        expected = torch.tensor(
            [[
                [[4.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [4.0, 2.0], [0.0, 0.0]],
                [[-0.8, -0.4], [-1.3, -0.65], [0.0, 0.0]],
            ]],
            device="cuda",
        )

        script_output = ops.create_rayspace_transform_matrix_script(view_pos, proj_matrix, output_shape)
        cuda_output = ops.create_rayspace_transform_matrix_cuda(view_pos, proj_matrix, output_shape)

        self.assertTrue(torch.allclose(script_output, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_output, expected, atol=1e-6, rtol=1e-6))

    def test_backend_priority_between_default_context_and_explicit_argument(self):
        backend_module.set_default_backend(backend_module.Backend.SCRIPT)
        self.assertEqual(
            backend_module.normalize_backend(None),
            backend_module.Backend.SCRIPT,
        )

        with ops.use_backend(ops.Backend.CUDA):
            self.assertEqual(
                backend_module.normalize_backend(None),
                backend_module.Backend.CUDA,
            )
            self.assertEqual(
                backend_module.normalize_backend("script"),
                backend_module.Backend.SCRIPT,
            )
            self.assertEqual(
                backend_module.normalize_backend(ops.Backend.AUTO),
                backend_module.Backend.AUTO,
            )

        self.assertEqual(
            backend_module.normalize_backend(None),
            backend_module.Backend.SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
