import unittest

import torch

from litegs.utils import ops
from litegs.utils.ops import backend as backend_module


class OpsArchitectureTests(unittest.TestCase):
    def tearDown(self):
        backend_module.set_default_backend(backend_module.Backend.AUTO)

    def test_create_viewproj_matches_expected_formula_for_script_and_cuda(self):
        view_params = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0]],
            device="cuda",
        )
        proj_params = torch.tensor([2.0], device="cuda")

        expected_view = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [10.0, 20.0, 30.0, 1.0]]],
            device="cuda",
        )
        expected_proj = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.1, 1.0], [0.0, 0.0, -1.1, 0.0]]],
            device="cuda",
        )
        expected_viewproj = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.1, 1.0], [20.0, 80.0, 31.9, 30.0]]],
            device="cuda",
        )
        expected_frustum = torch.tensor(
            [[[2.0, 0.0, 1.0, 50.0], [-2.0, 0.0, 1.0, 10.0], [0.0, 4.0, 1.0, 110.0], [0.0, -4.0, 1.0, -50.0], [0.0, 0.0, 1.1, 31.9], [0.0, 0.0, -0.1, -1.9]]],
            device="cuda",
        )

        script_view, script_proj, script_viewproj, script_frustum = ops.create_viewproj_script(view_params, proj_params, 10, 20, 1.0, 11.0)
        cuda_view, cuda_proj, cuda_viewproj, cuda_frustum = ops.create_viewproj_cuda(view_params, proj_params, 10, 20, 1.0, 11.0)

        self.assertTrue(torch.allclose(script_view, expected_view, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_view, expected_view, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_proj, expected_proj, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_proj, expected_proj, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_viewproj, expected_viewproj, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_viewproj, expected_viewproj, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_frustum, expected_frustum, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_frustum, expected_frustum, atol=1e-6, rtol=1e-6))

    def test_create_viewproj_backward_matches_expected_for_script_and_cuda(self):
        expected_view_params_grad = torch.tensor(
            [[0.0, -3.8, -0.2, 4.0, 3.0, 5.0, 3.1]],
            device="cuda",
        )
        expected_proj_params_grad = torch.tensor([56.0], device="cuda")

        script_view_params = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0]],
            device="cuda",
            requires_grad=True,
        )
        script_proj_params = torch.tensor([2.0], device="cuda", requires_grad=True)
        script_view, script_proj, script_viewproj, _ = ops.create_viewproj_script(script_view_params, script_proj_params, 10, 20, 1.0, 11.0)
        (script_view.sum() + script_proj.sum() + script_viewproj.sum()).backward()

        cuda_view_params = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0]],
            device="cuda",
            requires_grad=True,
        )
        cuda_proj_params = torch.tensor([2.0], device="cuda", requires_grad=True)
        cuda_view, cuda_proj, cuda_viewproj, _ = ops.create_viewproj_cuda(cuda_view_params, cuda_proj_params, 10, 20, 1.0, 11.0)
        (cuda_view.sum() + cuda_proj.sum() + cuda_viewproj.sum()).backward()

        self.assertTrue(torch.allclose(script_view_params.grad, expected_view_params_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_view_params.grad, expected_view_params_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_proj_params.grad, expected_proj_params_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_proj_params.grad, expected_proj_params_grad, atol=1e-6, rtol=1e-6))
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

    def test_mvp_transform_matches_expected_formula_for_script_and_cuda(self):
        position = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]],
            device="cuda",
        )
        view_matrix = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [10.0, 20.0, 30.0, 1.0]]],
            device="cuda",
        )
        proj_matrix = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 8.0, 1.0], [0.0, 0.0, 0.0, 2.0]]],
            device="cuda",
        )

        expected_view = torch.tensor(
            [[[11.0, 12.0], [23.0, 24.0], [35.0, 36.0], [1.0, 1.0]]],
            device="cuda",
        )
        expected_ndc = torch.tensor(
            [[[0.5945946, 0.6315789], [2.4864864, 2.5263157], [7.5675673, 7.5789471], [1.0, 1.0]]],
            device="cuda",
        )

        script_view, script_ndc = ops.mvp_transform_script(position, view_matrix, proj_matrix)
        cuda_view, cuda_ndc = ops.mvp_transform_cuda(position, view_matrix, proj_matrix)

        self.assertTrue(torch.equal(script_view, expected_view))
        self.assertTrue(torch.allclose(script_ndc, expected_ndc, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(cuda_view, expected_view))
        self.assertTrue(torch.allclose(cuda_ndc, expected_ndc, atol=1e-6, rtol=1e-6))

    def test_mvp_transform_backward_matches_script_autograd_for_cuda(self):
        view_matrix = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [10.0, 20.0, 30.0, 1.0]]],
            device="cuda",
        )
        proj_matrix = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 8.0, 1.0], [0.0, 0.0, 0.0, 2.0]]],
            device="cuda",
        )

        script_position = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]],
            device="cuda",
            requires_grad=True,
        )
        script_view, script_ndc = ops.mvp_transform_script(script_position, view_matrix, proj_matrix)
        (script_view.sum() + script_ndc.sum()).backward()
        expected_grad = script_position.grad.detach().clone()

        cuda_position = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]],
            device="cuda",
            requires_grad=True,
        )
        cuda_view, cuda_ndc = ops.mvp_transform_cuda(cuda_position, view_matrix, proj_matrix)
        (cuda_view.sum() + cuda_ndc.sum()).backward()

        self.assertTrue(torch.allclose(cuda_position.grad, expected_grad, atol=1e-6, rtol=1e-6))

    def test_create_cov2d_directly_matches_expected_formula_for_script_and_cuda(self):
        J = torch.tensor(
            [[
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ]],
            device="cuda",
        )
        view_matrix = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device="cuda",
        )
        transform_matrix = torch.tensor(
            [[[2.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [4.0]]],
            device="cuda",
            requires_grad=True,
        )
        expected = torch.tensor(
            [[[[4.3], [0.0]], [[0.0], [9.3]]]],
            device="cuda",
        )

        script_output = ops.create_cov2d_directly_script(J, view_matrix, transform_matrix)
        cuda_output = ops.create_cov2d_directly_cuda(J, view_matrix, transform_matrix)

        self.assertTrue(torch.allclose(script_output, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_output, expected, atol=1e-6, rtol=1e-6))

    def test_create_cov2d_directly_backward_matches_expected_for_script_and_cuda(self):
        J = torch.tensor(
            [[
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ]],
            device="cuda",
        )
        view_matrix = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device="cuda",
        )
        expected_grad = torch.tensor(
            [[[4.0], [4.0], [0.0]], [[6.0], [6.0], [0.0]], [[0.0], [0.0], [0.0]]],
            device="cuda",
        )

        script_transform = torch.tensor(
            [[[2.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [4.0]]],
            device="cuda",
            requires_grad=True,
        )
        script_output = ops.create_cov2d_directly_script(J, view_matrix, script_transform)
        script_output.sum().backward()

        cuda_transform = torch.tensor(
            [[[2.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]], [[0.0], [0.0], [4.0]]],
            device="cuda",
            requires_grad=True,
        )
        cuda_output = ops.create_cov2d_directly_cuda(J, view_matrix, cuda_transform)
        cuda_output.sum().backward()

        self.assertTrue(torch.allclose(script_transform.grad, expected_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_transform.grad, expected_grad, atol=1e-6, rtol=1e-6))

    def test_spherical_harmonic_to_rgb_matches_expected_formula_for_script_and_cuda(self):
        sh_base = torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            device="cuda",
        )
        sh_rest = torch.tensor(
            [
                [[7.0], [8.0], [9.0]],
                [[10.0], [11.0], [12.0]],
                [[4.0], [5.0], [6.0]],
            ],
            device="cuda",
        )
        dirs = torch.tensor(
            [[[1.0], [0.0], [0.0]]],
            device="cuda",
        )
        expected = torch.tensor(
            [[[-1.1723152558], [-1.3788229760], [-1.5853306961]]],
            device="cuda",
        )

        script_output = ops.spherical_harmonic_to_rgb_script(1, sh_base, sh_rest, dirs)
        cuda_output = ops.spherical_harmonic_to_rgb_cuda(1, sh_base, sh_rest, dirs)

        self.assertTrue(torch.allclose(script_output, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_output, expected, atol=1e-6, rtol=1e-6))

    def test_spherical_harmonic_to_rgb_backward_matches_expected_sh_grads_for_script_and_cuda(self):
        expected_sh_base_grad = torch.tensor(
            [[[0.2820947918], [0.2820947918], [0.2820947918]]],
            device="cuda",
        )
        expected_sh_rest_grad = torch.tensor(
            [
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[-0.4886025119], [-0.4886025119], [-0.4886025119]],
            ],
            device="cuda",
        )

        script_sh_base = torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            device="cuda",
            requires_grad=True,
        )
        script_sh_rest = torch.tensor(
            [
                [[7.0], [8.0], [9.0]],
                [[10.0], [11.0], [12.0]],
                [[4.0], [5.0], [6.0]],
            ],
            device="cuda",
            requires_grad=True,
        )
        script_dirs = torch.tensor(
            [[[1.0], [0.0], [0.0]]],
            device="cuda",
            requires_grad=True,
        )
        script_output = ops.spherical_harmonic_to_rgb_script(1, script_sh_base, script_sh_rest, script_dirs)
        script_output.sum().backward()

        cuda_sh_base = torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            device="cuda",
            requires_grad=True,
        )
        cuda_sh_rest = torch.tensor(
            [
                [[7.0], [8.0], [9.0]],
                [[10.0], [11.0], [12.0]],
                [[4.0], [5.0], [6.0]],
            ],
            device="cuda",
            requires_grad=True,
        )
        cuda_dirs = torch.tensor(
            [[[1.0], [0.0], [0.0]]],
            device="cuda",
            requires_grad=True,
        )
        cuda_output = ops.spherical_harmonic_to_rgb_cuda(1, cuda_sh_base, cuda_sh_rest, cuda_dirs)
        cuda_output.sum().backward()

        self.assertTrue(torch.allclose(script_sh_base.grad, expected_sh_base_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_sh_rest.grad, expected_sh_rest_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_sh_base.grad, expected_sh_base_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_sh_rest.grad, expected_sh_rest_grad, atol=1e-6, rtol=1e-6))

    def test_eigh_and_inverse_2x2_matrix_matches_expected_formula_for_script_and_cuda(self):
        cov2d = torch.tensor(
            [[[[4.0], [1.0]], [[1.0], [9.0]]]],
            device="cuda",
        )
        expected_val = torch.tensor(
            [[[3.8074176], [9.1925821]]],
            device="cuda",
        )
        expected_vec = torch.tensor(
            [[[[0.9819564], [0.1891075]], [[-0.1891075], [0.9819564]]]],
            device="cuda",
        )
        expected_inv = torch.tensor(
            [[[[0.2571429], [-0.0285714]], [[-0.0285714], [0.1142857]]]],
            device="cuda",
        )

        script_val, script_vec, script_inv = ops.eigh_and_inverse_2x2_matrix_script(cov2d)
        cuda_val, cuda_vec, cuda_inv = ops.eigh_and_inverse_2x2_matrix_cuda(cov2d)

        self.assertTrue(torch.allclose(script_val, expected_val, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_val, expected_val, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_vec, expected_vec, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_vec, expected_vec, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(script_inv, expected_inv, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_inv, expected_inv, atol=1e-6, rtol=1e-6))

    def test_eigh_and_inverse_2x2_matrix_backward_matches_expected_for_script_and_cuda(self):
        expected_grad = torch.tensor(
            [[[[-0.0522449], [-0.0195918]], [[-0.0195918], [-0.0073469]]]],
            device="cuda",
        )

        script_cov2d = torch.tensor(
            [[[[4.0], [1.0]], [[1.0], [9.0]]]],
            device="cuda",
            requires_grad=True,
        )
        _, _, script_inv = ops.eigh_and_inverse_2x2_matrix_script(script_cov2d)
        script_inv.sum().backward()

        cuda_cov2d = torch.tensor(
            [[[[4.0], [1.0]], [[1.0], [9.0]]]],
            device="cuda",
            requires_grad=True,
        )
        _, _, cuda_inv = ops.eigh_and_inverse_2x2_matrix_cuda(cuda_cov2d)
        cuda_inv.sum().backward()

        self.assertTrue(torch.allclose(script_cov2d.grad, expected_grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(cuda_cov2d.grad, expected_grad, atol=1e-6, rtol=1e-6))
    
    def test_binning_cuda_matches_expected_single_point_layout(self):
        ndc = torch.tensor(
            [[[0.0], [0.0], [0.5], [1.0]]],
            device="cuda",
        )
        view_depth = torch.tensor([[1.0]], device="cuda")
        inv_cov2d = torch.tensor(
            [[[[10.0], [0.0]], [[0.0], [10.0]]]],
            device="cuda",
        )
        opacity = torch.tensor([[0.9]], device="cuda")

        tile_start_index, sorted_point_id, primitive_visible = ops.binning_cuda(
            ndc,
            view_depth,
            inv_cov2d,
            opacity,
            None,
            None,
            None,
            (16, 16),
            (8, 8),
        )

        self.assertTrue(torch.equal(tile_start_index, torch.tensor([[-1, 0, -1, -1, -1, 1]], device="cuda", dtype=torch.int32)))
        self.assertTrue(torch.equal(sorted_point_id, torch.tensor([[0]], device="cuda", dtype=torch.int32)))
        self.assertTrue(torch.equal(primitive_visible, torch.tensor([1], device="cuda")))

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



