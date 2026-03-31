import unittest
from unittest import mock

import torch

from litegs.arguments import DensifyParams
from litegs.scene import cluster
from litegs.scene.model import GaussianSplattingModel
from litegs.training import optimizer_adapter
from litegs.training.densify import DensifyEdits, GaussianParams


class OptimizerAdapterTests(unittest.TestCase):
    def _build_optimizers(self, clustered: bool, point_count: int = 4, chunk_size: int = 2):
        def maybe_cluster(tensor: torch.Tensor) -> torch.Tensor:
            if not clustered:
                return tensor
            clustered_tensor, = cluster.cluster_points(chunk_size, tensor)
            return clustered_tensor

        xyz = torch.nn.Parameter(maybe_cluster(torch.randn(3, point_count)))
        opacity = torch.nn.Parameter(maybe_cluster(torch.randn(1, point_count)))
        scale = torch.nn.Parameter(maybe_cluster(torch.randn(3, point_count)))
        rot = torch.nn.Parameter(maybe_cluster(torch.randn(4, point_count)))
        sh_0 = torch.nn.Parameter(maybe_cluster(torch.randn(1, 3, point_count)))
        sh_rest = torch.nn.Parameter(maybe_cluster(torch.randn(15, 3, point_count)))

        optimizer = torch.optim.Adam([
            {"params": [xyz], "name": "xyz"},
            {"params": [opacity], "name": "opacity"},
            {"params": [scale], "name": "scale"},
            {"params": [rot], "name": "rot"},
        ], lr=1e-3)
        sh_optimizer = torch.optim.Adam([
            {"params": [sh_0], "name": "sh_0"},
            {"params": [sh_rest], "name": "sh_rest"},
        ], lr=1e-3)

        for param in [xyz, opacity, scale, rot, sh_0, sh_rest]:
            param.grad = torch.randn_like(param)
        optimizer.step()
        sh_optimizer.step()
        return optimizer, sh_optimizer

    def _named_parameters(self, optimizer, sh_optimizer):
        named = {}
        for group in optimizer.param_groups + sh_optimizer.param_groups:
            named[group["name"]] = group["params"][0]
        return named

    def test_sync_state_after_densify_rebinds_and_preserves_state(self):
        optimizer, sh_optimizer = self._build_optimizers(clustered=False)
        params = self._named_parameters(optimizer, sh_optimizer)

        append_params = GaussianParams(
            xyz=torch.randn(3, 2),
            scale=torch.randn(3, 2),
            rot=torch.randn(4, 2),
            sh_0=torch.randn(1, 3, 2),
            sh_rest=torch.randn(15, 3, 2),
            opacity=torch.randn(1, 2),
        )
        edits = DensifyEdits(
            changed=True,
            prune_index=torch.tensor([1]),
            append_params=append_params,
        )

        exchange_dict = {}
        for name, param in params.items():
            kept_tensor = torch.cat((param.detach()[..., :1], param.detach()[..., 2:]), dim=-1)
            append_tensor = getattr(append_params, name)
            exchange_dict[param] = torch.nn.Parameter(torch.cat((kept_tensor, append_tensor), dim=-1))

        old_xyz = params["xyz"]
        old_state = optimizer.state[old_xyz]["exp_avg"].clone()
        optimizer_adapter.sync_state_after_densify(
            [optimizer, sh_optimizer],
            edits,
            exchange_dict,
            False,
        )

        new_xyz = optimizer.param_groups[0]["params"][0]
        self.assertIs(new_xyz, exchange_dict[old_xyz])
        self.assertNotIn(old_xyz, optimizer.state)
        self.assertEqual(optimizer.state[new_xyz]["exp_avg"].shape, new_xyz.shape)
        self.assertTrue(torch.equal(optimizer.state[new_xyz]["exp_avg"][..., :1], old_state[..., :1]))
        self.assertTrue(torch.equal(optimizer.state[new_xyz]["exp_avg"][..., 1:3], old_state[..., 2:]))
        self.assertEqual(torch.count_nonzero(optimizer.state[new_xyz]["exp_avg"][..., 3:]).item(), 0)

    def test_sync_state_after_densify_handles_clustered_tensors(self):
        optimizer, sh_optimizer = self._build_optimizers(clustered=True)
        params = self._named_parameters(optimizer, sh_optimizer)

        append_params = GaussianParams(
            xyz=torch.randn(3, 2),
            scale=torch.randn(3, 2),
            rot=torch.randn(4, 2),
            sh_0=torch.randn(1, 3, 2),
            sh_rest=torch.randn(15, 3, 2),
            opacity=torch.randn(1, 2),
        )
        edits = DensifyEdits(
            changed=True,
            prune_index=torch.tensor([1, 3]),
            append_params=append_params,
        )

        exchange_dict = {}
        for name, param in params.items():
            unclustered_tensor, = cluster.uncluster(param.detach())
            kept_tensor = unclustered_tensor[..., torch.tensor([0, 2])]
            clustered_tensor, = cluster.cluster_points(2, torch.cat((kept_tensor, getattr(append_params, name)), dim=-1))
            exchange_dict[param] = torch.nn.Parameter(clustered_tensor)

        old_xyz = params["xyz"]
        optimizer_adapter.sync_state_after_densify(
            [optimizer, sh_optimizer],
            edits,
            exchange_dict,
            True,
        )

        new_xyz = optimizer.param_groups[0]["params"][0]
        unclustered_xyz, = cluster.uncluster(new_xyz.detach())
        self.assertEqual(unclustered_xyz.shape[-1], 4)
        self.assertEqual(optimizer.state[new_xyz]["exp_avg"].shape, new_xyz.shape)
        self.assertNotIn(old_xyz, optimizer.state)

    def test_reorder_states_by_index_reorders_optimizer_state_only(self):
        optimizer, sh_optimizer = self._build_optimizers(clustered=False)
        params = self._named_parameters(optimizer, sh_optimizer)
        xyz = params["xyz"]
        original_param = xyz.detach().clone()
        original_state = optimizer.state[xyz]["exp_avg"].clone()
        indices = torch.tensor([3, 1, 0, 2])

        optimizer_adapter.reorder_states_by_index([optimizer, sh_optimizer], indices, False)

        self.assertTrue(torch.equal(xyz.detach(), original_param))
        self.assertTrue(torch.equal(optimizer.state[xyz]["exp_avg"], original_state[..., indices]))


class ModelTests(unittest.TestCase):
    def _fake_model(self):
        model = GaussianSplattingModel.__new__(GaussianSplattingModel)
        torch.nn.Module.__init__(model)
        model.optimizer = None
        model.sh_optimizer = None
        model.cluster_size = 0
        model.sh_degree = 3
        model.active_sh_degree = 0
        model.xyz = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        model.scale = torch.nn.Parameter(torch.tensor([[10.0, 11.0, 12.0, 13.0]]))
        model.rot = torch.nn.Parameter(torch.tensor([[20.0, 21.0, 22.0, 23.0]]))
        model.sh_0 = torch.nn.Parameter(torch.tensor([[30.0, 31.0, 32.0, 33.0]]))
        model.sh_rest = torch.nn.Parameter(torch.tensor([[40.0, 41.0, 42.0, 43.0]]))
        model.opacity = torch.nn.Parameter(torch.tensor([[50.0, 51.0, 52.0, 53.0]]))
        model.density_controller = mock.Mock()
        model.density_controller.densify_params = DensifyParams(interval=5, start=0, end=10, opacity_reset_interval=5)
        model.density_controller.is_densify_actived = lambda epoch: True
        model.update_cluster_aabb = mock.Mock()
        return model

    def test_densify_step_applies_edits_and_syncs_state(self):
        model = self._fake_model()
        model.optimizer = object()
        model.sh_optimizer = object()
        edits = DensifyEdits(changed=True)
        exchange_dict = {model.xyz: torch.nn.Parameter(model.xyz.detach().clone())}
        model.density_controller.step.return_value = edits
        model.apply_densify_edits = mock.Mock(return_value=exchange_dict)
        model.spatial_rearrange = mock.Mock()

        with mock.patch("litegs.scene.model.optimizer_adapter.sync_state_after_densify") as sync_mock, \
             mock.patch("litegs.scene.model.StatisticsHelperInst.reset") as reset_mock, \
             mock.patch("litegs.scene.model.torch.cuda.empty_cache") as empty_cache_mock:
            model.densify_step(5)

        model.apply_densify_edits.assert_called_once_with(edits)
        sync_mock.assert_called_once_with([model.optimizer, model.sh_optimizer], edits, exchange_dict, False)
        model.spatial_rearrange.assert_called_once()
        reset_mock.assert_called_once()
        empty_cache_mock.assert_called_once()
        self.assertEqual(model.active_sh_degree, 1)

    def test_spatial_rearrange_reorders_params_and_optimizer_state(self):
        model = self._fake_model()
        model.xyz.grad = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
        model.scale.grad = torch.tensor([[110.0, 111.0, 112.0, 113.0]])
        model.rot.grad = torch.tensor([[120.0, 121.0, 122.0, 123.0]])
        model.sh_0.grad = torch.tensor([[130.0, 131.0, 132.0, 133.0]])
        model.sh_rest.grad = torch.tensor([[140.0, 141.0, 142.0, 143.0]])
        model.opacity.grad = torch.tensor([[150.0, 151.0, 152.0, 153.0]])

        model.optimizer = torch.optim.Adam([
            {"params": [model.xyz], "name": "xyz"},
            {"params": [model.scale], "name": "scale"},
            {"params": [model.rot], "name": "rot"},
            {"params": [model.opacity], "name": "opacity"},
        ], lr=1e-3)
        model.sh_optimizer = torch.optim.Adam([
            {"params": [model.sh_0], "name": "sh_0"},
            {"params": [model.sh_rest], "name": "sh_rest"},
        ], lr=1e-3)

        for group in model.optimizer.param_groups + model.sh_optimizer.param_groups:
            group["params"][0].grad = torch.ones_like(group["params"][0])
        model.optimizer.step()
        model.sh_optimizer.step()
        old_xyz = model.xyz.detach().clone()
        old_state = model.optimizer.state[model.xyz]["exp_avg"].clone()

        indices = torch.tensor([2, 0, 3, 1])
        with mock.patch("litegs.scene.model.scene.spatial_refine", return_value=indices) as refine_mock:
            model.spatial_rearrange()

        refine_mock.assert_called_once_with(False, model.xyz)
        self.assertTrue(torch.equal(model.xyz.detach(), old_xyz[..., indices]))
        self.assertTrue(torch.equal(model.xyz.grad, torch.ones_like(model.xyz.grad)[..., indices]))
        self.assertTrue(torch.equal(model.optimizer.state[model.xyz]["exp_avg"], old_state[..., indices]))
        model.update_cluster_aabb.assert_called()


if __name__ == "__main__":
    unittest.main()
