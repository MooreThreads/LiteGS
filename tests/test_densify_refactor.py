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
            {'params': [xyz], 'name': 'xyz'},
            {'params': [opacity], 'name': 'opacity'},
            {'params': [scale], 'name': 'scale'},
            {'params': [rot], 'name': 'rot'},
        ], lr=1e-3)
        sh_optimizer = torch.optim.Adam([
            {'params': [sh_0], 'name': 'sh_0'},
            {'params': [sh_rest], 'name': 'sh_rest'},
        ], lr=1e-3)

        for param in [xyz, opacity, scale, rot, sh_0, sh_rest]:
            param.grad = torch.randn_like(param)
        optimizer.step()
        sh_optimizer.step()
        return optimizer, sh_optimizer

    def test_optimizer_adapter_preserves_state_without_cluster(self):
        optimizer, sh_optimizer = self._build_optimizers(clustered=False)

        append_params = GaussianParams(
            xyz=torch.randn(3, 2),
            scale=torch.randn(3, 2),
            rot=torch.randn(4, 2),
            sh_0=torch.randn(1, 3, 2),
            sh_rest=torch.randn(15, 3, 2),
            opacity=torch.randn(1, 2),
        )

        optimizer_adapter.append_([optimizer, sh_optimizer], append_params, False)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        self.assertEqual(params['xyz'].shape[-1], 6)
        self.assertEqual(optimizer.state[params['xyz']]['exp_avg'].shape, params['xyz'].shape)

        keep_mask = torch.tensor([True, False, True, True, False, True])
        optimizer_adapter.prune_([optimizer, sh_optimizer], keep_mask, False)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        self.assertEqual(params['xyz'].shape[-1], 4)
        self.assertEqual(optimizer.state[params['xyz']]['exp_avg_sq'].shape, params['xyz'].shape)

        old_step = optimizer.state[params['opacity']]['step'].clone()
        optimizer_adapter.replace_([optimizer, sh_optimizer], 'opacity', torch.randn(1, 4), False)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        self.assertTrue(torch.equal(optimizer.state[params['opacity']]['step'], old_step))
        self.assertEqual(torch.count_nonzero(optimizer.state[params['opacity']]['exp_avg']).item(), 0)
        self.assertEqual(torch.count_nonzero(optimizer.state[params['opacity']]['exp_avg_sq']).item(), 0)

    def test_optimizer_adapter_preserves_state_with_cluster_alignment(self):
        optimizer, sh_optimizer = self._build_optimizers(clustered=True)

        append_params = GaussianParams(
            xyz=torch.randn(3, 3),
            scale=torch.randn(3, 3),
            rot=torch.randn(4, 3),
            sh_0=torch.randn(1, 3, 3),
            sh_rest=torch.randn(15, 3, 3),
            opacity=torch.randn(1, 3),
        )

        optimizer_adapter.append_([optimizer, sh_optimizer], append_params, True)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        unclustered_xyz, = cluster.uncluster(params['xyz'])
        self.assertEqual(unclustered_xyz.shape[-1], 6)
        self.assertEqual(optimizer.state[params['xyz']]['exp_avg'].shape, params['xyz'].shape)

        keep_mask = torch.tensor([True, False, True, False, True, True])
        optimizer_adapter.prune_([optimizer, sh_optimizer], keep_mask, True)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        unclustered_xyz, = cluster.uncluster(params['xyz'])
        self.assertEqual(unclustered_xyz.shape[-1], 4)
        self.assertEqual(optimizer.state[params['xyz']]['exp_avg_sq'].shape, params['xyz'].shape)

        optimizer_adapter.replace_([optimizer, sh_optimizer], 'opacity', torch.randn(1, 4), True)
        params = optimizer_adapter.collect_named_parameters([optimizer, sh_optimizer])
        unclustered_opacity, = cluster.uncluster(params['opacity'])
        self.assertEqual(unclustered_opacity.shape[-1], 4)


class ModelDensifyStepTests(unittest.TestCase):
    def _fake_model(self):
        model = GaussianSplattingModel.__new__(GaussianSplattingModel)
        torch.nn.Module.__init__(model)
        model.optimizer = object()
        model.sh_optimizer = object()
        model.sh_degree = 3
        model.active_sh_degree = 0
        model.xyz = torch.nn.Parameter(torch.zeros(1, 8))
        model.scale = torch.nn.Parameter(torch.zeros(1, 8))
        model.rot = torch.nn.Parameter(torch.zeros(1, 8))
        model.sh_0 = torch.nn.Parameter(torch.zeros(1, 8))
        model.sh_rest = torch.nn.Parameter(torch.zeros(1, 8))
        model.opacity = torch.nn.Parameter(torch.zeros(1, 8))
        model.density_controller = mock.Mock()
        model.density_controller.densify_params = DensifyParams(interval=5, start=0, end=10, opacity_reset_interval=5)
        model.density_controller.is_densify_actived = lambda epoch: True
        return model

    def test_densify_step_syncs_optimizers_and_resets_stats(self):
        model = self._fake_model()
        edits = DensifyEdits(changed=True, stats_need_reset=True)
        model.density_controller.step.return_value = edits
        model.get_gaussian_params = mock.Mock(return_value=GaussianParams(
            model.xyz, model.scale, model.rot, model.sh_0, model.sh_rest, model.opacity
        ))
        model.sync_optimizers_after_densify = mock.Mock()
        model.apply_densify_edits = mock.Mock()
        model.spatial_rearrange = mock.Mock()

        with mock.patch('litegs.scene.model.StatisticsHelperInst.reset') as reset_mock, \
             mock.patch('litegs.scene.model.torch.cuda.empty_cache') as empty_cache_mock:
            model.densify_step(5)

        model.sync_optimizers_after_densify.assert_called_once_with(edits)
        model.apply_densify_edits.assert_not_called()
        model.spatial_rearrange.assert_called_once()
        reset_mock.assert_called_once()
        empty_cache_mock.assert_called_once()
        self.assertEqual(model.active_sh_degree, 1)

    def test_densify_step_uses_direct_apply_without_optimizers(self):
        model = self._fake_model()
        model.optimizer = None
        model.sh_optimizer = None
        edits = DensifyEdits(changed=True, stats_need_reset=False)
        model.density_controller.step.return_value = edits
        model.get_gaussian_params = mock.Mock(return_value=GaussianParams(
            model.xyz, model.scale, model.rot, model.sh_0, model.sh_rest, model.opacity
        ))
        model.sync_optimizers_after_densify = mock.Mock()
        model.apply_densify_edits = mock.Mock()
        model.spatial_rearrange = mock.Mock()

        with mock.patch('litegs.scene.model.StatisticsHelperInst.reset') as reset_mock, \
             mock.patch('litegs.scene.model.torch.cuda.empty_cache') as empty_cache_mock:
            model.densify_step(5)

        model.apply_densify_edits.assert_called_once_with(edits)
        model.sync_optimizers_after_densify.assert_not_called()
        reset_mock.assert_not_called()
        empty_cache_mock.assert_not_called()

    def test_densify_step_skips_work_when_no_changes(self):
        model = self._fake_model()
        edits = DensifyEdits(changed=False, stats_need_reset=False)
        model.density_controller.step.return_value = edits
        model.get_gaussian_params = mock.Mock(return_value=GaussianParams(
            model.xyz, model.scale, model.rot, model.sh_0, model.sh_rest, model.opacity
        ))
        model.sync_optimizers_after_densify = mock.Mock()
        model.apply_densify_edits = mock.Mock()
        model.spatial_rearrange = mock.Mock()

        model.densify_step(4)

        model.sync_optimizers_after_densify.assert_not_called()
        model.apply_densify_edits.assert_not_called()
        model.spatial_rearrange.assert_not_called()


if __name__ == '__main__':
    unittest.main()

