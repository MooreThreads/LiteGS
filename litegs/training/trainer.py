import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import os
import json


from .. import arguments
from .. import data
from .. import io_manager
from ..utils.statistic_helper import StatisticsHelperInst
from ..scene.model import GaussianSplattingModel
from ..render.pipeline import RenderPipeline
from ..arguments import TrainConfig,EvalConfig
from ..utils import ops


def __l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs((network_output - gt)).mean()


def _get_resume_primitive_num(state_dict: dict, cluster_size: int) -> int:
    checkpoint_xyz = state_dict.get("model.xyz", None)
    if checkpoint_xyz is None:
        raise KeyError("Checkpoint is missing 'model.xyz', cannot infer primitive count for resume.")

    if cluster_size > 0:
        if checkpoint_xyz.ndim != 3:
            raise ValueError(
                f"Expected clustered checkpoint tensor 'model.xyz' to have 3 dims, got shape {tuple(checkpoint_xyz.shape)}."
            )
        if checkpoint_xyz.shape[-1] != cluster_size:
            raise ValueError(
                f"Checkpoint cluster size {checkpoint_xyz.shape[-1]} does not match config cluster_size {cluster_size}."
            )
        primitive_num = int(checkpoint_xyz.shape[-2] * checkpoint_xyz.shape[-1])
    else:
        if checkpoint_xyz.ndim != 2:
            raise ValueError(
                f"Expected unclustered checkpoint tensor 'model.xyz' to have 2 dims, got shape {tuple(checkpoint_xyz.shape)}."
            )
        primitive_num = int(checkpoint_xyz.shape[-1])

    if primitive_num <= 0:
        raise ValueError("Checkpoint contains no primitives, cannot resume training.")
    return primitive_num


def _build_resume_init_tensors(state_dict: dict, cluster_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    primitive_num = _get_resume_primitive_num(state_dict, cluster_size)
    point_ids = torch.arange(primitive_num, device='cuda', dtype=torch.float32)
    init_xyz_tensor = torch.stack(
        (
            point_ids,
            torch.zeros_like(point_ids),
            torch.zeros_like(point_ids),
        ),
        dim=1,
    )
    init_color_tensor = torch.zeros((primitive_num, 3), device='cuda', dtype=torch.float32)
    return init_xyz_tensor, init_color_tensor


def _load_checkpoint(path: str) -> tuple[dict, int]:
    checkpoint = torch.load(path, map_location='cuda')
    if "pipeline" not in checkpoint:
        raise KeyError("Checkpoint is missing 'pipeline'.")
    if "start_epoch" not in checkpoint:
        raise KeyError("Checkpoint is missing 'start_epoch'.")

    start_epoch = checkpoint["start_epoch"]
    if isinstance(start_epoch, torch.Tensor):
        start_epoch = int(start_epoch.item())
    else:
        start_epoch = int(start_epoch)

    return checkpoint["pipeline"], start_epoch


def _save_checkpoint(path: str, pipeline: RenderPipeline, start_epoch: int) -> None:
    checkpoint = {
        "pipeline": pipeline.state_dict(),
        "start_epoch": int(start_epoch),
    }
    torch.save(checkpoint, path)
    return


def start(cfg:TrainConfig):

    cameras_info: dict[int, data.CameraInfo] = None
    camera_frames: list[data.ImageFrame] = None
    cameras_info, camera_frames, init_xyz, init_color = io_manager.load_colmap_result(cfg.dataset.source_path, cfg.dataset.images)

    # preload
    for camera_frame in camera_frames:
        camera_frame.load_image(cfg.dataset.resolution)

    # Dataset
    if cfg.dataset.eval:
        if os.path.exists(os.path.join(cfg.dataset.source_path, "train_test_split.json")):
            with open(os.path.join(cfg.dataset.source_path, "train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames = [c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames = [c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames = camera_frames
        test_frames = None

    trainingset = data.CameraFrameDataset(cameras_info, training_frames, cfg.dataset.resolution, cfg.dataset.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1, shuffle=True, pin_memory=not cfg.dataset.device_preload)
    test_loader = None
    if cfg.dataset.eval:
        testset = data.CameraFrameDataset(cameras_info, test_frames, cfg.dataset.resolution, cfg.dataset.device_preload)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=not cfg.dataset.device_preload)

    norm_trans, norm_radius = trainingset.get_norm()

    # Training configuration
    total_epoch = int(cfg.opt.iterations / len(trainingset))
    if cfg.densify.end < 0:
        cfg.densify.end = int(total_epoch * 0.8 / cfg.densify.opacity_reset_interval) * cfg.densify.opacity_reset_interval + 1

    checkpoint_state_dict = None
    start_epoch = 0
    if cfg.resume is not None:
        checkpoint_state_dict, start_epoch = _load_checkpoint(cfg.resume)

    # Create model
    if checkpoint_state_dict is None:
        init_xyz_tensor = torch.tensor(init_xyz, dtype=torch.float32, device='cuda')
        init_color_tensor = torch.tensor(init_color, dtype=torch.float32, device='cuda')
    else:
        init_xyz_tensor, init_color_tensor = _build_resume_init_tensors(checkpoint_state_dict, cfg.model.cluster_size)
    model = GaussianSplattingModel.from_arrays(
        init_xyz_tensor, init_color_tensor, cfg.model,
        norm_radius, cfg.opt, cfg.densify
    )
    pipeline = RenderPipeline(cfg.pipeline, model ,trainingset)
    model=None
    if checkpoint_state_dict is not None:
        pipeline.load_state_dict(checkpoint_state_dict)

    StatisticsHelperInst.reset(pipeline.model.xyz.shape[-2], pipeline.model.xyz.shape[-1], pipeline.model.density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    ops.set_default_backend(ops.Backend.CUDA)

    for epoch in range(start_epoch, total_epoch):
        pipeline.train()
        torch.cuda.synchronize()#sync for feedback buffer. Do not remove it!
        with StatisticsHelperInst.try_start(epoch):
            for view_matrix, proj_matrix, frustumplane, gt_image, idx_tensor in train_loader:
                view_matrix = view_matrix.cuda()
                proj_matrix = proj_matrix.cuda()
                frustumplane = frustumplane.cuda()
                gt_image = gt_image.cuda() / 255.0

                img, transmitance, _, _ = pipeline(view_matrix, proj_matrix, frustumplane, idx_tensor, gt_image.shape[2:])

                # Compute loss
                loss = fused_ssim.fused_l1_ssim_loss(img, gt_image)
                if cfg.pipeline.enable_transmitance:
                    loss += (1 - transmitance).abs().mean()

                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()

                # Optimizer step
                pipeline.model.optimizer.step()
                pipeline.model.optimizer.zero_grad(set_to_none=True)
                pipeline.model.sh_optimizer.step()
                pipeline.model.sh_optimizer.zero_grad(set_to_none=True)
                if pipeline.learnable_viewproj is not None:
                    pipeline.learnable_viewproj.step()
                pipeline.model.scheduler.step()

        # Evaluation
        pipeline.eval()
        if epoch in cfg.test_epochs:
            with torch.no_grad():
                loaders = {"Trainingset": train_loader}
                if cfg.dataset.eval:
                    loaders["Testset"] = test_loader
                for name, loader in loaders.items():
                    psnr_list = []
                    for view_matrix, proj_matrix, frustumplane, gt_image, idx_tensor in loader:
                        view_matrix = view_matrix.cuda()
                        proj_matrix = proj_matrix.cuda()
                        frustumplane = frustumplane.cuda()
                        gt_image = gt_image.cuda() / 255.0

                        img, transmitance, _, _ = pipeline(view_matrix, proj_matrix, frustumplane, idx_tensor, gt_image.shape[2:])

                        psnr_list.append(psnr.PeakSignalNoiseRatio(data_range=(0.0, 1.0)).cuda()(img, gt_image).unsqueeze(0))
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {}".format(epoch, name, torch.concat(psnr_list, dim=0).mean()))

        pipeline.model.densify_step(epoch)# Density control step
        progress_bar.update()

        # Save PLY
        if epoch in cfg.save_epochs or epoch == total_epoch - 1:
            if epoch == total_epoch - 1:
                torch.cuda.synchronize()
                progress_bar.close()
                print("{} takes: {}".format(cfg.dataset.model_path, progress_bar.format_dict['elapsed']))
                save_path = os.path.join(cfg.dataset.model_path, "finish")
            else:
                save_path = os.path.join(cfg.dataset.model_path, "iteration_{}".format(epoch))

            pipeline.model.save_ply(os.path.join(save_path, "point_cloud.ply"))
            if pipeline.learnable_viewproj is not None:
                torch.save(pipeline.learnable_viewproj.state_dict(),os.path.join(save_path, "viewproj.pt"))
            #save config
            eval_cfg = EvalConfig(dataset=cfg.dataset,pipeline=cfg.pipeline,model=cfg.model,save_image=False)
            eval_config_path = os.path.join(save_path, "eval_config.json")
            with open(eval_config_path, "w") as f:
                f.write(eval_cfg.to_json())

        #save checkpoint
        if epoch in cfg.cp_epochs:
            os.makedirs(cfg.dataset.model_path, exist_ok = True) 
            file_path=os.path.join(cfg.dataset.model_path,"chkpnt_{0}.pth".format(epoch))
            _save_checkpoint(file_path, pipeline, epoch + 1)
            config_path = os.path.join(cfg.dataset.model_path,"chkpnt_{0}_config.json".format(epoch))
            with open(config_path, "w") as f:
                f.write(cfg.to_json())
            
    return
