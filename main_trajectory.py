"""Main script for trajectory optimization."""

import io
import os
from pathlib import Path
import random
from typing import Tuple, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tap
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn import functional as F

from datasets_module.dataset_engine import RLBenchDataset, RLBenchReachMovingTargetDataset
from engine import BaseTrainTester
from diffuser_actor import (
    DiffuserActor, Act3D, ForesightDiffuserActorV3,
    ForesightDiffuserActorV4, ForesightDiffuserActorV5,
    ForesightDiffuserActorV6
)

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds, log_git_status
)
from utils import common_utils
from tqdm import tqdm


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    bool_is_pretrain: int = 0
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0
    trainer: str = "TrainTester"
    bool_rtn_attn: int = 0

    # Training and validation datasets
    dataset: Path
    valset: Path
    dense_interpolation: int = 0
    interpolation_length: int = 100
    num_future_frames_obs: int = 3

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    diffusion_timesteps: int = 100
    denoise_model: str = "ddpm"  # "ddpm" or "rectified_flow"
    num_inference_steps: int = 100  # inference steps, RF uses 10
    keypose_only: int = 0
    num_history: int = 0
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5
    num_traj_interval: int = 1
    model_name: str = 'diffuser_actor'

    # Model Act3D
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    num_ghost_points: int = 1000
    num_ghost_points_val: int = 10000
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    regress_position_offset: int = 0

    # Loss Act3D
    position_loss: str = "ce"  # one of "ce" (our model), "mse" (HiveFormer)
    ground_truth_gaussian_spread: float = 0.01
    compute_loss_at_all_layers: int = 0
    position_loss_coeff: float = 1.0
    position_offset_loss_coeff: float = 10000.0
    rotation_loss_coeff: float = 10.0
    symmetric_rotation_loss: int = 0
    gripper_loss_coeff: float = 1.0
    label_smoothing: float = 0.0
    regress_position_offset: int = 0

    # Model Foresight DP
    ## strength of classifier-free guidance in inference
    bool_classifier_free_guidance: bool = False
    classifier_free_guidance_w: float = 0.5
    classifier_free_guidance_dropout_prob: float = 0.1
    
    # Fine-tuning and dropout
    bool_finetune: int = 0  # Enable fine-tuning mode with different learning rates
    prob_dropout: float = 0.0  # Probability of dropping out velocity features during training
    bool_use_gating_and_adapter: int = 1  # Use gating and adapter in ForesightDiffuserActorV6


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = RLBenchDataset(
            root=self.args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length
        )
        test_dataset = RLBenchDataset(
            root=self.args.valset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffuserActor(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced)
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        )
        out = model(
            sample["trajectory"],
            sample["trajectory_mask"],
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            curr_gripper
        )

        # Backward pass
        loss = criterion.compute_loss(out)
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                curr_gripper.to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)

class TrainTesterRMT(TrainTester):
    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        if self.args.model_name == 'diffuser_actor':
            _model = DiffuserActor(
                backbone=self.args.backbone,
                image_size=tuple(int(x) for x in self.args.image_size.split(",")),
                embedding_dim=self.args.embedding_dim,
                num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
                use_instruction=bool(self.args.use_instruction),
                fps_subsampling_factor=self.args.fps_subsampling_factor,
                gripper_loc_bounds=self.args.gripper_loc_bounds,
                rotation_parametrization=self.args.rotation_parametrization,
                quaternion_format=self.args.quaternion_format,
                diffusion_timesteps=self.args.diffusion_timesteps,
                denoise_model=self.args.denoise_model,
                num_inference_steps=self.args.num_inference_steps,
                nhist=self.args.num_history,
                relative=bool(self.args.relative_action),
                lang_enhanced=bool(self.args.lang_enhanced)
            )
        elif self.args.model_name == 'foresight_diffuser_actor_v3':
            _model = ForesightDiffuserActorV3(
                backbone=self.args.backbone,
                image_size=tuple(int(x) for x in self.args.image_size.split(",")),
                embedding_dim=self.args.embedding_dim,
                num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
                use_instruction=bool(self.args.use_instruction),
                fps_subsampling_factor=self.args.fps_subsampling_factor,
                gripper_loc_bounds=self.args.gripper_loc_bounds,
                rotation_parametrization=self.args.rotation_parametrization,
                quaternion_format=self.args.quaternion_format,
                diffusion_timesteps=self.args.diffusion_timesteps,
                denoise_model=self.args.denoise_model,
                num_inference_steps=self.args.num_inference_steps,
                nhist=self.args.num_history,
                relative=bool(self.args.relative_action),
                lang_enhanced=bool(self.args.lang_enhanced),
                bool_rtn_attn=bool(self.args.bool_rtn_attn),
                bool_classifier_free_guidance=bool(self.args.bool_classifier_free_guidance),
                classifier_free_guidance_w=self.args.classifier_free_guidance_w,
                classifier_free_guidance_dropout_prob=self.args.classifier_free_guidance_dropout_prob,
            )
        elif self.args.model_name == 'foresight_diffuser_actor_v4':
            _model = ForesightDiffuserActorV4(
                backbone=self.args.backbone,
                image_size=tuple(int(x) for x in self.args.image_size.split(",")),
                embedding_dim=self.args.embedding_dim,
                num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
                use_instruction=bool(self.args.use_instruction),
                fps_subsampling_factor=self.args.fps_subsampling_factor,
                gripper_loc_bounds=self.args.gripper_loc_bounds,
                rotation_parametrization=self.args.rotation_parametrization,
                quaternion_format=self.args.quaternion_format,
                diffusion_timesteps=self.args.diffusion_timesteps,
                denoise_model=self.args.denoise_model,
                num_inference_steps=self.args.num_inference_steps,
                nhist=self.args.num_history,
                relative=bool(self.args.relative_action),
                lang_enhanced=bool(self.args.lang_enhanced),
                prob_dropout_vel_features=self.args.prob_dropout
            )
        elif self.args.model_name == 'foresight_diffuser_actor_v5':
            _model = ForesightDiffuserActorV5(
                backbone=self.args.backbone,
                image_size=tuple(int(x) for x in self.args.image_size.split(",")),
                embedding_dim=self.args.embedding_dim,
                num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
                use_instruction=bool(self.args.use_instruction),
                fps_subsampling_factor=self.args.fps_subsampling_factor,
                gripper_loc_bounds=self.args.gripper_loc_bounds,
                rotation_parametrization=self.args.rotation_parametrization,
                quaternion_format=self.args.quaternion_format,
                diffusion_timesteps=self.args.diffusion_timesteps,
                denoise_model=self.args.denoise_model,
                num_inference_steps=self.args.num_inference_steps,
                nhist=self.args.num_history,
                relative=bool(self.args.relative_action),
                lang_enhanced=bool(self.args.lang_enhanced),
                prob_dropout_vel_features=self.args.prob_dropout
            )
        elif self.args.model_name == 'foresight_diffuser_actor_v6':
            _model = ForesightDiffuserActorV6(
                backbone=self.args.backbone,
                image_size=tuple(int(x) for x in self.args.image_size.split(",")),
                embedding_dim=self.args.embedding_dim,
                num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
                use_instruction=bool(self.args.use_instruction),
                fps_subsampling_factor=self.args.fps_subsampling_factor,
                gripper_loc_bounds=self.args.gripper_loc_bounds,
                rotation_parametrization=self.args.rotation_parametrization,
                quaternion_format=self.args.quaternion_format,
                diffusion_timesteps=self.args.diffusion_timesteps,
                denoise_model=self.args.denoise_model,
                num_inference_steps=self.args.num_inference_steps,
                nhist=self.args.num_history,
                relative=bool(self.args.relative_action),
                lang_enhanced=bool(self.args.lang_enhanced),
                prob_dropout_vel_features=self.args.prob_dropout,
                bool_use_gating_and_adapter=self.args.bool_use_gating_and_adapter,
            )
        elif self.args.model_name == 'act3d':
            args = self.args
            _model = Act3D(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            use_instruction=bool(args.use_instruction)
        )
        elif self.args.model_name == 'smolvla':
            from diffuser_actor.trajectory_optimization.smolvla import build_smolvla_model
            _model = build_smolvla_model(self.args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        print("Model parameters:", count_parameters(_model))

        return _model

    def get_criterion(self):
        if self.args.model_name in [
            'diffuser_actor',
            'foresight_diffuser_actor_v3',
            'foresight_diffuser_actor_v4',
            'foresight_diffuser_actor_v5',
            'foresight_diffuser_actor_v6',
        ]:
            return TrajectoryCriterion()
        elif self.args.model_name == 'act3d':
            return LossAndMetrics(
            rotation_parametrization=args.rotation_parametrization,
            position_loss=args.position_loss,
            compute_loss_at_all_layers=bool(args.compute_loss_at_all_layers),
            ground_truth_gaussian_spread=args.ground_truth_gaussian_spread,
            label_smoothing=args.label_smoothing,
            position_loss_coeff=args.position_loss_coeff,
            position_offset_loss_coeff=args.position_offset_loss_coeff,
            rotation_loss_coeff=args.rotation_loss_coeff,
            gripper_loss_coeff=args.gripper_loss_coeff,
            symmetric_rotation_loss=bool(args.symmetric_rotation_loss)
        )
        elif self.args.model_name == 'smolvla':
            return TrajectoryCriterion()
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        instruction_str = load_instructions(
            str(self.args.instructions).replace('.pkl', '_str.pkl'),
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = RLBenchReachMovingTargetDataset(
            root=self.args.dataset,
            instructions=instruction,
            instructions_str=instruction_str,
            taskvar=taskvar,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            num_waypoints_traj=self.args.interpolation_length,
            num_frames_interval_traj=self.args.num_traj_interval,
            num_waypoints_history_traj=2,
            num_frames_interval_history_traj=10,
            num_future_frames_obs=self.args.num_future_frames_obs,
            num_history_frames_obs=0,
            bool_eval_only=False,
            bool_use_wm_predictions=self.args.model_name in [
                'foresight_diffuser_actor_v3',
                'foresight_diffuser_actor_v4',
                'foresight_diffuser_actor_v5',
                'foresight_diffuser_actor_v6',
            ],

        )
        test_dataset = RLBenchReachMovingTargetDataset(
            root=self.args.valset,
            instructions=instruction,
            instructions_str=instruction_str,
            taskvar=taskvar,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            num_iters=len(self.args.tasks)*50,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            num_waypoints_traj=self.args.interpolation_length,
            num_frames_interval_traj=self.args.num_traj_interval,
            num_waypoints_history_traj=2,
            num_frames_interval_history_traj=10,
            num_future_frames_obs=self.args.num_future_frames_obs,
            num_history_frames_obs=0,
            bool_eval_only=True,
            bool_use_wm_predictions=self.args.model_name in [
                'foresight_diffuser_actor_v3',
                'foresight_diffuser_actor_v4',
                'foresight_diffuser_actor_v5',
                'foresight_diffuser_actor_v6',
            ],
        )
        return train_dataset, test_dataset

    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / f"epoch_{step_id}.pth")
        return best_loss

    def load_checkpoint(self, model, optimizer):
        """Load from checkpoint."""
        if not self.args.bool_is_pretrain:
            return super().load_checkpoint(model, optimizer)
        else:
            print("=> loading checkpoint '{}'".format(self.args.checkpoint))
            model_dict = torch.load(self.args.checkpoint, map_location="cpu")
            # weight_dict = _modify_weight_dict(model_dict["weight"])
            weight_dict = model_dict["weight"]
            missing_keys, unexpected_keys = model.load_state_dict(weight_dict, strict=False)
            print(f"Missing keys: ")
            for itm in missing_keys:
                print(f"Missing {itm}")
            print(f"Unexpected keys: ")
            for itm in unexpected_keys:
                print(f"Unexpected {itm}")
            start_iter = 0
            best_loss = None

            print("=> loaded successfully '{}' (step {})".format(
                self.args.checkpoint, model_dict.get("iter", 0)
            ))
            del model_dict
            del weight_dict
            torch.cuda.empty_cache()
            return start_iter, best_loss

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
        device = next(model.parameters()).device
        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        )
        kwargs = {}

        if self.args.model_name == 'foresight_diffuser_actor_v3':
            kwargs = {
                "next_rgb_obs": sample["next_rgbs"].to(device),
                "next_pcd_obs": sample["next_pcds"].to(device),
                "next_mask_obs": sample["next_masks"].to(device),
                "next_frame_relative_id": sample["next_frame_relative_id"].to(device),
                "next_gripper": sample["next_gripper"].to(device),
            }
        elif self.args.model_name in [
            'foresight_diffuser_actor_v4',
            'foresight_diffuser_actor_v5',
            'foresight_diffuser_actor_v6',
        ]:
            kwargs = {
                "velmap_pc": torch.from_numpy(np.concatenate(sample["velmappc"], axis=0)).float().to(device),
            }
        logger = common_utils.Logger(self.args.log_dir, bool_enable=True)
        if self.args.model_name == 'diffuser_actor':
            out = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                curr_gripper
            )
            # Backward pass
            loss = criterion.compute_loss(out)
        elif self.args.model_name == 'foresight_diffuser_actor_v3':
            out = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                curr_gripper,
                **kwargs,
            )
            # Backward pass
            loss = criterion.compute_loss(out)
        elif self.args.model_name in [
            'foresight_diffuser_actor_v4',
            'foresight_diffuser_actor_v5',
            'foresight_diffuser_actor_v6',
        ]:
            out = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                curr_gripper,
                **kwargs,
            )
            # Backward pass
            loss = criterion.compute_loss(out)
        elif self.args.model_name == 'act3d':
            out = model(
                visible_rgb=sample["rgbs"],
                visible_pcd=sample["pcds"],
                instruction=sample["instr"],
                curr_gripper=sample['curr_gripper'][:, 0, :],
            )
            loss = criterion.compute_loss(out, sample)
            loss = sum(list(loss.values()))
        elif self.args.model_name == 'smolvla':
            from diffuser_actor.trajectory_optimization.smolvla import convert_batch_to_lerobot_format
            batch = convert_batch_to_lerobot_format(model, sample, curr_gripper)
            loss, output_dict = model.forward(batch)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        # logger.log_training_dataset_model_forward(
        #     input_dict={**sample, **kwargs, "curr_gripper": curr_gripper},
        #     output_dict=out,
        #     step_id=step_id,
        #     model=model,
        #     rank=dist.get_rank(),
        # )
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()
        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        if bool(self.args.bool_finetune) and self.args.model_name in [
            'foresight_diffuser_actor_v4',
            'foresight_diffuser_actor_v5',
            'foresight_diffuser_actor_v6',
            'diffuser_actor',
        ]:
            optimizer_grouped_parameters = [
                {"params": [], "weight_decay": 0.0, "lr": self.args.lr},  # New params with no_decay: lr
                {"params": [], "weight_decay": 0.0, "lr": self.args.lr * 0.1},  # Old params with no_decay: lr * 0.1
                {"params": [], "weight_decay": 5e-4, "lr": self.args.lr},  # New params with decay: lr
                {"params": [], "weight_decay": 5e-4, "lr": self.args.lr * 0.1}  # Old params with decay: lr * 0.1
            ]
            no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
            new_params_list = []  # Track new parameters for logging
            for name, param in model.named_parameters():
                is_new_param = "vel_" in name
                is_no_decay = any(nd in name for nd in no_decay)
                
                if is_new_param:
                    new_params_list.append(name)
                    if is_no_decay:
                        optimizer_grouped_parameters[0]["params"].append(param)  # New + no_decay: lr
                    else:
                        optimizer_grouped_parameters[2]["params"].append(param)  # New + decay: lr
                else:
                    if is_no_decay:
                        optimizer_grouped_parameters[1]["params"].append(param)  # Old + no_decay: lr * 0.1
                    else:
                        optimizer_grouped_parameters[3]["params"].append(param)  # Old + decay: lr * 0.1
            
            print(f"[Fine-tuning] New parameters (using lr={self.args.lr}): {len(new_params_list)} params")
            print(f"  New param names: {new_params_list}")
            optimizer = optim.AdamW(optimizer_grouped_parameters)
            return optimizer
        else:
            return super().get_optimizer(model)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()
        for i, sample in tqdm(enumerate(loader), total=len(loader) if bool(self.args.eval_only) else val_iters, desc="Evaluating"):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            kwargs = {}
            if self.args.model_name == 'foresight_diffuser_actor_v3':
                kwargs = {
                    "next_rgb_obs": sample["next_rgbs"].to(device),
                    "next_pcd_obs": sample["next_pcds"].to(device),
                    "next_mask_obs": sample["next_masks"].to(device),
                    "next_frame_relative_id": sample["next_frame_relative_id"].to(device),
                    "next_gripper": sample["next_gripper"].to(device),
                }
            elif self.args.model_name in [
                'foresight_diffuser_actor_v4',
                'foresight_diffuser_actor_v5',
                'foresight_diffuser_actor_v6',
            ]:
                kwargs = {
                    "velmap_pc": torch.from_numpy(np.concatenate(sample["velmappc"], axis=0)).float().to(device),
                }
            if self.args.model_name == 'diffuser_actor':
                output_dict = model(
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device),
                    sample["rgbs"].to(device),
                    sample["pcds"].to(device),
                    sample["instr"].to(device),
                    curr_gripper.to(device),
                    run_inference=True
                )
                action = output_dict['action']
                losses, losses_B = criterion.compute_metrics(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
            elif self.args.model_name == 'foresight_diffuser_actor_v3':
                output_dict = model(
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device),
                    sample["rgbs"].to(device),
                    sample["pcds"].to(device),
                    sample["instr"].to(device),
                    curr_gripper.to(device),
                    run_inference=True,
                    **kwargs,
                )
                action = output_dict['action']
                losses, losses_B = criterion.compute_metrics(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
            elif self.args.model_name in [
                'foresight_diffuser_actor_v4',
                'foresight_diffuser_actor_v5',
                'foresight_diffuser_actor_v6',
            ]:
                output_dict = model(
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device),
                    sample["rgbs"].to(device),
                    sample["pcds"].to(device),
                    sample["instr"].to(device),
                    curr_gripper.to(device),
                    run_inference=True,
                    **kwargs,
                )
                action = output_dict['action']
                losses, losses_B = criterion.compute_metrics(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
            elif self.args.model_name == 'act3d':
                action = model(
                    visible_rgb=sample["rgbs"],
                    visible_pcd=sample["pcds"],
                    instruction=sample["instr"],
                    curr_gripper=sample['curr_gripper'][:, 0, :],
                    # DO NOT provide ground-truth action to sample ghost points at validation time
                    gt_action=None
                )
                losses = criterion.compute_metrics(
                    action,
                    sample
                )
                losses_B = {}
                output_dict = {
                    'action': torch.cat([
                        action['position'],
                        action['rotation'],
                        action['gripper']],
                        dim=1)
                }
            elif self.args.model_name == 'smolvla':
                from diffuser_actor.trajectory_optimization.smolvla import convert_batch_to_lerobot_format
                batch = convert_batch_to_lerobot_format(model, sample, curr_gripper)
                action = model.module._get_action_chunk(batch, noise=None)
                output_dict = {
                    'action': action
                }
                losses, losses_B = criterion.compute_metrics(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
            else:
                raise NotImplementedError(f"Model {self.args.model_name} not implemented")

            if self.args.eval_only:
                logger = common_utils.Logger(self.args.log_dir, bool_enable=True)
                # logger.log_evaluation_dataset_model_forward(
                #     input_dict={**sample, **kwargs, "curr_gripper": curr_gripper},
                #     output_dict=output_dict,
                #     eval_step=i,
                #     rank=dist.get_rank(),
                # )
            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    output_dict['action'].to(device),
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            for key, val in values.items():
                self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)

def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    ret_dict["frame_ids"] = []
    for item in batch:
        ret_dict["task"] += item['task']
        ret_dict["frame_ids"] += item['frame_ids']
    return ret_dict

def rmt_traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "instr", "action"
    ]
    # if 'velmappc' in batch[0]:
    #     keys.extend(["velmappc"])
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    ret_dict["frame_id"] = []
    ret_dict["variation"] = []
    ret_dict["episode"] = []
    ret_dict["instr_str"] = []
    ret_dict["velmappc"] = []
    for item in batch:
        ret_dict["task"] += item['task']
        ret_dict["frame_id"] += item['frame_id']
        ret_dict["variation"] += item['variation']
        ret_dict["episode"] += item['episode']
        ret_dict["instr_str"] += item['instr_str']
        if 'velmappc' in item:
            ret_dict["velmappc"] += [item['velmappc']]
    return ret_dict


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]['action_mse']
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        # pred/gt are (B, L, 7), mask (B, L)
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        # gripper openess
        openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        tr = 'traj_'

        # Trajectory metrics
        ret_1, ret_2 = {
            tr + 'action_mse': F.mse_loss(pred, gt),
            tr + 'pos_l2': pos_l2.mean(),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
            tr + 'rot_l1': quat_l1.mean(),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
            tr + 'gripper': openess.flatten().float().mean()
        }, {
            tr + 'pos_l2': pos_l2.mean(-1),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
            tr + 'rot_l1': quat_l1.mean(-1),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
        }

        # Keypose metrics
        pos_l2 = ((pred[:, -1, :3] - gt[:, -1, :3]) ** 2).sum(-1).sqrt()
        quat_l1 = (pred[:, -1, 3:7] - gt[:, -1, 3:7]).abs().sum(-1)
        quat_l1_ = (pred[:, -1, 3:7] + gt[:, -1, 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        ret_1.update({
            'pos_l2_final': pos_l2.mean(),
            'pos_l2_final<0.01': (pos_l2 < 0.01).float().mean(),
            'rot_l1': quat_l1.mean(),
            'rot_l1<0025': (quat_l1 < 0.025).float().mean()
        })
        ret_2.update({
            'pos_l2_final': pos_l2,
            'pos_l2_final<0.01': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_l1<0.025': (quat_l1 < 0.025).float(),
        })

        return ret_1, ret_2

class LossAndMetrics:
    """
    Each method expects two dictionaries:
     - pred: {
        'position': (B, 3) gripper position,
        'rotation': (B, 4) gripper rotation,
        'gripper': (B, 1) whether gripper should open/close (0/1),
        'position_pyramid': list of 3 elements, (B, 1, 3) interm gripper pos,
        'visible_rgb_mask_pyramid': not used in loss,
        'ghost_pcd_masks_pyramid',
        'ghost_pcd_pyramid',
        'fine_ghost_pcd_offsets',
        'task'
     }
     - sample: {
        'frame_id',
        'task_id',
        'task',
        'variation',
        'rgbs',
        'pcds',
        'action': (B, 1, 8),
        'padding_mask': (B, 1),
        'instr',
        'gripper'
     }
    """
    def __init__(
        self,
        position_loss,
        rotation_parametrization,
        ground_truth_gaussian_spread,
        compute_loss_at_all_layers=False,
        label_smoothing=0.0,
        position_loss_coeff=1.0,
        position_offset_loss_coeff=10000.0,
        rotation_loss_coeff=10.0,
        gripper_loss_coeff=1.0,
        symmetric_rotation_loss=False,
    ):
        assert position_loss in ["mse", "ce", "ce+mse"]
        assert rotation_parametrization in [
            "quat_from_top_ghost", "quat_from_query",
            "6D_from_top_ghost", "6D_from_query"
        ]
        self.position_loss = position_loss
        self.rotation_parametrization = rotation_parametrization
        self.compute_loss_at_all_layers = compute_loss_at_all_layers
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.position_loss_coeff = position_loss_coeff
        self.position_offset_loss_coeff = position_offset_loss_coeff
        self.rotation_loss_coeff = rotation_loss_coeff
        self.gripper_loss_coeff = gripper_loss_coeff
        self.symmetric_rotation_loss = symmetric_rotation_loss

    def compute_loss(self, pred, sample):
        device = pred["position"].device
        # padding_mask = sample["padding_mask"].to(device)
        gt_action = sample["action"].to(device)  # [padding_mask]

        losses = {}

        self._compute_position_loss(pred, gt_action[:, :3], losses)

        self._compute_rotation_loss(pred, gt_action[:, 3:7], losses)

        losses["gripper"] = F.binary_cross_entropy(pred["gripper"], gt_action[:, 7:8])
        losses["gripper"] *= self.gripper_loss_coeff

        return losses

    def _compute_rotation_loss(self, pred, gt_quat, losses):
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                quat_loss = F.mse_loss(pred["rotation"], gt_quat, reduction='none').mean(1)
                quat_loss_ = F.mse_loss(pred["rotation"], gt_quat_, reduction='none').mean(1)
                select_mask = (quat_loss < quat_loss_).float()
                losses['rotation'] = (select_mask * quat_loss + (1 - select_mask) * quat_loss_).mean()
            else:
                losses["rotation"] = F.mse_loss(pred["rotation"], gt_quat)

        losses["rotation"] *= self.rotation_loss_coeff

    def _compute_position_loss(self, pred, gt_position, losses):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            losses["position_mse"] = F.mse_loss(pred["position"], gt_position) * self.position_loss_coeff

        elif self.position_loss in ["ce", "ce+mse"]:
            # Select a normalized Gaussian ball around the ground-truth
            # as a proxy label for a soft cross-entropy loss
            l2_pyramid = []
            label_pyramid = []
            for ghost_pcd_i in pred['ghost_pcd_pyramid']:
                l2_i = ((ghost_pcd_i - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                label_i = torch.softmax(-l2_i / self.ground_truth_gaussian_spread, dim=-1).detach()
                l2_pyramid.append(l2_i)
                label_pyramid.append(label_i)

            loss_layers = range(len(pred['ghost_pcd_masks_pyramid'][0])) if self.compute_loss_at_all_layers else [-1]

            for j in loss_layers:
                for i, ghost_pcd_masks_i in enumerate(pred["ghost_pcd_masks_pyramid"]):
                    losses[f"position_ce_level{i}"] = F.cross_entropy(
                        ghost_pcd_masks_i[j], label_pyramid[i],
                        label_smoothing=self.label_smoothing
                    ).mean() * self.position_loss_coeff / len(pred["ghost_pcd_masks_pyramid"])

            # Supervise offset from the ghost point's position to the predicted position
            num_sampling_level = len(pred['ghost_pcd_masks_pyramid'])
            if pred.get("fine_ghost_pcd_offsets") is not None:
                if pred["ghost_pcd_pyramid"][-1].shape[-1] != pred["ghost_pcd_pyramid"][0].shape[-1]:
                    npts = pred["ghost_pcd_pyramid"][-1].shape[-1] // num_sampling_level
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])[:, :, -npts:]
                else:
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])
                losses["position_offset"] = F.mse_loss(
                    pred_with_offset,
                    gt_position.unsqueeze(-1).repeat(1, 1, pred_with_offset.shape[-1])
                )
                losses["position_offset"] *= (self.position_offset_loss_coeff * self.position_loss_coeff)

            if self.position_loss == "ce":
                # Clear gradient on pred["position"] to avoid a memory leak since we don't
                # use it in the loss
                pred["position"] = pred["position"].detach()
            else:
                losses["position_mse"] = (
                    F.mse_loss(pred["position"], gt_position)
                    * self.position_loss_coeff
                )

    def compute_metrics(self, pred, sample):
        device = pred["position"].device
        dtype = pred["position"].dtype
        # padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)  # [padding_mask]

        metrics = {}

        tasks = np.array(sample["task"])

        final_pos_l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()
        metrics["mean/pos_l2_final"] = final_pos_l2.to(dtype).mean()
        metrics["mean/pos_l2_final<0.01"] = (final_pos_l2 < 0.01).to(dtype).mean()

        for i in range(len(pred["position_pyramid"])):
            pos_l2_i = ((pred["position_pyramid"][i].squeeze(1) - outputs[:, :3]) ** 2).sum(1).sqrt()
            metrics[f"mean/pos_l2_level{i}"] = pos_l2_i.to(dtype).mean()

        for task in np.unique(tasks):
            task_l2 = final_pos_l2[tasks == task]
            metrics[f"{task}/pos_l2_final"] = task_l2.to(dtype).mean()
            metrics[f"{task}/pos_l2_final<0.01"] = (task_l2 < 0.01).to(dtype).mean()

        # Gripper accuracy
        pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
        true_gripper = outputs[:, 7].bool()
        acc = pred_gripper == true_gripper
        metrics["gripper"] = acc.to(dtype).mean()

        # Rotation accuracy
        gt_quat = outputs[:, 3:7]
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                l1 = (pred["rotation"] - gt_quat).abs().sum(1)
                l1_ = (pred["rotation"] - gt_quat_).abs().sum(1)
                select_mask = (l1 < l1_).float()
                l1 = (select_mask * l1 + (1 - select_mask) * l1_)
            else:
                l1 = ((pred["rotation"] - gt_quat).abs().sum(1))

        metrics["mean/rot_l1"] = l1.to(dtype).mean()
        metrics["mean/rot_l1<0.05"] = (l1 < 0.05).to(dtype).mean()
        metrics["mean/rot_l1<0.025"] = (l1 < 0.025).to(dtype).mean()

        for task in np.unique(tasks):
            task_l1 = l1[tasks == task]
            metrics[f"{task}/rot_l1"] = task_l1.to(dtype).mean()
            metrics[f"{task}/rot_l1<0.05"] = (task_l1 < 0.05).to(dtype).mean()
            metrics[f"{task}/rot_l1<0.025"] = (task_l1 < 0.025).to(dtype).mean()

        return metrics


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    if len(pred.shape) == 1:
        pred = pred[None, :]
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
        color='red', label='pred'
    )
    ax.scatter3D(
        gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
        color='blue', label='gt'
    )

    center = gt[~mask].mean(0)
    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    os.system("git config --global --add safe.directory $PWD")
    log_git_status(log_dir / "git_status.txt")
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    if args.trainer == "TrainTester":
        train_tester = TrainTester(args)
        train_tester.main(collate_fn=traj_collate_fn)
    elif args.trainer == "TrainTesterRMT":
        train_tester = TrainTesterRMT(args)
        train_tester.main(collate_fn=rmt_traj_collate_fn)
    else:
        raise NotImplementedError()
