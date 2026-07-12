"""Online evaluation script on RLBench."""
import random
from typing import Tuple, Optional
from pathlib import Path
import json
import os

import torch
import numpy as np
import tap

from diffuser_actor.keypose_optimization.act3d import Act3D
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v3 import ForesightDiffuserActorV3
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v4 import ForesightDiffuserActorV4
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v5 import ForesightDiffuserActorV5
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v6 import ForesightDiffuserActorV6
from utils.common_utils import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)
from utils.utils_with_rlbench import ExpertActioner, RLBenchEnv, Actioner, load_episodes
from online_evaluation_rlbench.policy_client import PolicyClient

class Arguments(tap.Tap):
    checkpoint: Path = ""
    seed: int = 2
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    collision_checking: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    gripper_loc_bounds_buffer: float = 0.04
    single_task_gripper_loc_bounds: int = 0
    predict_trajectory: int = 1
    vis_save_dir: str = None
    bool_future_info: int = 0
    bool_use_ground_truth_action: int = 0
    bool_expert_actioner: int = 0
    bool_classifier_free_guidance: int = 0
    classifier_free_guidance_w: float = 1.0
    bool_policy_server: int = 0  # 如果为 1，使用策略服务器客户端
    policy_server_port: int = 8765  # 策略服务器端口
    bool_world_server: int = 0  # 如果为 1，使用 WorldModel 服务器客户端模式
    world_server_port: int = 8766  # WorldModel 服务器端口

    # Act3D model parameters
    num_query_cross_attn_layers: int = 2
    num_ghost_point_cross_attn_layers: int = 2
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    regress_position_offset: int = 0

    # 3D Diffuser Actor model parameters
    diffusion_timesteps: int = 100
    num_history: int = 3
    num_future_frames: int = 10
    fps_subsampling_factor: int = 5
    lang_enhanced: int = 0
    dense_interpolation: int = 1
    interpolation_length: int = 2
    relative_action: int = 0
    denoise_model: str = "ddpm"  # "ddpm" or "rectified_flow"
    num_inference_steps: int = 100  # inference steps, RF uses 10
    bool_use_gating_and_adapter: int = 1  # ForesightDiffuserActorV6 gating and adapter

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'xyzw'


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Gripper workspace')
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )

    if args.test_model == "3d_diffuser_actor":
        model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
        ).to(device)
    elif args.test_model == "act3d":
        model = Act3D(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=(
                args.num_ghost_point_cross_attn_layers),
            num_query_cross_attn_layers=(
                args.num_query_cross_attn_layers),
            num_vis_ins_attn_layers=(
                args.num_vis_ins_attn_layers),
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=(
                args.fine_sampling_ball_diameter),
            regress_position_offset=bool(
                args.regress_position_offset),
            use_instruction=bool(args.use_instruction)
        ).to(device)
    elif args.test_model == "foresight_diffuser_actor_v3":
        model = ForesightDiffuserActorV3(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            bool_classifier_free_guidance=bool(args.bool_classifier_free_guidance),
            classifier_free_guidance_w=args.classifier_free_guidance_w,
            classifier_free_guidance_dropout_prob=1.0,
        ).to(device)
    elif args.test_model == 'foresight_diffuser_actor_v4':
        model = ForesightDiffuserActorV4(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            prob_dropout_vel_features=0.0,
        ).to(device)
    elif args.test_model == 'foresight_diffuser_actor_v5':
        model = ForesightDiffuserActorV5(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            prob_dropout_vel_features=0.0,
        ).to(device)
    elif args.test_model == 'foresight_diffuser_actor_v6':
        model = ForesightDiffuserActorV6(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            prob_dropout_vel_features=0.0,
            bool_use_gating_and_adapter=bool(args.bool_use_gating_and_adapter),
        ).to(device)
    else:
        raise NotImplementedError

    # Load model weights
    if Path(args.checkpoint).is_file():
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model_dict_weight = {}
        for key in model_dict["weight"]:
            _key = key[7:]
            model_dict_weight[_key] = model_dict["weight"][key]
        model.load_state_dict(model_dict_weight)
    else:
        print("Do not load model weights from", args.checkpoint)
        assert False
    model.eval()

    return model

def create_policy_client(args):
    """创建策略客户端"""
    # Gripper workspace is the union of workspaces for all tasks
    # 与 load_models 中的逻辑保持一致
    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Gripper workspace')
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )
    # 构建模型参数字典，参考 load_models 中的参数
    model_args = dict(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        use_instruction=bool(args.use_instruction),
        fps_subsampling_factor=args.fps_subsampling_factor,
        gripper_loc_bounds=gripper_loc_bounds,
        rotation_parametrization=args.rotation_parametrization,
        quaternion_format=args.quaternion_format,
        diffusion_timesteps=args.diffusion_timesteps,
        denoise_model=args.denoise_model,
        num_inference_steps=args.num_inference_steps,
        nhist=args.num_history,
        relative=bool(args.relative_action),
        lang_enhanced=bool(args.lang_enhanced),
        bool_classifier_free_guidance=bool(args.bool_classifier_free_guidance),
        classifier_free_guidance_w=args.classifier_free_guidance_w,
        classifier_free_guidance_dropout_prob=1.0,
        bool_use_gating_and_adapter=bool(args.bool_use_gating_and_adapter),
    )
    policy_client = PolicyClient(
        host="127.0.0.1",
        port=args.policy_server_port,
        model_name=args.test_model,
        model_args=model_args,
        model_checkpoint=str(args.checkpoint),
        device=args.device,
    )
    return policy_client

if __name__ == "__main__":
    # Arguments
    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    print("Arguments:")
    print(args)
    print("-" * 100)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    if args.bool_policy_server:
        print("使用策略服务器客户端模式", flush=True)
        model = create_policy_client(args)
    else:
        model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        apply_depth=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
        bool_expert_actioner=bool(args.bool_expert_actioner),
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    if args.bool_expert_actioner:
        actioner = ExpertActioner(
            policy=model,
            instructions=instruction,
            apply_cameras=args.cameras,
            action_dim=args.action_dim,
            predict_trajectory=bool(args.predict_trajectory)
        )
    else:
        actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory)
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            max_tries=args.max_tries,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history,
            num_future_frames=args.num_future_frames,
            vis_save_dir=args.vis_save_dir,
            bool_future_info=bool(args.bool_future_info),
            bool_use_ground_truth_action=bool(args.bool_use_ground_truth_action),
            bool_world_server=bool(args.bool_world_server),
            world_server_port=args.world_server_port,
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
