split=$1
ckpt_num=$2
policy_port=$3
world_port=$4
gpu_id=$5
num_episodes=$6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
THIRD_PARTY_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${THIRD_PARTY_DIR}/RLBench:${THIRD_PARTY_DIR}/PyRep:${PYTHONPATH:-}"

time=5
exp=physmani
checkpoint=train_logs/${exp}/diffusion_multitask-C120-B18-lr1e-4-2-H3-DT100/epoch_${ckpt_num}.pth
tasks=(
    $7
)
round=$8
data_dir=./data/rmt/physmani_bench/${split}
exp=${exp}_${split}_ckpt${ckpt_num}_time${time}_camready_round${round}
vis_save_dir=eval_logs/${exp}/diffusion_multitask-C120-B18-lr1e-4-2-H3-DT100/vis_evaluation

gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=5
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=1
embedding_dim=120
cameras="front,8,16,36"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
denoise_model=rectified_flow
num_inference_steps=10

quaternion_format=xyzw  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

export CUDA_VISIBLE_DEVICES=$gpu_id
num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --num_future_frames 10 \
    --test_model foresight_diffuser_actor_v6 \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 20 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.08 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1 \
    --vis_save_dir=$vis_save_dir \
    --bool_future_info 1 \
    --bool_use_ground_truth_action 0 \
    --bool_expert_actioner 0 \
    --denoise_model $denoise_model \
    --num_inference_steps $num_inference_steps \
    --bool_policy_server 1 \
    --bool_world_server 1 \
    --policy_server_port $policy_port \
    --world_server_port $world_port \
    --bool_use_gating_and_adapter 0
done
