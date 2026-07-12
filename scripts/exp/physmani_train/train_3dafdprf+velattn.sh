main_dir=physmani

dataset=data/rmt/physmani_bench/train_package_compressed/
valset=data/rmt/physmani_bench/val_package_compressed/

lr=1e-4
interpolation_length=2
num_history=3
diffusion_timesteps=100
denoise_model=rectified_flow
num_inference_steps=10
B=18
C=120
ngpus=3
max_episodes_per_task=-1
quaternion_format=xyzw


# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node $ngpus --master_port $RANDOM \
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --trainer TrainTesterRMT \
    --model_name foresight_diffuser_actor_v6 \
    --tasks pick_moving_target_on_the_table place_cups_on_rotating_frame remove_cups_from_rotating_frame put_rubbish_in_moving_bin push_moving_button moving_basketball_in_hoop moving_basketball_in_hoop_high_speed insert_onto_rotating_peg beat_the_rotating_buzz pick_moving_target_on_the_table_high_speed place_cups_on_rotating_frame_high_speed remove_cups_from_rotating_frame_high_speed beat_the_rotating_buzz_high_speed insert_onto_rotating_peg_high_speed push_moving_button_high_speed put_rubbish_in_moving_bin_high_speed \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --gripper_loc_bounds_buffer 0.08 \
    --num_workers 2 \
    --train_iters 100000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --denoise_model $denoise_model \
    --num_inference_steps $num_inference_steps \
    --val_freq 5000 \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 6 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr \
    --num_history $num_history \
    --cameras "front" "8" "16" "36" \
    --max_episodes_per_task $max_episodes_per_task \
    --quaternion_format $quaternion_format \
    --bool_is_pretrain 1 \
    --checkpoint train_logs/3darf_pretrain/diffusion_multitask-C120-B18-lr1e-4-2-H3-DT100/epoch_99999.pth \
    --num_future_frames_obs 1 \
    --bool_finetune 1 \
    --prob_dropout 0.15 \
    --bool_use_gating_and_adapter 0 \
    --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-$interpolation_length-H$num_history-DT$diffusion_timesteps