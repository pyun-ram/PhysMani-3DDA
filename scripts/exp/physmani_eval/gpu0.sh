GPUID=0
POLICY_PORT=8765
WORLD_PORT=8866
CKPT_LIST=( 99999 94999 89999 84999 79999 )

echo "bash scripts/exp/physmani_eval/eval_sim_3dafdprf_policy_server.sh $GPUID 0.0.0.0 $POLICY_PORT"
echo "bash scripts/exp/physmani_eval/eval_sim_3dafdprf_world_server.sh $GPUID 0.0.0.0 $WORLD_PORT"

# Wait for enter
read -p "Press Enter to continue"

for CKPT in ${CKPT_LIST[@]}; do
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 beat_the_rotating_buzz 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 beat_the_rotating_buzz_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 insert_onto_rotating_peg 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 insert_onto_rotating_peg_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 pick_moving_target_on_the_table 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 pick_moving_target_on_the_table_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 place_cups_on_rotating_frame 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 place_cups_on_rotating_frame_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 remove_cups_from_rotating_frame 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 remove_cups_from_rotating_frame_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 put_rubbish_in_moving_bin 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 put_rubbish_in_moving_bin_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 push_moving_button 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 push_moving_button_high_speed 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 moving_basketball_in_hoop 1
bash scripts/exp/physmani_eval/eval_sim_3dafdprfv6_nogateadapter_client.sh test $CKPT $POLICY_PORT $WORLD_PORT $GPUID 100 moving_basketball_in_hoop_high_speed 1
done
