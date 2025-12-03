CUDA_VISIBLE_DEVICES=3
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /primus_xpfs_workspace_T04/huangshijie/miniconda3/envs/wan_s2v/bin/torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "704*384" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model True \
     --convert_model_dtype \
     --prompt "A vibrant 3D anime style avatar of a young blonde woman with high pigtails, wearing a black Gothic Lolita dress with an off-shoulder neckline, lace choker, and gloves, speaking playfully to the camera with expressive, exaggerated hand gestures and a slight smirk. The camera is steady, and the background is a dark, simple studio."  \
     --image "examples/demo/case1.jpg" \
     --audio "examples/talk.wav" \
     --infer_frames 48 \
     --load_lora \
     --lora_path "/primus_xpfs_workspace_T04/huangyubo/Causvid/checkpoints/1005_s2v_causal_sft_1_4_0/ckpt/merged_model_step_step_25000/only_lora.pt"    \
     --lora_path_dmd "/primus_xpfs_workspace_T04/huangyubo/LongLive/1027_s2v_selfforcing_1_13_2/ckptcopy/step_2500_generator_lora.pt" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 10 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ 
     
