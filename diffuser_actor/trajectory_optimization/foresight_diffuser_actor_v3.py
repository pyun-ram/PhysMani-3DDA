import torch
from .diffuser_actor import DiffuserActor, normalise_quat, einops, FFWRelativeCrossAttentionModule
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffusionHead
import torch.nn.functional as F

class ForesightDiffuserActorV3(DiffuserActor):
    def __init__(self, backbone="clip", image_size=..., embedding_dim=60, num_vis_ins_attn_layers=2, use_instruction=False, fps_subsampling_factor=5, gripper_loc_bounds=None, rotation_parametrization='6D', quaternion_format='xyzw', diffusion_timesteps=100, denoise_model="ddpm", num_inference_steps=100, nhist=3, relative=False, lang_enhanced=False, bool_rtn_attn=False, bool_classifier_free_guidance=False, classifier_free_guidance_w=1.0, classifier_free_guidance_dropout_prob=0.1):
        super().__init__(backbone, image_size, embedding_dim, num_vis_ins_attn_layers, use_instruction, fps_subsampling_factor, gripper_loc_bounds, rotation_parametrization, quaternion_format, diffusion_timesteps, denoise_model, num_inference_steps, nhist, relative, lang_enhanced)
        self.prediction_head = ForesightDiffusionHeadV3(
                    embedding_dim=embedding_dim,
                    use_instruction=use_instruction,
                    rotation_parametrization=rotation_parametrization,
                    nhist=nhist,
                    lang_enhanced=lang_enhanced
                )
        self.foresight_cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads=8, num_layers=2, use_adaln=False, bool_rtn_attn=bool_rtn_attn
        )
        self.bool_rtn_attn = bool_rtn_attn
        self.bool_classifier_free_guidance = bool_classifier_free_guidance
        self.classifier_free_guidance_w = classifier_free_guidance_w
        self.classifier_free_guidance_dropout_prob = classifier_free_guidance_dropout_prob
        return

    def conditional_sample_CFG(self, condition_data, condition_mask, fixed_inputs):
        # 根据模型类型设置 timesteps
        if self.denoise_model == "ddpm":
            self.position_noise_scheduler.set_timesteps(self.n_steps)
            self.rotation_noise_scheduler.set_timesteps(self.n_steps)
        elif self.denoise_model == "rectified_flow":
            device = condition_data.device
            self.position_noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
            self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps, device=device)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        if self.denoise_model == "ddpm":
            noise_t = torch.ones(
                (len(condition_data),), device=condition_data.device
            ).long().mul(self.position_noise_scheduler.timesteps[0])
        elif self.denoise_model == "rectified_flow":
            # RF uses float timesteps, start from 1.0
            noise_t = torch.ones(
                (len(condition_data),), device=condition_data.device
            ).float() * 1.0
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        cond_fixed_inputs = fixed_inputs
        uncond_fixed_inputs = [itm.clone() for itm in fixed_inputs]
        uncond_fixed_inputs[-1] *= 0
        uncond_fixed_inputs[-2] *= 0
        for t_ind, t in enumerate(timesteps):
            # 对于 policy_forward_pass，DDPM 需要 long [0, n_steps-1]，RF 需要 float [0, 1]
            if self.denoise_model == "ddpm":
                timestep_for_forward = t * torch.ones(len(trajectory)).to(trajectory.device).long()
            elif self.denoise_model == "rectified_flow":
                # RF 的 t 是 float [1.0, 0.0]，直接使用（与训练时一致）
                timestep_for_forward = t * torch.ones(len(trajectory)).to(trajectory.device)
            
            pred_eps_cond = self.policy_forward_pass(
                trajectory,
                timestep_for_forward,
                cond_fixed_inputs
            )[-1]
            pred_eps_uncond = self.policy_forward_pass(
                trajectory,
                timestep_for_forward,
                uncond_fixed_inputs
            )[-1]
            out = (1+self.classifier_free_guidance_w) * pred_eps_cond - self.classifier_free_guidance_w * pred_eps_uncond

            # step 方法：DDPM 使用 t（timestep 值），RF 使用 t_ind（索引）
            if self.denoise_model == "ddpm":
                pos = self.position_noise_scheduler.step(
                    out[..., :3], t, trajectory[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    out[..., 3:9], t, trajectory[..., 3:9]
                ).prev_sample
            elif self.denoise_model == "rectified_flow":
                pos = self.position_noise_scheduler.step(
                    out[..., :3], t_ind, trajectory[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    out[..., 3:9], t_ind, trajectory[..., 3:9]
                ).prev_sample
            trajectory = torch.cat((pos, rot), -1)
        trajectory = torch.cat((trajectory, pred_eps_cond[..., 9:]), -1)
        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        next_rgb_obs,
        next_pcd_obs,
        next_mask_obs,
        next_gripper,
        next_frame_relative_id,
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        next_pcd_obs = next_pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        next_gripper = next_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        next_pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(next_pcd_obs, [0, 1, 2, 4, 5, 3])
        ), [0, 1, 2, 5, 3, 4])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)
        next_gripper[..., :3] = self.normalize_pos(next_gripper[..., :3])
        next_gripper = self.convert_rot(next_gripper)

        # Prepare inputs
        B, T, num_cameras, _, H, W = next_rgb_obs.shape
        # Create a all valid mask for current observation
        mask_obs = torch.ones((B,num_cameras,1,H,W), dtype=next_mask_obs.dtype, device=next_mask_obs.device)
        # Create instruction and gripper for next observation
        next_instruction = instruction.unsqueeze(1).expand(-1, T, -1, -1)
        dummy_gripper = curr_gripper.unsqueeze(1).expand(-1, T, -1, -1)

        rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper = self.assemble_inputs(
            rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper,
            next_rgb_obs, next_pcd_obs, next_mask_obs, next_instruction, dummy_gripper,
        )
        fixed_inputs = self.encode_inputs(rgb_obs, pcd_obs, visible_mask=mask_obs, instruction=instruction, curr_gripper=curr_gripper)
        fixed_inputs, fixed_next_inputs = self.split_fixed_inputs(
            fixed_inputs,
            B=B,
            T=T,
            bool_next_pcd_obs=next_pcd_obs is not None,
        )
        fixed_inputs = self.encode_feature_in_time(
            fixed_inputs,
            fixed_next_inputs,
            next_frame_relative_id,
            next_gripper,
        )
        # remove the sampled xyz of fixed_inputs
        attn_weights_dict = fixed_inputs[-1]
        fixed_inputs = fixed_inputs[:-1]
        # Condition on start-end pose
        _, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        if self.bool_classifier_free_guidance:
            trajectory = self.conditional_sample_CFG(
                cond_data,
                cond_mask,
                fixed_inputs
            )
        else:
            trajectory = self.conditional_sample(
                cond_data,
                cond_mask,
                fixed_inputs
            )
        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()
        output_dict = {
            "action": trajectory,
            "attention": attn_weights_dict,
        }
        return output_dict

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        next_rgb_obs,
        next_pcd_obs,
        next_mask_obs,
        next_gripper,
        next_frame_relative_id,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)
            next_gripper: (B, T, 3+4+X)
            next_rgb_obs: (B, T, num_cameras, 3, H, W) in [0, 1]
            next_pcd_obs: (B, T, num_cameras, 3, H, W) in world coordinates
            next_mask_obs: (B, T, num_cameras, 1, H, W) in [0, 1], True if the pixel is valid
            next_frame_relative_id: (B, T)
            run_inference: bool
        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if self._relative:
            curr_gripper_org = curr_gripper.clone()
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
            next_pcd_obs, _ = self.convert2rel(next_pcd_obs, curr_gripper_org)
            assert False
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]
        next_gripper = next_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                next_gripper=next_gripper,
                next_rgb_obs=next_rgb_obs,
                next_pcd_obs=next_pcd_obs,
                next_mask_obs=next_mask_obs,
                next_frame_relative_id=next_frame_relative_id,
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        next_pcd_obs = next_pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        next_gripper = next_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        next_pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(next_pcd_obs, [0, 1, 2, 4, 5, 3])
        ), [0, 1, 2, 5, 3, 4])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        next_gripper[..., :3] = self.normalize_pos(next_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)
        next_gripper = self.convert_rot(next_gripper)

        # Prepare inputs
        B, T, num_cameras, _, H, W = next_rgb_obs.shape
        # Create a all valid mask for current observation
        mask_obs = torch.ones((B,num_cameras,1,H,W), dtype=next_mask_obs.dtype, device=next_mask_obs.device)
        # Create instruction and gripper for next observation
        next_instruction = instruction.unsqueeze(1).expand(-1, T, -1, -1)
        # fake next_gripper, just for assemble and split
        dummy_gripper = curr_gripper.unsqueeze(1).expand(-1, T, -1, -1)

        rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper = self.assemble_inputs(
            rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper,
            next_rgb_obs, next_pcd_obs, next_mask_obs, next_instruction, dummy_gripper,
        )
        fixed_inputs = self.encode_inputs(rgb_obs, pcd_obs, visible_mask=mask_obs, instruction=instruction, curr_gripper=curr_gripper)
        fixed_inputs, fixed_next_inputs = self.split_fixed_inputs(
            fixed_inputs,
            B=B,
            T=T,
            bool_next_pcd_obs=next_pcd_obs is not None,
        )
        fixed_inputs = self.encode_feature_in_time(
            fixed_inputs,
            fixed_next_inputs,
            next_frame_relative_id,
            next_gripper,
        )
        # remove the sampled xyz of fixed_inputs
        fixed_inputs = fixed_inputs[:-1]

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        if self.denoise_model == "ddpm":
            timesteps = torch.randint(
                0,
                self.position_noise_scheduler.config.num_train_timesteps,
                (len(noise),), device=noise.device
            ).long()
        elif self.denoise_model == "rectified_flow":
            timesteps = self.position_noise_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        if self.bool_classifier_free_guidance:
            # randomly dropout conditions
            dropout_rate = self.classifier_free_guidance_dropout_prob
            c = fixed_inputs[-2]
            c_pos = fixed_inputs[-1]
            mask = torch.rand(c.shape[1], device=c.device) < dropout_rate
            maskout_idx = torch.where(mask)[0]
            c_mask = torch.ones_like(c)
            c_mask[:,maskout_idx] = 0
            c = c * c_mask
            c_pos_mask = torch.ones_like(c_pos)
            c_pos_mask[maskout_idx, :] = 0
            c_pos = c_pos * c_pos_mask

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            # 根据模型类型选择 target
            if self.denoise_model == "ddpm":
                # DDPM: 预测噪声
                target_pos = noise[..., :3]
                target_rot = noise[..., 3:9]
            elif self.denoise_model == "rectified_flow":
                # RF: 预测速度场 (noise - gt)
                target_pos = self.position_noise_scheduler.prepare_target(
                    noise[..., :3], gt_trajectory[..., :3]
                )
                target_rot = self.rotation_noise_scheduler.prepare_target(
                    noise[..., 3:9], gt_trajectory[..., 3:9]
                )
            loss = (
                30 * F.l1_loss(trans, target_pos, reduction='mean')
                + 10 * F.l1_loss(rot, target_rot, reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            fps_pos_xyz,
            c, c_pos
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos,
            condition=c,
            condition_pos=c_pos,
        )
    
    def split_fixed_inputs(
        self,
        fixed_inputs,
        B: int,
        T: int,
        bool_next_pcd_obs: bool = True,
    ):
        '''
        Args:
            fixed_inputs: (B+BT)
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos, fps_pos_xyz  # sampled visual features
        Returns:
            curr_fixed_inputs
                context_feats torch.Size([B, 4096, 120])
                context torch.Size([B, 4096, 3])
                instr_feats torch.Size([B, 53, 120])
                adaln_gripper_feats torch.Size([B, 3, 120])
                fps_feats torch.Size([819, B, 120])
                fps_pos torch.Size([B, 819, 120, 2])
                fps_pos_xyz torch.Size([B, 819, 3]) / None
            next_fixed_inputs
                context_feats torch.Size([B, T, 4096, 120])
                context torch.Size([B, T, 4096, 3])
                instr_feats None (del)
                adaln_gripper_feats None (del)
                fps_feats torch.Size([819, B, T, 120])
                fps_pos torch.Size([B, T, 819, 120, 2])
                fps_pos_xyz torch.Size([B, T, 819, 3]) / None
        '''
        if not bool_next_pcd_obs:
            return fixed_inputs, None
        context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz = fixed_inputs
        context_feats = einops.rearrange(
            context_feats, '(b tp1) l c -> b tp1 l c', b=B, tp1=T+1)
        context = einops.rearrange(
            context, '(b tp1) l c -> b tp1 l c', b=B, tp1=T+1)
        instr_feats = einops.rearrange(
            instr_feats, '(b tp1) l c -> b tp1 l c', b=B, tp1=T+1)
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, '(b tp1) l c -> b tp1 l c', b=B, tp1=T+1)
        fps_feats = einops.rearrange(
            fps_feats, 'n (b tp1) c -> n b tp1 c', b=B, tp1=T+1)
        fps_pos = einops.rearrange(
            fps_pos, '(b tp1) l c d -> b tp1 l c d', b=B, tp1=T+1)
        if fps_pos_xyz is not None:
            fps_pos_xyz = einops.rearrange(
                fps_pos_xyz, '(b tp1) l c -> b tp1 l c', b=B, tp1=T+1)
        else:
            fps_pos_xyz = None
        fixed_inputs = (
            context_feats[:, 0],
            context[:, 0],
            instr_feats[:, 0],
            adaln_gripper_feats[:, 0],
            fps_feats[:, :, 0],
            fps_pos[:, 0],
            fps_pos_xyz[:, 0] if fps_pos_xyz is not None else None,
        )
        next_fixed_inputs = (
            context_feats[:, 1:],
            context[:, 1:],
            None,
            None,
            fps_feats[:, :,1:],
            fps_pos[:, 1:],
            fps_pos_xyz[:, 1:] if fps_pos_xyz is not None else None,
        )
        return fixed_inputs, next_fixed_inputs

    def assemble_inputs(
        self,
        rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper,
        next_rgb_obs, next_pcd_obs, next_mask_obs, next_instruction, next_gripper,
    ):
        # visible_rgb: [B, ncam, 3, 256, 256] + [B, T, ncam, 3, 256, 256]
        # -> [B + BT, ncam, 3, 256, 256]
        rgb_obs = torch.cat([rgb_obs.unsqueeze(1), next_rgb_obs], dim=1)
        rgb_obs = einops.rearrange(rgb_obs, 'b t n c h w -> (b t) n c h w')
        # visible_pcd: [B, ncam, 3, 256, 256] + [B, T, ncam, 3, 256, 256]
        # -> [B + BT, ncam, 3, 256, 256]
        pcd_obs = torch.cat([pcd_obs.unsqueeze(1), next_pcd_obs], dim=1)
        pcd_obs = einops.rearrange(pcd_obs, 'b t n c h w -> (b t) n c h w')
        # visible_mask: [B, ncam, 1, 256, 256] + [B, T, ncam, 1, 256, 256]
        # -> [B + BT, ncam, 1, 256, 256]
        mask_obs = torch.cat([mask_obs.unsqueeze(1), next_mask_obs], dim=1)
        mask_obs = einops.rearrange(mask_obs, 'b t n c h w -> (b t) n c h w')
        # instruction: [B, 53, 512] + [B, T, 53, 512] -> [B + BT, 53, 512]
        instruction = torch.cat([instruction.unsqueeze(1), next_instruction], dim=1)
        instruction = einops.rearrange(instruction, 'b t l c -> (b t) l c')
        # curr_gripper: [B, 3, 9] + [B, T, 3, 9] -> [B + BT, 3, 9]
        curr_gripper = torch.cat([curr_gripper.unsqueeze(1), next_gripper], dim=1)
        curr_gripper = einops.rearrange(curr_gripper, 'b t l c -> (b t) l c')
        return rgb_obs, pcd_obs, mask_obs, instruction, curr_gripper

    def encode_feature_in_time(
        self,
        fixed_inputs,
        fixed_next_inputs,
        next_frame_relative_id,
        next_gripper,
    ):
        context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz = fixed_inputs
        _, _, _, _, next_fps_feats, next_fps_pos, _ = fixed_next_inputs
        t = 0
        global_time_pos = self.prediction_head.traj_time_emb(torch.tensor([t], device=fps_feats.device))[None]
        context_time_pos = global_time_pos.repeat(context_feats.shape[0], context_feats.shape[1], 1) 
        context_feats = context_feats + context_time_pos
        fps_time_pos = global_time_pos.repeat(fps_feats.shape[0], fps_feats.shape[1], 1) 
        fps_feats = fps_feats + fps_time_pos

        B, T = next_frame_relative_id.shape
        global_time_pos = self.prediction_head.traj_time_emb(
            next_frame_relative_id.reshape(-1)
        )
        global_time_pos = global_time_pos.reshape(B, T, 1, -1)
        next_fps_time_pos = global_time_pos.permute(2,0,1,3).repeat(next_fps_feats.shape[0], 1, 1, 1)
        next_fps_feats = next_fps_feats + next_fps_time_pos

        T = next_frame_relative_id.shape[1]
        next_gripper_feats = self.prediction_head.traj_encoder(next_gripper)
        next_gripper_time_pos = self.prediction_head.traj_time_emb(
            next_frame_relative_id.reshape(-1)
        ).reshape(next_gripper_feats.shape)
        next_gripper_feats, _ = self.prediction_head.traj_lang_attention[0](
            seq1=next_gripper_feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=next_gripper_time_pos, seq2_sem_pos=None
        )
        next_gripper_feats = next_gripper_feats + next_gripper_time_pos
        next_gripper_feats = einops.rearrange(
            next_gripper_feats, 'b l c -> l b c'
        )
        rel_next_gripper_pos = self.prediction_head.relative_pe_layer(next_gripper[..., :3])
        L, B, T, C = next_fps_feats.shape
        attn_weights_dict = {}
        c_list = []
        for idx_future in range(T):
            output = self.foresight_cross_attn(
                query=next_gripper_feats[idx_future:idx_future+1],
                value=next_fps_feats[:,:,idx_future],
                query_pos=rel_next_gripper_pos[:, idx_future:idx_future+1],
                value_pos=next_fps_pos[:,idx_future],
                diff_ts=None
            )[-1]
            c_list.append(output)
        c = torch.cat(c_list, dim=0)
        c_pos = rel_next_gripper_pos
        return (context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz, c, c_pos, attn_weights_dict)
class ForesightDiffusionHeadV3(DiffusionHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3,
                 lang_enhanced=False):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced,
        )
        self.foresight_cross_attn1 = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )
        self.foresight_cross_attn2 = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos, condition, condition_pos):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats,
            condition,
            condition_pos
        )
        # pos_pred torch.Size([24, 1, 3])
        # rot_pred torch.Size([24, 1, 6])
        # openess_pred torch.Size([24, 1, 1])
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats,
                        condition,
                        condition_pos):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # gripper_pcd torch.Size([32, 1, 3])
        # gripper_features torch.Size([1, 32, 120])
        # context_pcd torch.Size([32, 1024, 3])
        # context_features torch.Size([1024, 32, 120])
        # timesteps torch.Size([32])
        # curr_gripper_features torch.Size([3, 32, 120])
        # sampled_context_features torch.Size([204, 32, 120])
        # sampled_rel_context_pos torch.Size([32, 204, 120, 2])
        # instr_feats torch.Size([32, 53, 120])
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]
        res = self.foresight_cross_attn1(
            query=gripper_features,
            value=condition,
            query_pos=rel_gripper_pos,
            value_pos=condition_pos,
            diff_ts=None,
        )[-1]
        gripper_features = gripper_features + res

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]
        gripper_features = features[:num_gripper]
        rel_gripper_pos = rel_gripper_pos[:, :num_gripper]
        res = self.foresight_cross_attn2(
            query=gripper_features,
            value=condition,
            query_pos=rel_gripper_pos,
            value_pos=condition_pos,
            diff_ts=None,
        )[-1]
        new_features = torch.zeros_like(features)
        new_features[:num_gripper] = res
        features = features + new_features

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return position, rotation, openess
