import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .diffuser_actor import DiffuserActor, DiffusionHead
from diffuser_actor.utils.position_encodings import SinusoidalPosEmb
from diffuser_actor.utils.utils import normalise_quat


class ForesightDiffuserActorV4(DiffuserActor):
    """
    ForesightDiffuserActorV4 extends DiffuserActor with velmap_pc support.
    Only adds velmap_pc query and fusion functionality, inherits all other logic.
    """

    def __init__(self, *args, **kwargs):
        # Extract prob_dropout_vel_features before passing to parent
        self.prob_dropout_vel_features = kwargs.pop('prob_dropout_vel_features', 0.0)
        super().__init__(*args, **kwargs)
        self.prediction_head = ForesightDiffusionHeadV4(
            embedding_dim=kwargs['embedding_dim'],
            use_instruction=kwargs['use_instruction'],
            rotation_parametrization=kwargs['rotation_parametrization'],
            nhist=kwargs['nhist'],
            lang_enhanced=kwargs['lang_enhanced'],
        )

    def _query_velmap(self, pos, fps_pos, velmap_pc_pos, velmap_pc_feat):
        """
        Query velmap_pc features for context and fps positions using nearest neighbor search.
        Uses batch-parallelized cdist for efficient GPU computation.
        
        Args:
            pos: (B, N_ctx, 3) context positions
            fps_pos: (B, N_fps, 3) fps sampled positions  
            velmap_pc_pos: (B, N_vel, 3) velmap point cloud positions
            velmap_pc_feat: (B, N_vel, 6) velmap point cloud features (6D velocity)
            
        Returns:
            context_vel_feats: (B, N_ctx, 6) velocity features for context points
            fps_vel_feats: (B, N_fps, 6) velocity features for fps points
        """
        # 1. 拼接查询点 (B, N_total, 3)
        query_pos = torch.cat([pos, fps_pos], dim=1)
        N_ctx = pos.shape[1]
        
        # 2. 一次性计算 Batch 距离 (B, N_total, N_vel)
        # torch.cdist 原生支持 Batch 运算，不需要 for loop，GPU 会并行处理 batch
        dists = torch.cdist(query_pos, velmap_pc_pos)
        
        # 3. 找到最近邻索引 (B, N_total)
        min_dists, nn_indices = torch.min(dists, dim=-1) # (B, N_total)
        
        # 4. Gather 特征
        # velmap_pc_feat: (B, N_vel, 6)
        B, N_total = nn_indices.shape
        C = velmap_pc_feat.shape[-1]
        
        # 使用 gather 获取特征
        # index_expanded: (B, N_total, 6) - 每个特征维度都用同一个 index 取
        index_expanded = nn_indices.unsqueeze(-1).expand(-1, -1, C)
        all_vel_feats = torch.gather(velmap_pc_feat, 1, index_expanded)
        
        DIST_THRESHOLD = 0.05
        
        # 创建 mask (B, N_total, 1)
        valid_mask = (min_dists < DIST_THRESHOLD).unsqueeze(-1).float().detach()
        
        # 距离过远的点，速度强制置 0
        all_vel_feats = all_vel_feats * valid_mask

        # 5. Split
        context_vel_feats = all_vel_feats[:, :N_ctx, :]
        fps_vel_feats = all_vel_feats[:, N_ctx:, :]
        
        return context_vel_feats, fps_vel_feats

    def _update_fixed_inputs(self, fixed_inputs, context_vel_feats, fps_vel_feats):
        """
        Update fixed_inputs with velocity features.
        
        Args:
            fixed_inputs: tuple of (context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz)
            context_vel_feats: (B, N_ctx, 6) velocity features for context points
            fps_vel_feats: (B, N_fps, 6) velocity features for fps points
            
        Returns:
            Updated fixed_inputs tuple with velocity features appended
        """
        context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz = fixed_inputs
        return (
            context_feats, context, instr_feats, adaln_gripper_feats, 
            fps_feats, fps_pos, context_vel_feats, fps_vel_feats
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs (includes velocity features)
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            context_vel_feats,
            fps_vel_feats
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
            context_vel_feats=context_vel_feats,
            fps_vel_feats=fps_vel_feats
        )

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        velmap_pc=None
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, visible_mask=None, instruction=instruction, curr_gripper=curr_gripper
        )
        
        # Query velocity features from velmap_pc if available
        if velmap_pc is not None:
            # Normalize velmap_pc positions to match normalized context
            # fixed_inputs[1] (context) and fixed_inputs[6] (fps_pos_xyz) are already normalized
            velmap_pc_normalized = velmap_pc.clone()
            velmap_pc_normalized[..., :3] = self.normalize_pos(velmap_pc[..., :3])
            context_vel_feats, fps_vel_feats = self._query_velmap(
                pos=fixed_inputs[1],  # context positions (normalized)
                fps_pos=fixed_inputs[6],  # fps_pos_xyz (normalized)
                velmap_pc_pos=velmap_pc_normalized[..., :3],  # (B, N_vel, 3) normalized
                velmap_pc_feat=velmap_pc_normalized[..., 3:9],  # (B, N_vel, 6)
            )
        else:
            # Create zero velocity features for backward compatibility
            B = fixed_inputs[1].shape[0]  # batch size from context
            N_ctx = fixed_inputs[1].shape[1]  # number of context points
            N_fps = fixed_inputs[6].shape[1]  # number of fps points
            device = fixed_inputs[1].device
            context_vel_feats = torch.zeros(B, N_ctx, 6, device=device, dtype=fixed_inputs[1].dtype)
            fps_vel_feats = torch.zeros(B, N_fps, 6, device=device, dtype=fixed_inputs[1].dtype)
        
        fixed_inputs = self._update_fixed_inputs(
            fixed_inputs=fixed_inputs,
            context_vel_feats=context_vel_feats,
            fps_vel_feats=fps_vel_feats,
        )
        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
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
            "attention": {},
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
        run_inference=False,
        velmap_pc=None,
        **kwargs
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)
            velmap_pc: (B, N, 3+6)
        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                velmap_pc=velmap_pc
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, visible_mask=None, instruction=instruction, curr_gripper=curr_gripper
        )
        
        # Query velocity features from velmap_pc if available
        if velmap_pc is not None:
            # Normalize velmap_pc positions to match normalized context
            # fixed_inputs[1] (context) and fixed_inputs[6] (fps_pos_xyz) are already normalized
            velmap_pc_normalized = velmap_pc.clone()
            velmap_pc_normalized[..., :3] = self.normalize_pos(velmap_pc[..., :3])
            context_vel_feats, fps_vel_feats = self._query_velmap(
                pos=fixed_inputs[1],  # context positions (normalized)
                fps_pos=fixed_inputs[6],  # fps_pos_xyz (normalized)
                velmap_pc_pos=velmap_pc_normalized[..., :3],  # (B, N_vel, 3) normalized
                velmap_pc_feat=velmap_pc_normalized[..., 3:9],  # (B, N_vel, 6)
            )
        else:
            # Create zero velocity features for backward compatibility
            B = fixed_inputs[1].shape[0]  # batch size from context
            N_ctx = fixed_inputs[1].shape[1]  # number of context points
            N_fps = fixed_inputs[6].shape[1]  # number of fps points
            device = fixed_inputs[1].device
            context_vel_feats = torch.zeros(B, N_ctx, 6, device=device, dtype=fixed_inputs[1].dtype)
            fps_vel_feats = torch.zeros(B, N_fps, 6, device=device, dtype=fixed_inputs[1].dtype)
        
        # Apply velocity features dropout during training
        # Sample-wise Dropout (Best for RSS DDL)
        if self.training and self.prob_dropout_vel_features > 0:
            B = context_vel_feats.shape[0]
            device = context_vel_feats.device
            
            # 对每个样本独立决策：要么全留(1)，要么全丢(0)
            keep_mask = (torch.rand(B, device=device) > self.prob_dropout_vel_features).float()
            
            keep_mask = keep_mask.view(B, 1, 1) 
            
            context_vel_feats = context_vel_feats * keep_mask
            fps_vel_feats = fps_vel_feats * keep_mask

        fixed_inputs = self._update_fixed_inputs(
            fixed_inputs=fixed_inputs,
            context_vel_feats=context_vel_feats,
            fps_vel_feats=fps_vel_feats,
        )
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


class ForesightDiffusionHeadV4(DiffusionHead):
    """
    ForesightDiffusionHeadV4 extends DiffusionHead with velocity feature encoding and fusion.
    Only adds velocity-related functionality, inherits all other logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedding_dim = kwargs.get('embedding_dim', 60)
        
        # Velocity feature encoder: 6D -> tanh -> sin_embed -> MLP -> embedding_dim
        # Process each of the 6 dimensions separately with sin embedding, then combine
        # Each dimension gets approximately embedding_dim//6, with remainder distributed
        base_dim = embedding_dim // 6
        remainder = embedding_dim % 6
        # Distribute remainder across first few dimensions
        self.vel_sin_emb_dims = [base_dim + (1 if i < remainder else 0) for i in range(6)]
        self.vel_sin_embs = nn.ModuleList([
            SinusoidalPosEmb(dim) for dim in self.vel_sin_emb_dims
        ])
        total_sin_emb_dim = sum(self.vel_sin_emb_dims)
        # MLP to combine the 6 sin-embedded dimensions into embedding_dim
        self.vel_feature_mlp = nn.Sequential(
            nn.Linear(total_sin_emb_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Context-Aware Velocity Adapter
        # Input: [Visual_Feat, Velocity_Feat] -> Output: Delta_Feat
        # Expanded middle layer (4x) for better cross-modal interaction
        hidden_dim = embedding_dim * 4  # 120 * 4 = 480
        self.vel_adapter = nn.Sequential(
            # Normalize input to align Visual and Velocity feature distributions
            nn.LayerNorm(embedding_dim * 2),
            # First layer: expand and fuse
            nn.Linear(embedding_dim * 2, hidden_dim),  # 240 -> 480
            # Activation: GELU is standard for Transformer
            nn.GELU(),
            # Second layer: reduce dimension (Zero-Init target)
            nn.Linear(hidden_dim, embedding_dim)  # 480 -> 120
        )
        # Zero initialization for the last layer to ensure training stability
        # This is critical: with zero weights and bias, vel_adapter output is zero at initialization
        # Residual connection: context_features = context_features + 0 = context_features
        # This ensures training starts with behavior identical to DiffuserActor (no velocity influence)
        # The network gradually learns to incorporate velocity features as training progresses
        nn.init.zeros_(self.vel_adapter[-1].weight)
        nn.init.zeros_(self.vel_adapter[-1].bias)

    def encode_velocity_features(self, vel_feats):
        """
        Encode 6D velocity features to embedding_dim.
        
        Args:
            vel_feats: (B, N, 6) velocity features
            
        Returns:
            vel_emb: (B, N, embedding_dim) encoded velocity features
        """
        B, N, _ = vel_feats.shape
        
        # Apply tanh to limit range to [-1, 1]
        vel_feats = torch.tanh(vel_feats)  # (B, N, 6)
        # scale up to [1000, -1000] for distinguish 0.01 resolution
        vel_feats = vel_feats * 1000
        
        # Apply sin embedding to each dimension separately
        # vel_feats: (B, N, 6) -> split into 6 dims -> (B, N, 1) each
        vel_emb_list = []
        for dim_idx in range(6):
            dim_values = vel_feats[:, :, dim_idx:dim_idx+1]  # (B, N, 1)
            dim_values_flat = dim_values.reshape(-1)  # (B*N,)
            dim_emb = self.vel_sin_embs[dim_idx](dim_values_flat)  # (B*N, vel_sin_emb_dims[dim_idx])
            dim_emb = dim_emb.reshape(B, N, self.vel_sin_emb_dims[dim_idx])  # (B, N, vel_sin_emb_dims[dim_idx])
            vel_emb_list.append(dim_emb)
        
        # Concatenate all 6 dimensions: (B, N, sum(vel_sin_emb_dims))
        vel_emb = torch.cat(vel_emb_list, dim=-1)
        
        # Apply MLP to reduce to embedding_dim
        vel_emb = self.vel_feature_mlp(vel_emb)  # (B, N, embedding_dim)
        
        return vel_emb

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos, context_vel_feats, fps_vel_feats):
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
            context_vel_feats: (B, N_ctx, 6) velocity features for context points
            fps_vel_feats: (B, N_fps, 6) velocity features for fps points
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
            context_vel_feats,
            fps_vel_feats,
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
                        instr_feats, context_vel_feats, fps_vel_feats):
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
            context_vel_feats: (B, N_ctx, 6) velocity features for context points
            fps_vel_feats: (B, N_fps, 6) velocity features for fps points
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # ================= VELOCITY FEATURE FUSION START =================
        # Optimized: concat first, then encode once, finally split
        # This is more efficient than encoding separately
        B = context_vel_feats.shape[0]
        N_ctx = context_vel_feats.shape[1]
        N_fps = fps_vel_feats.shape[1]
        
        # Concat all velocity features: (B, N_ctx+N_fps, 6)
        all_vel_feats = torch.cat([context_vel_feats, fps_vel_feats], dim=1)
        
        # Encode once: (B, N_ctx+N_fps, embedding_dim)
        all_vel_emb = self.encode_velocity_features(all_vel_feats)
        
        # Split back: (B, N_ctx, F) and (B, N_fps, F)
        context_vel_emb = all_vel_emb[:, :N_ctx, :]
        fps_vel_emb = all_vel_emb[:, N_ctx:, :]
        
        # Adjust dimensions to match context_features: (N, B, F)
        context_vel_emb = einops.rearrange(context_vel_emb, 'b n f -> n b f')  # (N_ctx, B, F)
        fps_vel_emb = einops.rearrange(fps_vel_emb, 'b n f -> n b f')  # (N_fps, B, F)
        
        # Concatenate visual and velocity features
        # context_features: (N_ctx, B, F), context_vel_emb: (N_ctx, B, F)
        ctx_input = torch.cat([context_features, context_vel_emb], dim=-1)  # (N_ctx, B, 2*F)
        # sampled_context_features: (N_fps, B, F), fps_vel_emb: (N_fps, B, F)
        smp_input = torch.cat([sampled_context_features, fps_vel_emb], dim=-1)  # (N_fps, B, 2*F)
        
        # Record split points for later separation
        num_ctx = ctx_input.shape[0]  # N_ctx
        num_smp = smp_input.shape[0]   # N_fps
        
        # Concatenate along point dimension (dim 0) for batch processing
        all_input = torch.cat([ctx_input, smp_input], dim=0)  # (N_ctx+N_fps, B, 2*F)
        
        # Pass through MLP to compute residual update (Delta)
        all_delta = self.vel_adapter(all_input)  # (N_ctx+N_fps, B, F)
        
        # Split back into two parts
        ctx_delta, smp_delta = torch.split(all_delta, [num_ctx, num_smp], dim=0)
        
        # Residual connection
        context_features = context_features + ctx_delta  # (N_ctx, B, F)
        sampled_context_features = sampled_context_features + smp_delta  # (N_fps, B, F)
        # ================= VELOCITY FEATURE FUSION END =================

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

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
