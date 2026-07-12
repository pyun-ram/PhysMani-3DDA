import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .diffuser_actor import DiffuserActor
from diffuser_actor.utils.position_encodings import SinusoidalPosEmb
from diffuser_actor.utils.utils import normalise_quat
import math

class VelocitySinEmbedding(nn.Module):
    """
    Shared velocity sinusoidal embedding module.
    Encodes 6D velocity -> sin_embed -> MLP -> embedding_dim.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Distribute embedding_dim across 6 velocity dimensions for sin emb
        base_dim = embedding_dim // 6
        remainder = embedding_dim % 6
        self.vel_sin_emb_dims = [base_dim + (1 if i < remainder else 0) for i in range(6)]
        self.vel_sin_embs = nn.ModuleList([
            SinusoidalPosEmb(dim) for dim in self.vel_sin_emb_dims
        ])
        self.total_sin_emb_dim = sum(self.vel_sin_emb_dims)
        # MLP: sin_emb -> embedding_dim
        self.vel_feature_mlp = nn.Sequential(
            nn.Linear(self.total_sin_emb_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, vel):
        """
        Encode 6D velocity to embedding_dim.
        
        Args:
            vel: (..., 6) velocity features (any shape with last dim=6)
            
        Returns:
            encoded_vel: (..., embedding_dim) encoded velocity
        """
        original_shape = vel.shape
        vel_flat = vel.reshape(-1, 6)  # (..., 6)
        
        # Apply tanh and scale
        vel_flat = torch.tanh(vel_flat) * 1000.0  # (..., 6)
        
        # Apply sin embedding to each dimension separately
        vel_emb_list = []
        for dim_idx in range(6):
            dim_values = vel_flat[:, dim_idx]  # (...,)
            dim_emb = self.vel_sin_embs[dim_idx](dim_values)  # (..., vel_sin_emb_dims[dim_idx])
            vel_emb_list.append(dim_emb)
        
        # Concatenate all 6 dimensions
        encoded_vel = torch.cat(vel_emb_list, dim=-1)  # (..., total_sin_emb_dim)
        
        # Reshape to original shape (except last dimension)
        new_shape = list(original_shape[:-1]) + [self.total_sin_emb_dim]
        encoded_vel = encoded_vel.reshape(new_shape)
        
        # MLP -> embedding_dim
        encoded_vel = self.vel_feature_mlp(encoded_vel)  # (..., embedding_dim)
        
        return encoded_vel

# --- 新增核心组件：Local Cross-Attention Layer ---
class VelMapCrossAttentionLayer(nn.Module):
    """
    Local Cross-Attention Layer:
    - Query: Base Visual Features (Context)
    - Key/Value: VelMap Neighbors (Velocity + Relative Position)
    
    参数量估算 (D=120):
    - Projections (Q, K, V, O): 4 * 120 * 120 ≈ 57k
    - Pos MLP: 3->120->120 ≈ 15k
    - FFN: 120 -> 480 -> 120 ≈ 115k
    - Norms: Negligible
    -> 单层约 187k 参数。堆叠 4 层就能增加 ~0.75M 参数。
    """
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "dim must be divisible by heads"
        
        # Multi-head Attention Projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Position Encoding MLP (Encode Relative Position into Key)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # FFN (Feed Forward Network) - 参数量大户
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_feats, neighbor_vel_emb, neighbor_rel_pos, mask=None):
        """
        Args:
            query_feats: (B, N, C) - Visual features (Query)
            neighbor_vel_emb: (B, N, K, C) - Velocity embeddings (Value Source)
            neighbor_rel_pos: (B, N, K, 3) - Relative position (Geometry Source)
            mask: (B, N, K) - True where neighbor is INVALID (padding)
        """
        B, N, C = query_feats.shape
        _, _, K, _ = neighbor_vel_emb.shape
        
        residual = query_feats
        
        # 1. Prepare Q, K, V
        # Q comes from Visual Features
        q = self.q_proj(query_feats).view(B, N, 1, self.num_heads, self.head_dim) # (B, N, 1, H, D)
        
        # K comes from Velocity + Position
        pos_emb = self.pos_mlp(neighbor_rel_pos) # (B, N, K, C)
        k_input = neighbor_vel_emb + pos_emb     # Fuse velocity content with geometry
        k = self.k_proj(k_input).view(B, N, K, self.num_heads, self.head_dim) # (B, N, K, H, D)
        
        # V comes from Velocity
        v = self.v_proj(neighbor_vel_emb).view(B, N, K, self.num_heads, self.head_dim) # (B, N, K, H, D)
        
        # 2. Attention Score: Q @ K.T -> (B, N, 1, H, K)
        # Transpose K to (B, N, H, D, K) for matmul
        k_t = k.permute(0, 1, 3, 4, 2)
        q_perm = q.permute(0, 1, 3, 2, 4) # (B, N, H, 1, D)
        
        attn_weights = torch.matmul(q_perm, k_t) / math.sqrt(self.head_dim) # (B, N, H, 1, K)
        
        # 3. Masking
        if mask is not None:
            # mask is (B, N, K), need (B, N, 1, 1, K) for broadcasting
            mask_expanded = mask.unsqueeze(2).unsqueeze(3) 
            attn_weights = attn_weights.masked_fill(mask_expanded, float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1) # (B, N, H, 1, K)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0) # Handle all-masked case

        # 4. Weighted Sum: Attn @ V -> (B, N, H, 1, D)
        v_perm = v.permute(0, 1, 3, 2, 4) # (B, N, H, K, D)
        # (B, N, H, 1, K) @ (B, N, H, K, D) -> (B, N, H, 1, D)
        output = torch.matmul(attn_probs, v_perm).squeeze(3) # (B, N, H, D)
        
        # Reshape and Project
        output = output.reshape(B, N, C)
        output = self.o_proj(output)
        
        # 5. Residual + Norm + FFN
        output = self.norm1(residual + output)
        output = self.norm2(output + self.ffn(output))
        
        return output

class VelocityFeatureModuleV2(nn.Module):
    """
    VelocityFeatureModule V2:
    - 替换 MaxPool 为 Deep Local Cross-Attention
    - 支持堆叠多层 Transformer Layer 以增加参数量和深度
    """
    def __init__(self, embedding_dim, K=16, dist_threshold=0.05, num_layers=3, bool_use_gating_and_adapter=True):
        super().__init__()
        self.K = K
        self.embedding_dim = embedding_dim
        self.DIST_THRESHOLD = dist_threshold
        self.bool_use_gating_and_adapter = bool_use_gating_and_adapter
        
        # 定义一个形状为 (1, 1, C) 的可学习 Token，代表“通用运动特征提取器”
        self.vel_query_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        # 使用截断正态分布初始化，利于 Transformer 早期收敛
        nn.init.trunc_normal_(self.vel_query_token, std=0.02)
        # 1. Velocity Encoder (MLP)
        self.vel_sin_embedding = VelocitySinEmbedding(embedding_dim=embedding_dim)
        
        # 2. Stacked Attention Layers (核心参数来源)
        # num_layers=3 时，这里约贡献 0.5M - 0.6M 参数
        self.layers = nn.ModuleList([
            VelMapCrossAttentionLayer(embedding_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        
        if bool_use_gating_and_adapter:
            # 4. Gating (保留，用于控制整体权重)
            self.gating_mlp = nn.Sequential(
                nn.Linear(embedding_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            nn.init.constant_(self.gating_mlp[-2].bias, 0.0)
            
            # 5. Adapter (特征融合层)
            hidden_dim = embedding_dim * 4
            self.vel_adapter = nn.Sequential(
                nn.LayerNorm(embedding_dim * 2),
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            nn.init.zeros_(self.vel_adapter[-1].weight)
            nn.init.zeros_(self.vel_adapter[-1].bias)
    
    def forward(self, base_feats, query_pos, velmap_pc):
        """
        base_feats: (B, N, C)
        query_pos: (B, N, 3) normalized
        velmap_pc: (B, N_vel, 9) [pos(3), vel(6)] normalized
        """
        velmap_pos = velmap_pc[..., :3]
        velmap_feat = velmap_pc[..., 3:9]
        
        B, N, _ = query_pos.shape
        K = min(self.K, velmap_pos.shape[1])
        
        # --- Chunked KNN Logic (保留你之前的优化) ---
        knn_vals_list = []
        knn_idx_list = []
        for b in range(B):
            q_b = query_pos[b]
            v_b = velmap_pos[b]
            dists_b = torch.cdist(q_b.unsqueeze(0), v_b.unsqueeze(0)).squeeze(0)
            vals_b, idx_b = torch.topk(dists_b, k=K, dim=-1, largest=False, sorted=True)
            knn_vals_list.append(vals_b)
            knn_idx_list.append(idx_b)
            del dists_b
        knn_vals = torch.stack(knn_vals_list, dim=0) # (B, N, K)
        knn_idx = torch.stack(knn_idx_list, dim=0)   # (B, N, K)
        
        # True means INVALID (distance too far), will be masked out in Attention
        invalid_mask = (knn_vals >= self.DIST_THRESHOLD) # (B, N, K)

        # Gather Neighbors
        velmap_all = torch.cat([velmap_pos, velmap_feat], dim=-1) # (B, N_vel, 9)
        N_vel, C_all = velmap_all.shape[1], velmap_all.shape[2]
        velmap_flat = velmap_all.view(B * N_vel, C_all)
        
        batch_offsets = torch.arange(B, device=velmap_all.device) * N_vel
        batch_offsets = batch_offsets.view(B, 1, 1)
        flat_indices = knn_idx + batch_offsets
        neighbor_data = velmap_flat[flat_indices.view(-1)].view(B, N, K, C_all)
        
        neighbor_pos = neighbor_data[..., :3] # (B, N, K, 3)
        neighbor_vel = neighbor_data[..., 3:] # (B, N, K, 6)
        
        # --- Deep Feature Extraction ---
        
        # 1. Embed Velocities (B, N, K, D)
        encoded_vel = self.vel_sin_embedding(neighbor_vel)
        
        # 2. Compute Relative Position (Geometry) (B, N, K, 3)
        rel_pos = neighbor_pos - query_pos.unsqueeze(2)
        
        # 3. Stacked Transformer Layers
        # Iteratively refine `base_feats` by attending to velocity neighbors
        current_feats = self.vel_query_token.expand(B, N, -1)
        
        for layer in self.layers:
            current_feats = layer(
                query_feats=current_feats,      # Q: Visual Context
                neighbor_vel_emb=encoded_vel,   # V: Velocity
                neighbor_rel_pos=rel_pos,       # K info: Relative Pos
                mask=invalid_mask               # Mask invalid neighbors
            )
        # `current_feats` now contains the fused visual+velocity information
        if self.bool_use_gating_and_adapter:
            # 4. Global Gating (Stabilization)
            # Use the aggregated info to compute a gate
            gate = self.gating_mlp(current_feats) # (B, N, 1)
            gated_feats = current_feats * gate
            
            # 5. Final Adapter Fusion (Residual)
            combined = torch.cat([base_feats, gated_feats], dim=-1)
            delta = self.vel_adapter(combined)
        else:
            delta = current_feats
        fused_feats = base_feats + delta
        
        return fused_feats


class ForesightDiffuserActorV6(DiffuserActor):
    """
    ForesightDiffuserActorV6 extends DiffuserActor with velmap_pc support.
    Only adds velmap_pc query and fusion functionality, inherits all other logic.
    """

    def __init__(self, *args, **kwargs):
        # Extract prob_dropout_vel_features before passing to parent
        self.prob_dropout_vel_features = kwargs.pop('prob_dropout_vel_features', 0.0)
        bool_use_gating_and_adapter = kwargs.pop('bool_use_gating_and_adapter', 1)
        super().__init__(*args, **kwargs)
        
        # Create unified velocity feature module (combines structural encoder + fusion)
        self.vel_feature_module = VelocityFeatureModuleV2(
            embedding_dim=kwargs['embedding_dim'],
            K=16,
            dist_threshold=0.20,
            bool_use_gating_and_adapter=bool_use_gating_and_adapter,
        )

    def encode_inputs(self, visible_rgb, visible_pcd, visible_mask, instruction,
                      curr_gripper, velmap_pc=None):
        """
        Encode inputs with optional velmap_pc fusion before FPS sampling.
        Uses single cdist computation in VelocityFeatureModule for both pointwise and structural features.
        
        Args:
            visible_rgb: (B, num_cameras, 3, H, W)
            visible_pcd: (B, num_cameras, 3, H, W)
            visible_mask: (B, num_cameras, 1, H, W) or None
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+)
            velmap_pc: (B, N, 3+6) or None - velmap point cloud with positions and velocities
            
        Returns:
            Tuple of (context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos, fps_pos_xyz)
            Note: fps_pos_xyz is dropped before passing to conditional_sample (6-tuple)
        """
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid, mask_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd, mask=visible_mask,
        )

        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]
        context_mask = mask_pyramid[0]

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)
        
        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )
        
        # Fuse velocity features with context features before FPS sampling
        if velmap_pc is not None:
            # Normalize velmap_pc positions to match normalized context
            velmap_pc_normalized = velmap_pc.clone()
            velmap_pc_normalized[..., :3] = self.normalize_pos(velmap_pc[..., :3])

            # Extract velocity features and fuse with context features
            context_feats = self.vel_feature_module(
                base_feats=context_feats,  # (B, N_ctx, F)
                query_pos=context,  # (B, N_ctx, 3) normalized
                velmap_pc=velmap_pc_normalized,  # (B, N_vel, 9); positions already normalized
            )  # (B, N_ctx, F)
        
        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context)

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        # Now context_feats may have been enhanced with velocity information
        fps_feats, fps_pos, fps_pos_xyz = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context),
            context,
            context_mask=context_mask,
        )
        
        return (
            context_feats, context,  # contextualized visual features (with triple-fused velocity)
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos, fps_pos_xyz  # sampled visual features
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

        # Prepare inputs (velmap_pc is already used in encode_inputs for FPS guidance)
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, visible_mask=None, instruction=instruction, curr_gripper=curr_gripper,
            velmap_pc=velmap_pc
        )
        # Remove fps_pos_xyz to match base DiffuserActor (6-tuple for conditional_sample)
        fixed_inputs = fixed_inputs[:-1]
        
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
        # Apply velocity features dropout during training (before querying velmap)
        # Sample-wise Dropout: mask entire velmap_pc batch with prob_dropout_vel_features
        if velmap_pc is not None and self.training and self.prob_dropout_vel_features > 0:
            B = velmap_pc.shape[0]
            device = velmap_pc.device
            
            # 对每个样本独立决策：要么全留(1)，要么全丢(0)
            keep_mask = (torch.rand(B, device=device) > self.prob_dropout_vel_features).float()
            
            # Expand mask to match velmap_pc shape: (B, 1, 1) for broadcasting
            keep_mask = keep_mask.view(B, 1, 1)
            
            # Mask velmap_pc: set velocity features to zero for dropped batches
            # velmap_pc: (B, N, 9) where last 6 dims are velocity features
            velmap_pc = velmap_pc.clone()  # Avoid modifying original
            velmap_pc[..., 3:9] = velmap_pc[..., 3:9] * keep_mask  # Mask velocity features (3:9)

        # Prepare inputs (velmap_pc is already used in encode_inputs for FPS guidance)
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, visible_mask=None, instruction=instruction, curr_gripper=curr_gripper,
            velmap_pc=velmap_pc
        )
        # Remove fps_pos_xyz to match base DiffuserActor (6-tuple for conditional_sample)
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
