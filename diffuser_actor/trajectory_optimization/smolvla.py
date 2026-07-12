from copy import deepcopy
from transformers import AutoTokenizer
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

def build_smolvla_model(args):
    cfg = SmolVLAConfig(repo_id="reedee123/record-train_1026time1")

    visual_feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    state_feature = PolicyFeature(type=FeatureType.STATE, shape=(24,))
    action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(8,))

    cfg.input_features['observation.images.front'] = visual_feature
    cfg.input_features['observation.images.wrist'] = visual_feature
    cfg.input_features['observation.images.nerf'] = visual_feature
    cfg.input_features['observation.images.nerf2'] = visual_feature
    cfg.input_features['observation.state'] = state_feature
    cfg.output_features['action'] = action_feature
    cfg.chunk_size = 1
    cfg.n_action_steps = 1

    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.repo_id)
    wrist_feature_meta = deepcopy(ds_meta.features['observation.images.wrist'])
    ds_meta.features['observation.images.nerf'] = wrist_feature_meta
    ds_meta.features['observation.images.nerf2'] = deepcopy(wrist_feature_meta)

    _model = make_policy(
        cfg=cfg,
        ds_meta=ds_meta,
        rename_map={},
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    _model.dev_dict = {
        'tokenizer': tokenizer,
        'tokenizer_padding': cfg.pad_language_to,
        'tokenizer_max_length': cfg.tokenizer_max_length,
        'task_prompt': 'Reach Single Moving Target On The Table',
    }
    return _model


def convert_batch_to_lerobot_format(model, sample, curr_gripper):
    B = sample['rgbs'].shape[0]
    tokenizer = model.module.dev_dict['tokenizer']
    padding_strategy = model.module.dev_dict['tokenizer_padding']
    max_length = model.module.dev_dict['tokenizer_max_length']
    tasks = [itm if itm.endswith('\n') else f"{itm}\n" for itm in sample['instr_str']]
    encoded = tokenizer(
        tasks,
        padding=padding_strategy,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    model_device = next(model.parameters()).device
    language_tokens = encoded['input_ids'].to(model_device)
    language_attention = encoded['attention_mask'].to(model_device).bool()
    batch = {
        'observation.images.front': sample['rgbs'][:,0].to(model_device),
        'observation.images.wrist': sample['rgbs'][:,1].to(model_device),
        'observation.images.nerf': sample['rgbs'][:,2].to(model_device),
        'observation.images.nerf2': sample['rgbs'][:,3].to(model_device),
        'observation.state': curr_gripper.reshape(B, -1).to(model_device),
        'observation.language.tokens': language_tokens.to(model_device),
        'observation.language.attention_mask': language_attention.to(model_device),
        'action': sample['trajectory'].to(model_device),
    }
    return batch
