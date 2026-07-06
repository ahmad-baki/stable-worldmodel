"""Script to evaluate a feedforward policy on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import shutil
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import pickle

import stable_worldmodel as swm
import numpy as np, gymnasium as gym

import sys
# scripts/ lives at the repo root and is not part of the installed package,
# so add the repo root (parent of scripts/) to the path before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.prejepa import get_encoder
# from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
ALGOS = {'a2c': A2C, 'ddpg': DDPG, 'dqn': DQN, 'ppo': PPO, 'sac': SAC, 'td3': TD3}

class SB3Policy(swm.policy.BasePolicy):
    def __init__(self, algo, ckpt_path, vecnorm_path=None, **kw):
        super().__init__(**kw)
        self.type = algo
        self.model = ALGOS[algo].load(ckpt_path, device='cpu')
        # Load only the saved normalization stats for inference; VecNormalize.load
        # would call set_venv(None) and crash, so unpickle directly instead.
        if vecnorm_path:
            with open(vecnorm_path, 'rb') as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False
        else:
            self.vecnorm = None
        space = self.model.observation_space          # read keys from the model
        self.obs_keys = list(space.spaces) if isinstance(space, gym.spaces.Dict) else None

    def get_action(self, infos, **kw):
        if self.obs_keys is not None:                 # Dict obs
            obs = {k: np.asarray(infos[k]).squeeze(1) for k in self.obs_keys}
        else:                                         # flat Box obs
            obs = np.asarray(infos['observation']).squeeze(1)  # wrapper's name for non-dict obs
        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)
        actions, _ = self.model.predict(obs, deterministic=True)
        return actions


def img_transform():
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
        ]
    )
    return transform


class EncoderTransform:
    """Chain ImageNet preprocessing with a frozen backbone forward pass.

    Applies ``img_transform`` to a single image, batches it, runs the encoder,
    and returns the per-sample embedding (batch dim stripped) on CPU.
    """

    def __init__(self, backbone, interp_pos_enc=True, device='cuda'):
        self.transform = img_transform()
        self.backbone = backbone.to(device).eval().requires_grad_(False)
        self.interp_pos_enc = interp_pos_enc
        self.device = device

    @torch.no_grad()
    def __call__(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        try:
            out = self.backbone(x, interpolate_pos_encoding=self.interp_pos_enc)
        except TypeError:  # e.g. resnet classifier doesn't take the kwarg
            out = self.backbone(x)
        emb = getattr(out, 'last_hidden_state', None)
        if emb is None:
            emb = getattr(out, 'pooler_output', None)
        if emb is None:
            emb = out.logits        # (1, P, pixel_dim)
            emb = emb.mean(dim=1)   # (1, pixel_dim)
        return emb.squeeze(0).cpu() # (pixel_dim,)


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
    )
    return dataset


def resolve_wandb_policy(cfg):
    """Resolve a checkpoint path from a wandb run's reference artifact.

    Reads ``cfg.wandb`` (``entity``, ``project``, ``run`` and optional
    ``alias``), finds the ``world-model`` artifact the training run logged, and
    follows its ``file://`` reference to the on-disk ``*_object.ckpt``.

    Returns ``(ckpt_path, run)`` where ``run`` is the wandb public-API run so the
    caller can write the eval success rate back into the training run.
    """
    import wandb

    wb = cfg.wandb
    api = wandb.Api()
    run = api.run(f'{wb.entity}/{wb.project}/{wb.run}')

    artifacts = [a for a in run.logged_artifacts() if a.type == 'model']
    if not artifacts:
        raise ValueError(f"No 'model' artifact logged by run {wb.run}")

    alias = wb.get('alias', 'best')
    artifact = next(
        (a for a in artifacts if alias in a.aliases), artifacts[-1]
    )
    print(f'Using artifact {artifact.name} (aliases: {artifact.aliases})')

    ckpt_path = None
    for entry in artifact.manifest.entries.values():
        if entry.ref and entry.ref.startswith('file://'):
            ckpt_path = entry.ref[len('file://') :]
            break
    if ckpt_path is None:
        raise ValueError(
            f'No file:// reference found in artifact {artifact.name}'
        )

    return Path(ckpt_path), run


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 4 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world, image_shape=(224, 224), render_mode='rgb_array'
    )

    # create the transform
    if cfg.get('backbone'):
        backbone, embed_dim, num_patches, interp_pos_enc = get_encoder(cfg)
        encoder_transform = EncoderTransform(backbone, interp_pos_enc)
        transform = {
            'pixels': encoder_transform,
            'goal': encoder_transform,
        }
    else:
        transform = {
            'pixels': img_transform(),
            'goal': img_transform(),
        }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    # create the processing
    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset.get_col_data('action'))

    process = {'action': action_process}

    if 'proprio' in dataset.column_names:
        proprio_process = preprocessing.StandardScaler()
        proprio_process.fit(dataset.get_col_data('proprio'))
        process['proprio'] = proprio_process
        process['goal_proprio'] = proprio_process

    # -- run evaluation
    policy = cfg.get('policy', 'random')
    rl_algo = cfg.get('rl_algo', None)
    wandb_run = cfg.get('wandb', {}).get('run', None)
    train_run = None  # wandb run to log the success rate back to


    
    if wandb_run:
        # Load the policy from the wandb run's reference artifact.
        ckpt_path, train_run = resolve_wandb_policy(cfg)
        results_path = ckpt_path.parent / 'eval'
    else:
        ckpt_path = Path(cfg.policy)
        if ckpt_path.is_file():
            sub_folder = str(ckpt_path.parent / 'eval')
        else:
            sub_folder = str(ckpt_path / 'eval')

        results_path = Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            sub_folder
        )

    if policy == 'random' and not wandb_run:
        policy = swm.policy.RandomPolicy()
        results_path = Path(HydraConfig.get().runtime.output_dir)
    elif policy == 'noop' and not wandb_run:
        policy = swm.policy.NoOpPolicy()
        results_path = Path(HydraConfig.get().runtime.output_dir)
    elif rl_algo and rl_algo in ALGOS:
        # find vecnorm path
        vecnorm_path = get_vecnorm_path(cfg, ckpt_path)
        policy = SB3Policy(rl_algo, ckpt_path, vecnorm_path)
    else:
        model = swm.policy.AutoActionableModel(str(ckpt_path))
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)

        policy = swm.policy.FeedForwardPolicy(
            model=model, process=process, transform=transform
        )
    world.set_policy(policy)

    print(f"result path: {results_path}")

    if cfg.eval.use_dataset:
        # sample starts whose remaining expert horizon is at most capped by
        # max_goal_offset_steps. Starts near the end of an episode therefore
        # evaluate against the final available dataset state instead of being
        # discarded.
        episode_len = get_episodes_length(dataset, ep_indices)
        max_goal_offset_steps = cfg.eval.get(
            'max_goal_offset_steps', cfg.eval.goal_offset_steps
        )
        max_start_idx = episode_len - 2
        max_start_idx_dict = {
            ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
        }
        episode_len_dict = {
            ep_id: episode_len[i] for i, ep_id in enumerate(ep_indices)
        }
        # Map each dataset row’s episode_idx to its max_start_idx
        col_name = (
            'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
        )
        row_episode_idx = dataset.get_col_data(col_name)
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in row_episode_idx]
        )

        # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
        valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), 'valid starting points found for evaluation.')

        if len(valid_indices) < cfg.eval.num_eval:
            raise ValueError(
                'Not enough episodes with sufficient length for evaluation.'
            )

        g = np.random.default_rng(cfg.seed)
        random_episode_indices = g.choice(
            len(valid_indices), size=cfg.eval.num_eval, replace=False
        )

        # sort increasingly to avoid issues with HDF5Dataset indexing
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

        print(random_episode_indices)

        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']
        eval_goal_offsets = np.array(
            [
                min(
                    max_goal_offset_steps,
                    episode_len_dict[ep_id] - start_idx - 1,
                )
                for ep_id, start_idx in zip(eval_episodes, eval_start_idx)
            ]
        )

        start_time = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=eval_goal_offsets,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ), # type: ignore
            video=results_path,
        )
        metrics['goal_offsets'] = eval_goal_offsets
        end_time = time.time()
    else:
        start_time = time.time()
        metrics = world.evaluate(
            # dataset=dataset,
            episodes=cfg.eval.num_eval,
            seed=cfg.seed,
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ), # type: ignore
            video=results_path,
        )
        end_time = time.time()


    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')

    # -- write the eval metrics into the policy's training wandb run's summary.
    # Poking run.summary through the public API is silently overwritten whenever
    # the training process re-syncs its own summary (e.g. Lightning's WandbLogger
    # at wandb.finish()), so the eval value disappears. Instead we resume the run
    # and set the summary through the SDK: resuming restores the existing summary,
    # merges the eval keys in, and re-uploads them on finish. We only touch the
    # summary (no wandb.log) so these show up as summary values, not as history
    # time-series charts. This is only safe once training has finished; if the run
    # is still live, both writers race and the eval value is clobbered.
    if train_run is not None:
        import wandb

        successes = np.asarray(metrics['episode_successes'])
        eval_summary = {
            'eval/success_rate': float(metrics['success_rate']),
            'eval/num_successes': float(successes.sum()),
            'eval/num_episodes': int(successes.size),
        }

        if train_run.state == 'running':
            # The live training process still owns the summary and will overwrite
            # anything we write when it finishes. Do a best-effort poke and warn.
            print(
                f'WARNING: wandb run {cfg.wandb.run} is still running; the eval '
                'summary will likely be overwritten when the training job '
                'finishes. Re-run eval after training completes for a durable '
                'summary.'
            )
            for key, value in eval_summary.items():
                train_run.summary[key] = value
            train_run.summary.update()
        else:
            eval_run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                id=cfg.wandb.run,
                resume='must',
            )
            eval_run.summary.update(eval_summary)
            eval_run.finish()

        print(f'Logged {eval_summary} to wandb run {cfg.wandb.run}')

def env_dir_name(env_id: str) -> str:
    """
    matches rl-zoo exactly (slash + colon)
    """
    name = env_id.replace('/', '-')          # same rule rl-zoo uses
    return name.split(':')[1] if ':' in name else name

def get_vecnorm_path(cfg, ckpt_path):
    env_dir = env_dir_name(cfg.world.env_name)          
    vecnorm_path = ckpt_path.parent / env_dir / 'vecnormalize.pkl'
    vecnorm_path = vecnorm_path if vecnorm_path.exists() else None
    return vecnorm_path


if __name__ == '__main__':
    run()
