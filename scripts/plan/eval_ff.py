"""Script to evaluate a feedforward policy on a dataset of episodes."""

import os


os.environ['MUJOCO_GL'] = 'egl'


import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


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

    artifacts = [a for a in run.logged_artifacts() if a.type == 'world-model']
    if not artifacts:
        raise ValueError(f"No 'world-model' artifact logged by run {wb.run}")

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
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world, image_shape=(224, 224), render_mode='rgb_array'
    )

    # create the transform
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
    wandb_run = cfg.get('wandb', {}).get('run', None)
    train_run = None  # wandb run to log the success rate back to

    if wandb_run:
        # Load the policy from the wandb run's reference artifact.
        ckpt_path, train_run = resolve_wandb_policy(cfg)
        model = swm.policy.AutoActionableModel(str(ckpt_path))
        model = model.to('cuda').eval()
        model.requires_grad_(False)
        policy = swm.policy.FeedForwardPolicy(
            model=model, process=process, transform=transform
        )
        results_path = ckpt_path.parent / 'eval'
    elif policy != 'random':
        model = swm.policy.AutoActionableModel(cfg.policy)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)

        policy = swm.policy.FeedForwardPolicy(
            model=model, process=process, transform=transform
        )
        results_path = Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            cfg.policy,
            'eval',
        )
    else:
        policy = swm.policy.RandomPolicy()
        results_path = Path(__file__).parent

    print(f"result path: {results_path}")

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
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

    # -- write the eval success rate into the policy's training wandb run
    if train_run is not None:
        train_run.summary['eval/success_rate'] = metrics['success_rate']
        train_run.summary.update()
        print(
            f"Logged eval/success_rate={metrics['success_rate']} "
            f'to wandb run {cfg.wandb.run}'
        )


if __name__ == '__main__':
    run()
