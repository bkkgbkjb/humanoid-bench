from utils.reporter import get_reporter
import envpool
import numpy as np
from os import path
from utils.time import start_time_str
from pathlib import Path
from utils.saver import saver
import gym
import time
import jax


def make_envs(env_id, num_envs, seed):
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=num_envs,
        seed=seed,
        repeat_action_probability=0.2,
    )
    envs.num_envs = num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    return envs


class TrainEnvPool:
    def __init__(self, **config):
        self.config = config
        self.env_id = self.config["env_id"]
        self.num_envs = self.config["num_envs"]
        self.seed = self.config["seed"]

        self.env = make_envs(self.env_id, self.num_envs, self.seed)
        self.state_dim = self.env.single_observation_space.shape
        self.is_vector_env = True
        self.reset()

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    def reset(self):
        state, info = self.env.reset()
        return state, {"elapsed_step": info["elapsed_step"]}

    def step(self, actions):
        next_state, r, term, trunc, info = self.env.step(actions)
        return (
            next_state,
            r,
            term,
            trunc,
            {"elapsed_step": info["elapsed_step"]},
        )

    def reset_dead_envs(self, next_state, r, term, trunc, info):
        stop = np.logical_or(term.astype(bool), trunc.astype(bool))

        if not np.any(stop):
            return next_state, r, term, trunc, info

        dead_idx = np.argwhere(stop).flatten()
        new_states, new_info = self.env.reset(dead_idx)

        assert new_states.shape == (dead_idx.shape[0], *self.state_dim)

        next_state[dead_idx] = new_states

        rlt_info = {}
        rlt_info["elapsed_step"] = info["elapsed_step"]
        rlt_info["elapsed_step"][dead_idx] = new_info["elapsed_step"]
        assert np.all(rlt_info["elapsed_step"][dead_idx] == 0)

        return next_state, r, term, trunc, rlt_info


class EvalEnvPool:
    def __init__(self, **config):
        self.config = config
        self.env_id = self.config["env_id"]
        self.num_envs = self.config["num_envs"]
        self.seed = self.config["seed"]
        self.env = make_envs(
            self.env_id,
            self.config["num_envs"],
            seed=config["seed"],
        )
        self.state_dim = self.env.single_observation_space.shape
        self.is_vector_env = True
        self.reset()

    @property
    def nums(self):
        return self.num_envs

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    def reset(self):
        state, info = self.env.reset()
        self.env_is_alive = np.ones((self.nums,), dtype=np.float32)
        return state, {"elapsed_step": info["elapsed_step"]}

    def step(self, actions):
        next_state, r, term, trunc, info = self.env.step(actions)
        return (
            next_state,
            r * self.env_is_alive,
            term,
            trunc,
            {"elapsed_step": info["elapsed_step"]},
        )

    def mark_dead_envs(self, next_state, r, term, trunc, info):
        stop = np.logical_or(term.astype(bool), trunc.astype(bool))

        if not np.any(stop):
            return

        dead_idx = np.argwhere(stop).flatten()
        self.env_is_alive[dead_idx] = 0.0

    def is_all_dead(self):
        return np.all(self.env_is_alive == 0.0)


def eval_in_envs(
    key, get_action, eval_envs: EvalEnvPool, counter: int, name: str, to_save
):
    rets = np.zeros((eval_envs.nums,), dtype=np.float32)
    s, i = eval_envs.reset()

    save_path = Path(path.abspath(path.join("models", start_time_str + "_" + name)))
    save_path.mkdir(parents=True, exist_ok=True)

    while not eval_envs.is_all_dead():

        key, act_key = jax.random.split(key)
        acts = get_action(s, act_key)

        assert acts.shape == (eval_envs.nums,)

        s, r, term, trunc, i = eval_envs.step(acts)

        assert rets.shape == r.shape
        rets += r

        eval_envs.mark_dead_envs(s, r, term, trunc, i)

    get_reporter().add_distributions(
        {"returns": (rets, counter)},
        "eval",
    )
    get_reporter().add_scalars(
        {
            "Return Mean": (np.mean(rets), counter),
            "Return Std": (np.std(rets), counter),
        },
        "eval",
    )
    saver.save(save_path / str(counter), to_save)
    return rets, dict(return_mean=np.mean(rets), return_std=np.std(rets))


def play(env, policy, seed: int, key):

    env.seed(seed)
    obs, _ = env.reset()

    stop = False
    rew = 0
    while not stop:
        key, act_key = jax.random.split(key)
        a = policy(obs, act_key)
        obs, _rew, term, trunc, _ = env.step(a)

        stop = term or trunc
        rew += _rew

        env.render()
