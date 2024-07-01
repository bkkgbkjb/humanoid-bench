import jax
import jax.numpy as jnp
import flax.struct as fstruct
import jax.random as jrnd
from typing import TypeVar, Generic, Dict, Any
import flax.core as fcore

T = TypeVar("T")


@fstruct.dataclass
class GenericEnvState(Generic[T]):
    sys: T

    obs: jnp.ndarray
    acts: jnp.ndarray
    rwds: jnp.ndarray
    terms: jnp.ndarray
    truncs: jnp.ndarray

    infos: fcore.FrozenDict[str, Any]


@jax.jit
def to_grid_index(idx: jnp.ndarray):
    _grid = jnp.ogrid[tuple(map(slice, idx.shape))]
    _grid.append(idx)
    return tuple(_grid)


def extract_in_out_trees(raw_env_tree):
    _in_tree = jax.tree_map(
        lambda x: (
            (None if jnp.isnan(x[0]) else int(x[0].item()))
            if x.ndim >= 1
            else (None if jnp.isnan(x) else int(x.item()))
        ),
        raw_env_tree,
    )
    _out_tree = jax.tree_map(
        lambda x: (
            (None if jnp.isnan(x[1]) else int(x[1].item(0)))
            if x.ndim >= 1
            else (None if jnp.isnan(x) else int(x.item()))
        ),
        raw_env_tree,
    )
    return _in_tree, _out_tree


def build_merge_trees(in_tree):
    return jax.vmap(
        lambda reset_tree, oldtree, newtree: jax.tree_map(
            lambda old, new, need_resets: jnp.where(
                need_resets,
                new,
                old,
            ),
            oldtree,
            newtree,
            reset_tree,
        ),
        in_axes=(in_tree, in_tree, in_tree),
        out_axes=in_tree,
    )
