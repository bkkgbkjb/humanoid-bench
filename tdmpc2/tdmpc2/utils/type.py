import flax.struct as fstruct
import jax.numpy as jnp


@fstruct.dataclass
class EnvStep:
    obs: jnp.ndarray
    acts: jnp.ndarray
    rwds: jnp.ndarray
    terms: jnp.ndarray
    truncs: jnp.ndarray
    next_obs: jnp.ndarray
