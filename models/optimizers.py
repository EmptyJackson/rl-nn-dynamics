import optax
from functools import partial
import jax
import jax.numpy as jnp


def make_inner_batch_scaling_fn(args):
    total_updates = args.num_minibatches * args.ppo_num_epochs

    def up_then_const_fn(step):
        up_steps = total_updates / 2
        return jnp.minimum(step, up_steps) / up_steps

    def down_then_const_fn(step):
        down_steps = total_updates / 2
        floor_scale = 0.2
        return 1.0 - (jnp.minimum(step, down_steps) / down_steps) * (1 - floor_scale)

    def zero_then_const_fn(step):
        zero_steps = 12
        return jnp.where(step < zero_steps, 0.0, 1.0)

    def const_fn(_):
        return 1.0

    if args.inner_batch_scaling == "up_then_const":
        return up_then_const_fn
    elif args.inner_batch_scaling == "down_then_const":
        return down_then_const_fn
    elif args.inner_batch_scaling == "zero_then_const":
        return zero_then_const_fn
    elif args.inner_batch_scaling == "const":
        return const_fn
    else:
        raise ValueError("Not a valid type of inner batch scaling")


def _linear_schedule(count, args):
    if args.agent == "ppo":
        frac = (
            1.0
            - (count // (args.num_minibatches * args.ppo_num_epochs))
            / args.num_train_iters
        )
    else:
        frac = 1.0 - (count / args.num_train_iters)
    return frac


def create_lr_schedule(args):
    # have only implemented PPO
    assert args.agent == "ppo"
    inner_batch_scaling_fn = make_inner_batch_scaling_fn(args)
    if args.anneal_lr:
        global_scaling_fn = partial(_linear_schedule, args=args)
    else:
        global_scaling_fn = lambda _: 1.0

    def lr_schedule(count):
        inner_batch_step = count % (args.num_minibatches * args.ppo_num_epochs)
        inner_batch_frac = inner_batch_scaling_fn(inner_batch_step)
        global_frac = global_scaling_fn(count)
        return global_frac * inner_batch_frac * args.lr

    return lr_schedule


def create_optimizer(args):
    lr_schedule = create_lr_schedule(args)
    if args.optimizer == "adam":
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(
                learning_rate=lr_schedule,
                b1=args.b1,
                b2=args.b2,
                eps=1e-5,
            ),
        )
    elif args.optimizer == "dadam":
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            dadam(
                learning_rate=lr_schedule,
                b1=args.b1,
                b2=args.b2,
                eps=1e-5,
            ),
        )
    elif args.optimizer == "sgd":
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.sgd(learning_rate=lr_schedule),
        )


# ---
# OPTAX EXTENSIONS
# ---

import chex
from typing import Any, Optional, NamedTuple
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import numerics
from optax._src import utils


class ScaleByDadamState(NamedTuple):
    """State for the Dadam algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_fast: base.Updates


def scale_by_dadam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    References:
        [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A `GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_fast = jax.tree_util.tree_map(jnp.zeros_like, params)  # Fast second moment
        return ScaleByDadamState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_fast=nu_fast
        )

    def update_fn(updates, state, params=None):
        del params
        mu = transform.update_moment(updates, state.mu, b1, 1)
        nu = transform.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        nu_fast = transform.update_moment_per_elem_norm(
            updates, state.nu_fast, b2**rho, 2
        )
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = transform.bias_correction(mu, b1, count_inc)
        nu_hat = transform.bias_correction(nu, b2, count_inc)
        nu_fast_hat = transform.bias_correction(nu_fast, b1, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v, v_f: m
            / (jnp.maximum(jnp.sqrt(v + eps_root), jnp.sqrt(v_f + eps_root)) + eps),
            mu_hat,
            nu_hat,
            nu_fast_hat,
        )
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByDadamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def dadam(
    learning_rate: alias.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    r"""The classic Adam optimizer.

  Adam is an SGD variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectievly. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::
    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
    return combine.chain(
        scale_by_dadam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        alias._scale_by_learning_rate(learning_rate),
    )
