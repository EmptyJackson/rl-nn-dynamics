import optax
from functools import partial
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
    elif args.optimizer == "sgd":
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.sgd(learning_rate=lr_schedule),
        )
