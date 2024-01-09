import optax
from functools import partial


def _linear_schedule(count, args):
    if args.agent == "ppo":
        frac = (
            1.0
            - (count // (args.num_minibatches * args.ppo_num_epochs))
            / args.num_train_iters
        )
    else:
        frac = 1.0 - (count / args.num_train_iters)
    return args.lr * frac


def create_optimizer(args):
    if args.optimizer == "adam":
        if args.anneal_lr:
            return optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(
                    learning_rate=partial(_linear_schedule, args=args),
                    b1=args.b1,
                    b2=args.b2,
                    eps=1e-5,
                ),
            )
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        )
    elif args.optimizer == "sgd":
        if args.anneal_lr:
            return optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.sgd(learning_rate=partial(_linear_schedule, args=args)),
            )
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), optax.sgd(args.lr)
        )
