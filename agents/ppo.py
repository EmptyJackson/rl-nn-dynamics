import jax
import jax.numpy as jnp

from analysis.activations import threshold_grad_second_moment
from agents.common import construct_minibatches, calculate_gae


def compute_tree_norm(tree):
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree))
    )


def make_train_step(args, network):
    def _update_step(train_state, aux_train_states, traj_batch, last_obs, rng):
        def _update_epoch(update_state, _):
            train_state, traj_batch, advantages, targets, rng = update_state

            def _update_minbatch(train_state, batch_info):
                # --- Update agent ---
                traj_batch, advantages, targets = batch_info

                # We want to look at the second moment of the
                # gradient in this project. Usually we use the
                # property that the gradient of the mean is the
                # mean of the gradient (because differentiation is linear)
                # but now that we are squaring the gradient, we need to
                # compute the loss and differentiate *per-sample*
                # then we can square and average all the gradient estimates.
                def _loss_fn(params, traj_batch, gae, targets):
                    pi, value, _ = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-args.ppo_clip_eps, args.ppo_clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # Actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.ppo_clip_eps,
                            1.0 + args.ppo_clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        actor_loss
                        + args.value_loss_coef * value_loss
                        - args.entropy_coef * entropy
                    )
                    return total_loss, (value_loss, actor_loss, entropy)

                grad_fn = jax.vmap(
                    jax.value_and_grad(_loss_fn, has_aux=True), in_axes=(None, 0, 0, 0)
                )

                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
                (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                metrics = {
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "total_loss": total_loss,
                    "entropy": entropy,
                }
                if args.log_gsm:
                    grad_second_moment = jax.tree_map(jnp.square, grads)
                    threshold_gsm = threshold_grad_second_moment(
                        grad_second_moment,
                        train_state.params,
                        zeta_abs=args.zeta_abs,
                        zeta_rel=args.zeta_rel,
                    )
                    grad_second_moment = jax.tree_map(
                        lambda x: x.mean(axis=0), grad_second_moment
                    )
                    metrics = {
                        **metrics,
                        "grad_second_moment": grad_second_moment,
                        "threshold_grad_second_moment": threshold_gsm,
                    }
                grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
                metrics["grad_norm"] = compute_tree_norm(grads)
                # compute the adam update size
                metrics["update_norm"] = compute_tree_norm(
                    train_state.tx.update(
                        grads, train_state.opt_state, train_state.params
                    )[0]
                )
                # computemetrics the similarity between the gradient and the momentum estimates
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            # --- Iterate over minibatches ---
            batch = (traj_batch, advantages, targets)
            rng, _rng = jax.random.split(rng)
            minibatches = construct_minibatches(_rng, args, batch)
            train_state, metrics = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, metrics

        # --- Calculate advantage ---
        _, last_val, _ = network.apply(train_state.params, last_obs)
        advantages, targets = jax.vmap(calculate_gae, in_axes=(None, 0, 0))(
            args, traj_batch, last_val
        )

        # --- Iterate over epochs ---
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, args.ppo_num_epochs
        )
        train_state = update_state[0]

        # --- Reset optimizer state ---
        opt_state = train_state.opt_state
        assert args.ppo_reset_on_batch in {"count", "all", "none"}
        if args.ppo_reset_on_batch == "count":
            opt_state = tuple(
                [
                    opt_state[0],
                    tuple([opt_state[1][0]._replace(count=0), opt_state[1][1]]),
                ]
            )
        elif args.ppo_reset_on_batch == "all":
            opt_state = jax.tree_map(jnp.zeros_like, opt_state)
        train_state = train_state.replace(opt_state=opt_state)

        # --- Return train state and metrics ---
        info = traj_batch.info
        if args.log_gsm:
            grad_second_moment = metrics["grad_second_moment"]
            metrics = jax.tree_map(lambda x: x.mean(), metrics)
            metrics["grad_second_moment"] = jax.tree_map(
                lambda x: jnp.histogram(jnp.log(x + args.zeta_abs), bins=64),
                grad_second_moment,
            )
        # No auxiliary networks
        return train_state, aux_train_states, metrics, info

    return _update_step
