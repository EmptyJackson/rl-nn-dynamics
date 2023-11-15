import jax
import jax.numpy as jnp


INTER_LAYERS = ["actor_0", "actor_1", "critic_0", "critic_1"]


def dormancy_rate(activations, tau):
    def _layer_dormancy(activations):
        """Proportion of dormant layer neurons in an activation batch"""
        activations = jnp.abs(activations)
        layer_mean = activations.mean()

        def _batch_dormant(activations):
            # V1 - All samples dormant
            # sample_dormant = (activations / layer_mean) <= tau
            # return jnp.all(sample_dormant)
            # V2 - Mean dormant
            return (jnp.mean(activations) / layer_mean) <= tau

        neuron_dormant = jax.vmap(_batch_dormant, in_axes=-1, out_axes=-1)(activations)
        return jnp.mean(neuron_dormant)

    return jax.tree_map(_layer_dormancy, activations)


def threshold_grad_second_moment(
    params, grad_second_moment, zeta_abs=1e-14, zeta_rel=1e-6
):
    def _threshold_grad_second_moment(grad_second_moment, params):
        grad_second_moment = grad_second_moment.reshape(grad_second_moment.shape[0], -1)
        params = params.reshape(-1)

        def _batch_threshold_grad_second_moment(grad_second_moment, params):
            return (jnp.mean(grad_second_moment) <= zeta_abs) | (
                (jnp.sqrt(jnp.mean(grad_second_moment)) / params) <= zeta_rel
            )

        threshold_gsm = jax.vmap(
            _batch_threshold_grad_second_moment, in_axes=(-1, -1), out_axes=-1
        )(grad_second_moment, params)
        return jnp.mean(threshold_gsm)

    return jax.tree_map(_threshold_grad_second_moment, grad_second_moment, params)


if __name__ == "__main__":
    gsm = {"actor_0": {"kernel": 1e-5 * jnp.ones((3, 2, 4)), "bias": jnp.ones((3, 4))}}
    params = {
        "actor_0": {
            "kernel": 1e-6 * jnp.ones((3, 2, 4)),
            "bias": 1000 * jnp.ones((3, 4)),
        }
    }

    print(threshold_grad_second_moment(params, gsm, zeta_abs=1e-4, zeta_rel=1e-9))

    print(threshold_grad_second_moment(params, gsm, zeta_abs=1e-10, zeta_rel=1e-2))
