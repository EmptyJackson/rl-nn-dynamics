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


def threshold_grad_second_moment(grad_second_moment, zeta):
    def _threshold_grad_second_moment(grad_second_moment):
        grad_second_moment = grad_second_moment.reshape(grad_second_moment.shape[0], -1)

        def _batch_threshold_grad_second_moment(grad_second_moment):
            return jnp.mean(grad_second_moment) <= zeta

        threshold_gsm = jax.vmap(
            _batch_threshold_grad_second_moment, in_axes=-1, out_axes=-1
        )(grad_second_moment)
        return jnp.mean(threshold_gsm)

    return jax.tree_map(_threshold_grad_second_moment, grad_second_moment)


if __name__ == "__main__":
    gsm = {
        "params": {
            "actor_0": {
                "kernel": jnp.ones((16, 32, 64)),
                "bias": 2 + jnp.zeros((16, 32)),
            }
        }
    }
    print(threshold_grad_second_moment(gsm, 1))
    gsm["params"]["actor_0"]["bias"] = (
        gsm["params"]["actor_0"]["bias"].at[0:16:2, 0].set(0)
    )
    print(threshold_grad_second_moment(gsm, 1))
