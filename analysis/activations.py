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
            # sample_dormant = (activations / layer_mean) < tau
            # return jnp.all(sample_dormant)
            # V2 - Mean dormant
            return (jnp.mean(activations) / layer_mean) < tau

        neuron_dormant = jax.vmap(_batch_dormant, in_axes=-1, out_axes=-1)(activations)
        return jnp.mean(neuron_dormant)

    return jax.tree_map(_layer_dormancy, activations)
