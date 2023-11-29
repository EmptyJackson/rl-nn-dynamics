import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


class ActorCritic(nn.Module):
    num_actions: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        logged_activations = {}
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        else:
            raise ValueError("Activation not recognised")
        actor_mean = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_0",
        )(x)
        actor_mean = activation(actor_mean)
        logged_activations["actor_0"] = actor_mean
        actor_mean = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_1",
        )(actor_mean)
        actor_mean = activation(actor_mean)
        logged_activations["actor_1"] = actor_mean
        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_0",
        )(x)
        critic = activation(critic)
        logged_activations["critic_0"] = critic
        critic = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_1",
        )(critic)
        critic = activation(critic)
        logged_activations["critic_1"] = critic
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_out"
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1), logged_activations

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape),)


class ActorCriticContinuous(nn.Module):
    num_actions: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param(
            "log_std", nn.initializers.zeros, (self.num_actions,)
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape),)
