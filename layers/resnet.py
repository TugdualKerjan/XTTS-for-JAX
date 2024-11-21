#!/usr/bin/env python
# coding: utf-8



import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import typing as tp




import equinox as eqx
import jax


class SELayer(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, channel, reduction=8, key=None):
        key1, key2 = jax.random.split(key, 2)
        self.fc1 = eqx.nn.Linear(channel, channel // reduction, use_bias=True, key=key1)
        self.fc2 = eqx.nn.Linear(channel // reduction, channel, use_bias=True, key=key2)

    def __call__(self, x):
        y = eqx.nn.AdaptiveAvgPool2d(1)(x)
        y = jax.numpy.squeeze(y)
        y = self.fc1(y)
        y = jax.nn.relu(y)
        y = self.fc2(y)
        y = jax.nn.sigmoid(y)
        y = jax.numpy.expand_dims(y, (1, 2))

        return x * y




import jax
import equinox as eqx


class SEBasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    se: SELayer
    downsample: None

    def __init__(self, channels_in, channels_out, stride=1, downsample=None, key=None):
        key1, key3, key5 = jax.random.split(key, 3)

        # TODO Understand why bias isn't added.
        # TODO Do we want to have a state or simply do GroupNorm instead ?

        self.conv1 = eqx.nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            use_bias=False,
            key=key1,
        )
        self.bn1 = eqx.nn.BatchNorm(channels_out, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(
            channels_out,
            channels_out,
            kernel_size=(3, 3),
            padding=1,
            use_bias=False,
            key=key3,
        )
        self.bn2 = eqx.nn.BatchNorm(channels_out, axis_name="batch")

        self.se = SELayer(channels_out, key=key5)
        self.downsample = downsample

    def __call__(self, x, state):
        residual = x

        y = self.conv1(x)

        y = jax.nn.relu(y)
        y, state = self.bn1(y, state)

        y = self.conv2(y)
        y, state = self.bn2(y, state)

        y = self.se(y)

        if self.downsample is not None:
            residual, state = self.downsample(x, state)

        y = y + residual  # Residual
        y = jax.nn.relu(y)

        return y, state




import jax
import equinox as eqx
import jax.tools


class ResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    batch_norm: eqx.nn.BatchNorm

    layer1: list
    layer2: list
    layer3: list
    layer4: list

    instance_norm: eqx.nn.GroupNorm

    attention_conv1: eqx.nn.Conv1d
    attention_batch_norm: eqx.nn.BatchNorm
    attention_conv2: eqx.nn.Conv1d

    log_input: bool

    fc: eqx.nn.Linear

    def create_layer(self, channels_in, channels_out, layers, stride=1, key=None):

        downsample = None
        if stride != 1 or channels_in != channels_out:
            key, grab = jax.random.split(key, 2)
            downsample = eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=1,
                        stride=stride,
                        use_bias=False,
                        key=grab,
                    ),
                    eqx.nn.BatchNorm(channels_out, axis_name="batch"),
                ]
            )

        stack_of_blocks = []
        # print(key)
        key, grab = jax.random.split(key, 2)

        stack_of_blocks.append(
            SEBasicBlock(channels_in, channels_out, stride, downsample, key=grab)
        )
        for _ in range(1, layers):

            key, grab = jax.random.split(key, 2)
            stack_of_blocks.append(
                SEBasicBlock(channels_out, channels_out, stride=1, key=grab)
            )

        return stack_of_blocks

    def __init__(
        self,
        input_dims,
        proj_dim,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        log_input=False,
        key=None,
    ):
        # he_init = jax.nn.initializers.variance_scaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
        self.log_input = log_input
        key, grab = jax.random.split(key, 2)
        # TODO self.conv1 = eqx.nn.Conv2d(1, num_filters[0], key=grab, weight_init=he_init)
        self.conv1 = eqx.nn.Conv2d(
            1, num_filters[0], kernel_size=3, padding=1, key=grab
        )
        self.batch_norm = eqx.nn.BatchNorm(
            num_filters[0], axis_name="batch", momentum=0.9
        )

        key, key1, key2, key3, key4 = jax.random.split(key, 5)
        self.layer1 = self.create_layer(
            num_filters[0], num_filters[0], layers[0], key=key2
        )
        self.layer2 = self.create_layer(
            num_filters[0], num_filters[1], layers[1], stride=(2, 2), key=key3
        )
        self.layer3 = self.create_layer(
            num_filters[1], num_filters[2], layers[2], stride=(2, 2), key=key4
        )
        self.layer4 = self.create_layer(
            num_filters[2], num_filters[3], layers[3], stride=(2, 2), key=key1
        )

        # Instance norm seems to be a specific example of groupnorm.
        self.instance_norm = eqx.nn.GroupNorm(
            input_dims, input_dims, channelwise_affine=False
        )

        key, key1, key2 = jax.random.split(key, 3)

        # Basically a FFN but without needing to deal with the channel dimensions.
        # doesn't really explain the lowering of dimensions in the middle though...
        current_channel_size = int(num_filters[3] * input_dims / 8)
        self.attention_conv1 = eqx.nn.Conv1d(
            current_channel_size, 128, kernel_size=1, key=key1
        )
        self.attention_batch_norm = eqx.nn.BatchNorm(128, axis_name="batch")
        self.attention_conv2 = eqx.nn.Conv1d(
            128, current_channel_size, kernel_size=1, key=key2
        )
        # TODO  nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Encoder type is ASP, thus the current dims are B, Input_dim / 8 because of the 4 layers,  * 2 * output of layer4.
        self.fc = eqx.nn.Linear(current_channel_size * 2, proj_dim, key=key)

    def __call__(self, x, state, l2_norm=False):
        y = x

        # We expect a mel spectrogram as input for now.

        # y = self.torch_spec(y)
        # print(y.shape)
        if self.log_input:
            # y = jax.lax.clamp(min=float(1), x=y, max=jax.numpy.inf)
            y = jax.numpy.log(y + 1e-6)

        # y = jax.numpy.log(x + 1e-6)
        y = jax.vmap(self.instance_norm)(y)
        # y = jax.numpy.expand_dims(y, 1)
        # print(y.shape)

        y = self.conv1(y)
        # print(y.shape)

        y = jax.nn.relu(y)
        y, state = self.batch_norm(y, state)
        # print(y.shape)
        # y, state = self.test(y, state)
        for block in self.layer1:
            y, state = block(y, state)
            # print(y.shape)

        for block in self.layer2:
            y, state = block(y, state)

        for block in self.layer3:
            y, state = block(y, state)

        for block in self.layer4:
            y, state = block(y, state)

        y = jax.numpy.reshape(y, (-1, y.shape[-1]))

        # TODO not really justified...
        w = self.attention_conv1(y)
        w = jax.nn.relu(w)
        w, state = self.attention_batch_norm(w, state)
        w = self.attention_conv2(w)  # W represents the
        w = jax.nn.softmax(w, axis=1)

        mu = jax.numpy.sum(y * w, axis=1)
        sg = jax.lax.clamp(
            min=1e-5,
            x=jax.numpy.sum((y**2) * w, axis=1) - mu**2,
            max=jax.numpy.inf,
        )
        sg = jax.numpy.sqrt(sg)

        y = jax.lax.concatenate((mu, sg), 0)

        y = self.fc(y)

        if l2_norm:
            y = y / jax.numpy.linalg.norm(y, ord=2, axis=0)
        print(f"Ours {y.shape}")
        print(f"{y[0]}")

        return y, state
