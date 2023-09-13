import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
                dim,
                kernel_initializer=normc_initializer(1.0),
                )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = tf.keras.activations.tanh(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = DenseLayer(dim)
        self.dense2 = DenseLayer(dim)

    def call(self, x, **kwargs):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class FCNet(TFModelV2):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            ):
        super(FCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        inputs = tf.keras.layers.Input(
                shape=(int(np.product(obs_space.shape)),),
                name="observations",
                )

        feature = Encoder(256)(inputs)
        logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                kernel_initializer=normc_initializer(0.01),
                )(feature)
        value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                # kernel_initializer=normc_initializer(0.01),
                kernel_initializer=normc_initializer(1.),
                )(feature)

        self.base_model = tf.keras.Model(
                inputs=[inputs],
                outputs=[logits_out, value_out],
                )
        self.register_variables(self.base_model.variables)

        self._value_out = None

    def forward(
            self,
            input_dict,
            state,
            seq_lens,
            ):
        output, self._value_out = self.base_model([input_dict["obs_flat"]])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        output = output + inf_mask

        return output, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("custom_fcnet", FCNet)
