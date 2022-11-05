import d3rlpy
from d3rlpy.models.torch import VectorEncoder

from replay.models.rl.sdac.vector_encoder_w_action import VectorEncoderWithAction


class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return VectorEncoder(observation_shape, [self.feature_size, self.feature_size])

    def create_with_action(self, observation_shape, action_size):
        return VectorEncoderWithAction(
            observation_shape, action_size, [self.feature_size, self.feature_size]
        )

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
