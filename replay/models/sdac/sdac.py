from typing import Optional, Any, Callable

from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory

import replay.models.sdac.sdac_impl as sdac_impl
from replay.models.cql import RLRecommender


class SDAC(RLRecommender):
    r"""FIXME: add docstring"""

    model: sdac_impl.SDAC

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "n_epochs": {"type": "int", "args": [3, 20]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }

    def __init__(
            self, *,
            top_k: int, n_epochs: int = 1,
            action_randomization_scale: float = 0.,
            use_negative_events: bool = False,
            rating_based_reward: bool = False,
            reward_top_k: bool = False,
            epoch_callback: Optional[Callable[[int, RLRecommender], None]] = None,

            # SDAC inner params
            actor_learning_rate: float = 3e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 3e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            actor_encoder_factory: EncoderArg = "default",
            critic_encoder_factory: EncoderArg = "default",
            q_func_factory: QFuncArg = "mean",
            batch_size: int = 256,
            n_frames: int = 1,
            n_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            n_critics: int = 2,
            initial_temperature: float = 1.0,
            use_gpu: UseGPUArg = False,
            scaler: ScalerArg = None,
            action_scaler: ActionScalerArg = None,
            reward_scaler: RewardScalerArg = None,
            gumb_temp: float = 1, 
            dist_tresh: float = 0.5,
            **params: Any
    ):
        model = sdac_impl.SDAC(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            impl=None,
            gumb_temp = gumb_temp, 
            dist_tresh =  dist_tresh,
            **params
        )

        super(SDAC, self).__init__(
            model=model,
            top_k=top_k, n_epochs=n_epochs,
            action_randomization_scale=action_randomization_scale,
            use_negative_events=use_negative_events,
            rating_based_reward=rating_based_reward,
            reward_top_k=reward_top_k,
            epoch_callback=epoch_callback
        )
