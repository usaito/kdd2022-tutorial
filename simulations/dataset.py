from dataclasses import dataclass

import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.utils import softmax, sample_action_fast

@dataclass
class SyntheticBanditDataset(SyntheticBanditDataset):

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        if self.reward_type == "binary":
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif self.reward_type == "continuous":
            reward = self.random_.normal(
                loc=expected_reward_factual,
                scale=self.reward_std,
            )
        else:
            raise NotImplementedError

        return reward

    def obtain_batch_bandit_feedback(self, n_rounds: int):
        """Obtain batch logged bandit data."""
        contexts = self.random_.normal(size=(n_rounds, self.dim_context))

        # calc expected reward given context and action
        expected_reward_ = self.calc_expected_reward(contexts)

        # calculate the action choice probabilities of the logging policy
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # create some deficient actions based on the value of `n_deficient_actions`
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(pi_b_logits)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)

        # sample actions for each round based on the logging policy
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            expected_reward=expected_reward_,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.0,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    base_pol = np.zeros_like(expected_reward)
    if is_optimal:
        a = np.argmax(expected_reward, axis=1)
    else:
        a = np.argmin(expected_reward, axis=1)
    base_pol[
        np.arange(expected_reward.shape[0]),
        a,
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol[:, :, np.newaxis]
