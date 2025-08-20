import torch
from torch.nn import functional as F
import random
from simuclr.utils import get_default_device, paint


class ContrastiveAugPolicyPlanner:
    def __init__(
        self,
        sub_policies,
        device=None,
    ):
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device

        self.sub_policies = sub_policies
        self.current_subpolicies = [None, None]
        # Initialize current_subpolicy and current_subpolicy_one_hot
        self.sample_policies()

        print(
            paint(
                f"Initialized AugPolicyPlanner with {len(sub_policies)} sub-policies",
                "green",
            )
        )
        for i, sub_policy in enumerate(sub_policies):
            formatted_policy = ", ".join([str(aug) for aug in sub_policy])
            print(
                paint(
                    f"Policy {i+1}: {formatted_policy}",
                    "green",
                )
            )

    def sample_policies(self):
        """
        Randomly sample two distinct augmentation policies.
        If only one policy is available, both policies are set to the same.
        :return: Two sampled policies.
        """
        if len(self.sub_policies) == 1:
            policy1 = policy2 = self.sub_policies[0]
        else:
            policy1, policy2 = random.choices(self.sub_policies, k=2)
        self.current_subpolicies = [policy1, policy2]
        return policy1, policy2

    @property
    def current_subpolicies(self):
        return self._current_subpolicies

    @current_subpolicies.setter
    def current_subpolicies(self, policies):
        self._current_subpolicies = policies
