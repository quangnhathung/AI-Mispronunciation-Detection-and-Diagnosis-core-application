"""

[M-aug] SpecAugment trên feature CNN (chỉ khi model.train()).



Che ngẫu nhiên dải thời gian (time) và chiều feature (frequency) để giảm overfit giọng.

"""

import random

import torch

import torch.nn as nn

from config import Config





class FeatureSpecAugment(nn.Module):

    """

    SpecAugment kiểu Google — áp dụng lên tensor [B, T, D] sau encoder CNN.



    Chỉ chạy khi self.training và Config.USE_SPEC_AUGMENT.

    """



    def __init__(

        self,

        freq_mask_max=None,

        time_mask_max=None,

        num_freq_masks=None,

        num_time_masks=None,

    ):

        super().__init__()

        self.freq_mask_max = freq_mask_max or Config.SPEC_FREQ_MASK_MAX

        self.time_mask_max = time_mask_max or Config.SPEC_TIME_MASK_MAX

        self.num_freq_masks = num_freq_masks or Config.SPEC_NUM_FREQ_MASKS

        self.num_time_masks = num_time_masks or Config.SPEC_NUM_TIME_MASKS



    def forward(self, x):

        if not self.training or not Config.USE_SPEC_AUGMENT:

            return x
        
        x = x.clone()

        b, t, d = x.shape

        for _ in range(self.num_freq_masks):

            f = random.randint(0, min(self.freq_mask_max, d))

            if f == 0:

                continue

            f0 = random.randint(0, d - f)

            x[:, :, f0 : f0 + f] = 0



        for _ in range(self.num_time_masks):

            tm = random.randint(0, min(self.time_mask_max, t))

            if tm == 0:

                continue

            t0 = random.randint(0, t - tm)

            x[:, t0 : t0 + tm, :] = 0



        return x


