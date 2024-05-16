# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Callable
import importlib
import torch
from openfold.model.primitives import Linear as LinearOriginal
import loralib as lora

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

class Linear(LinearOriginal,lora.Linear):
    """
    Make the original Linear layer to be a child class of lora linear layer as well
    to enable Low-rank Adaptation (LoRA)

    Details can be found: https://github.com/microsoft/LoRA
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        precision=None
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        LinearOriginal.__init__(in_dim, out_dim, bias=bias, init=init, init_fn=init_fn,precision=precision)
        lora.Linear.__init__(in_dim, out_dim, r=8) # r means rank, a hyperperameter when using LoRA, default is 8

    def LoRA(self, input : torch.Tensor,weight :torch.Tensor, bias : torch.Tensor):
        """
        Implement LoRA process 

        Adapted from https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L144
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = torch.nn.functional.linear(input, T(weight), bias=bias)            
            result += (self.lora_dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling # apply LoRA to results instead of using the full weight matrix
            return result
        else:
            # this mean LoRA is not used, apply linear transformation using the full weight matrix directly
            return torch.nn.functional.linear(input, weight, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Need to combine the forward() methods from both classes"""
        d = input.dtype
        deepspeed_is_initialized = (
                deepspeed_is_installed and
                deepspeed.comm.comm.is_initialized()
        )
        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return self.LoRA(input.to(dtype=self.precision),
                                            self.weight.to(dtype=self.precision),
                                            bias).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return self.LoRA(input, self.weight.to(dtype=d), bias)

        