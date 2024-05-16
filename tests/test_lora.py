# Copyright 2021 AlQuraishi Laboratory
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
import random
from openfold.model.primitives_lora import Linear
import torch

class FakeModel(torch.nn.Module):
    """
    A small neural net, pretend to learn a multiclass classification task 
    """

    def __init__(self, layer1 : Linear, layer2 : Linear):
        super(FakeModel, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
    
    def forward(
            self, input : torch.Tensor
    ):
        x = self.layer1(input)
        x = self.layer2(x)

        return x

class TestLoRA(unittest.TestCase):
    """
    Test the LoRA implementation of the linear layer
    """

    def setUp(self):
        self.input = torch.rand(34*57).reshape(24, 57)
        self.layer1 = Linear(in_dim=57, out_dim=21, bias=True)
        self.layer2 = Linear(in_dim=21, out_dim=6, bias=False)
        self.ground_truth = [random.choice(range(6)) for _ in range(6)]