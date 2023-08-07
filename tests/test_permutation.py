# Copyright 2021 AlQuraishi Laboratory
# Dingquan Yu @ EMBL-Hamburg Kosinski group
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

from pathlib import Path
import pickle
import torch
import torch.nn as nn
import numpy as np
import unittest
from openfold.config import model_config
from openfold.data import data_transforms
from openfold.model.model import AlphaFold
from openfold.utils.loss import AlphaFoldMultimerLoss
from openfold.data.data_modules import OpenFoldMultimerDataModule
import logging
logger = logging.getLogger(__name__)
import os
from tests.data_utils import (
    random_template_feats,
    random_extra_msa_feats,
    random_affines_vector
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.rigid_utils import (
    Rigid,
)
from openfold.utils.tensor_utils import tensor_tree_map

class TestPermutation(unittest.TestCase):
    def setUp(self):
        """
        Firstly setup model configs and model as in
        test_model.py

        In the test case, use PDB ID 1e4k as the label
        """
        config = model_config(
        'model_1_multimer_v3', 
        train=True, 
        low_prec=True) 
        self.test_data_dir = os.path.join(os.getcwd(),"tests/test_data")
        self.config = config
        self.config.loss.masked_msa.num_classes = 22 # somehow need overwrite this part in multimer loss config
        self.config.model.evoformer_stack.no_blocks = 4  # no need to go overboard here
        self.config.model.evoformer_stack.blocks_per_ckpt = None  # don't want to set up
        self.model = AlphaFold(config).to('cuda')
        self.loss = AlphaFoldMultimerLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1
        self.data_module = OpenFoldMultimerDataModule(config=self.config.data,
                                                      batch_seed=42,
                                                      max_template_date='2500-01-01',
                                                      train_mmcif_data_cache_path=os.path.join(self.test_data_dir,"train_mmcifs_cache.json"),
                                                      template_mmcif_dir="/g/alphafold/AlphaFold_DBs/pdb_mmcif/mmcif_files/",
                                                      train_data_dir="/g/alphafold/AlphaFold_DBs/pdb_mmcif/mmcif_files/",
                                                      train_alignment_dir=os.path.join(self.test_data_dir,"alignments"))
        self.data_module.prepare_data()
        self.data_module.setup()
    
    def test_dry_run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        train_dataloader = self.data_module.train_dataloader()
        for dataset in train_dataloader:
            all_chain_features,ground_truth = dataset
            move_to_gpu = lambda t: (t.to('cuda'))
            all_chain_features = tensor_tree_map(move_to_gpu,all_chain_features)
            ground_truth = [tensor_tree_map(move_to_gpu,l) for l in ground_truth]
            out = self.model(all_chain_features)
            asym_id = all_chain_features["asym_id"][...,-1]
            for id in torch.unique(out['asym_id']):
                id = int(id.item())
                asym_mask = (asym_id == id).bool()
                pred_pos = out['final_atom_positions'][0][asym_mask[0]]
                pred_pos = np.expand_dims(pred_pos.detach().to('cpu').numpy(),0)
                ground_truth[id-1]['all_atom_positions'] = torch.tensor(pred_pos,device='cuda')
            asym_id=None
            pred_pos=None
            asym_mask=None
            loss = self.loss(
                out,(all_chain_features,ground_truth),_return_breakdown=False
            )