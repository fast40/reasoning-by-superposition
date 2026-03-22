# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np
from pydantic import BaseModel

class Config(BaseModel):
      model_config = {'frozen': True}

      project: str
      save_path: str
      name: str

      only_eval: bool

      coconut: bool
      cot: bool
      no_thoughts: bool
      no_cot: bool

      c_thought: int
      epochs_per_stage: int
      max_latent_stage: int
      pad_latent_to_max: bool

      save_only_improve: bool
      uniform_prob: float
      model_id: str
      load_model_path: str | None
      seed: int
      resume: int
      bf16: bool
      train_path: str
      val_path: str
      reset_optimizer: bool
      batch_size_training: int
      debug: bool
      gradient_accumulation_steps: int
      num_epochs: int
      lr: float
      weight_decay: float


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
