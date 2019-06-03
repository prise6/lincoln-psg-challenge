
import os
import pandas as pd
import numpy as np
from src.tools import Config
from dotenv import find_dotenv, load_dotenv
from src.models import DataLoader
from src.models import PositionModel_embeded
from src.models import ModelTrainer

## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## Data loader
loader = DataLoader(cfg, "simple")

## Model
model = PositionModel_embeded(cfg.get("models")['position']['embeded'], loader.len_seq_train, loader.len_seq_pred, loader.output_dim)
model.load()

## summary
model.model.summary()

## Trainer
trainer = ModelTrainer(model, loader, cfg.get("trainer")['classic'])

trainer.train()

