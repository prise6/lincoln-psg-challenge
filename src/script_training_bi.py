
import os
import pandas as pd
import numpy as np
from src.tools import Config
from dotenv import find_dotenv, load_dotenv
from src.models import DataLoader, DataLoaderEvents
from src.models import PositionModel_embeded, PositionModel_events, PositionModel_embeded_bid
from src.models import ModelTrainer

## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## import des données
# games = pd.read_csv(cfg.get('data_loader')['simple']['data_path'], nrows = 10)
# print(games.shape)

## Data loader
loader = DataLoader(cfg, "simple")

## Model simple pour récuprérer les poids
model_simple = PositionModel_embeded(cfg.get("models")['position']['embeded'], loader.len_seq_train, loader.len_seq_pred, loader.output_dim)
# model_simple.load(which = 'final_embeded_1_200_epochs')
# weights_embeded = model_simple.model.get_weights()
# embeded_position = weights_embeded[0]
# embeded_event = weights_embeded[1]
# embeded_zone = weights_embeded[2]

## Model
model = PositionModel_embeded_bid(cfg.get("models")['position']['embeded_bi'], loader.len_seq_train, loader.len_seq_pred, loader.output_dim)

# weights = model.model.get_weights()
# weights[0] = embeded_position
# weights[1] = embeded_event
# weights[2] = embeded_zone
# model.model.set_weights(weights)

model.load()

## summary
model.model.summary()

## Trainer
trainer = ModelTrainer(model, loader, cfg.get("trainer")['classic'])

trainer.train()

