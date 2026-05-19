# models/transformer_forecaster.py

import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):

    def __init__(self,
                 input_dim=1,
                 d_model=64,
                 nhead=4,
                 num_layers=2):

        super(TransformerForecaster, self).__init__()

        # ==========================================
        # INPUT PROJECTION
        # ==========================================

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # ==========================================
        # TRANSFORMER ENCODER
        # ==========================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer_encoder = \
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

        # ==========================================
        # DEMAND PREDICTION HEAD
        # ==========================================

        self.demand_head = nn.Linear(
            d_model,
            1
        )

        # ==========================================
        # CONFIDENCE HEAD
        # ==========================================

        self.confidence_head = nn.Linear(
            d_model,
            1
        )

        self.sigmoid = nn.Sigmoid()

    # ==============================================
    # FORWARD
    # ==============================================

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer_encoder(x)

        x = x[:, -1, :]

        # ==========================================
        # DEMAND FORECAST
        # ==========================================

        demand = self.demand_head(x)

        demand = torch.relu(demand)

        # ==========================================
        # CONFIDENCE SCORE
        # ==========================================

        confidence = self.confidence_head(x)

        confidence = self.sigmoid(confidence)

        return demand, confidence