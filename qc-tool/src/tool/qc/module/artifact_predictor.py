import pandas as pd

from ..struct.recording import Recording
from ..struct.epoch import Epoch
from ..struct.model_config import ModelsConfig, ModelConfig


def _predict_for_epochs(epochs: list[Epoch], model_config: ModelConfig) -> None:
    """What preforms the actual prediction. Batches all epochs (of a channel) at once."""

    # Build a DataFrame of features for all epochs at once
    X = pd.DataFrame(
        [
            {
                feature: getattr(epoch, feature)
                for feature in model_config.features_names
            }
            for epoch in epochs
        ]
    )
    # Predict
    y = model_config.model.predict(X)

    # Save
    for epoch, pred in zip(epochs, y):
        epoch.predictions[model_config.model_name] = bool(pred)


def predict(recording: Recording, models_config: ModelsConfig) -> None:
    """Predicts artifacts using the different models and saves the results to the epochs"""

    # Initialize has_predictions
    for channel in recording.channels:
        channel.has_predictions = False

    # Go over the different models
    for model_config in models_config.models:

        # Go over the target channels
        for channel in recording.channels:
            if channel.name not in model_config.target_channels:
                continue

            channel.has_predictions = True

            # Predict for all epochs in this channel at once
            _predict_for_epochs(channel.epochs, model_config)
