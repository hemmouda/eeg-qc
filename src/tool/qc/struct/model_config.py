import joblib
from pathlib import Path
from typing import Any


class ModelsConfig:
    """This, along side ModelConfig, should allow future work to run different models with different configurations.
    Unlike other parts of the QC, operations on these "structs" do not perform any checks. So make sure to provide
    all needed data."""

    epoch_duration_s: float | int
    """You can use different models but they should use features from the same epoch duration.
    Otherwise rework this so that it re-cuts the epochs differently every time for each model."""

    models: list["ModelConfig"]

    def __init__(self):
        self.models = []


class ModelConfig:
    """Individual model config. Defines elements like what features it use, where model can be found, etc..."""

    models_config: "ModelsConfig"

    model_joblib_path: Path
    """File path of the model as a joblib"""

    model: Any
    """The model itself that is loaded from model_joblib_path. Loaded automatically."""

    model_name: str
    """Model name. These need to be unique for the epochs to store different model predictions"""

    artifact_name: str
    """Name of artifact that is being predicted"""

    target_channels: set[str]
    """Channel names on which prediction should be made."""

    features_names: list[str]
    """List of feature names that are needed for the prediction. The epoch should have these."""

    def __init__(self, models_config: "ModelsConfig", model_joblib_path: Path):
        """Automatically assign the models_config and appends itself to
        its list of the models. And loads the model."""

        self.models_config = models_config
        self.models_config.models.append(self)

        self.model_joblib_path = Path(model_joblib_path)

        self.model = joblib.load(self.model_joblib_path)

        # These little behaviors do make it become considered a class
        # and not a struct, but meh
