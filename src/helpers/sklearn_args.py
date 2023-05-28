from dataclasses import dataclass

@dataclass
class CommonArgs:
    dataset: str
    split: str
    data_path: str
    data_url: str

@dataclass
class ModelChoiceArgs:
    model: str

@dataclass
class DataloaderArgs:
    seed: int
    radius: int
    n_bits: int
    val_frac: float
    test_frac: float
    dataset_frac: float

@dataclass
class MLPRegArgs:
    layer_size: int
    learning_rate: float
    max_iter: int
    n_iter_no_change: int
    tol: float

@dataclass
class RandomForestRegArgs:
    num_estimators: int

@dataclass
class SklearnArgs:
    common_args: CommonArgs
    model_choice_args: ModelChoiceArgs
    dataloader_args: DataloaderArgs
    mlpreg_args: MLPRegArgs
    randomforest_args: RandomForestRegArgs
