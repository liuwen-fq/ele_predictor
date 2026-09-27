"""Legacy Bamboo oracle boundary; preserves checkpoint and device semantics."""
from pathlib import Path
import yaml
from .mono import Mono
from .molmix import MolMix


def load_bamboo_oracle(config_path, checkpoint, device="cuda"):
    config = yaml.safe_load(Path(config_path).read_text())
    mono = Mono(**config["model"]["pretrain"]["model_config"])
    mono.load_ckpt(str(Path(checkpoint) / "pretrain.pt"))
    model = MolMix(mono, **config["model"]["model"])
    model.load_ckpt(str(Path(checkpoint) / "optimal.pt"))
    return model.eval().to(device)
