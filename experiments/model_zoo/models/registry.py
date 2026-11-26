from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "han": {
        "family": "metapath",
        "class": "HAN",
        "module": "models.metapath.han",
        "needs_metapaths": True,
    },
    "rgcn": {
        "family": "relation",
        "class": "RelationalGCN",
        "module": "models.relation.rgcn",
        "needs_metapaths": False,
    },
    "hetgnn": {
        "family": "relation",
        "class": "HetGNN",
        "module": "models.relation.hetgnn",
        "needs_metapaths": False,
    },
    "rshn": {
        "family": "relation",
        "class": "RSHN",
        "module": "models.relation.rshn",
        "needs_metapaths": False,
    },
    "simplehgn": {
        "family": "relation",
        "class": "SimpleHGN",
        "module": "models.relation.simplehgn",
        "needs_metapaths": False,
    },
}


def build_model(model_name: str, data, target_type: str, num_classes: int, config: Dict[str, Any]):
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {', '.join(sorted(MODEL_REGISTRY))}")
    entry = MODEL_REGISTRY[key]
    mod = importlib.import_module(f".{entry['module']}", package="model_zoo")
    cls = getattr(mod, entry["class"])
    logger.info("Building %s", entry["class"])
    return cls.from_data(data=data, target_type=target_type, num_classes=num_classes, **config)
