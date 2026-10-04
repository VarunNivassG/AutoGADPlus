from __future__ import annotations
from typing import Any
from autogad_reproduction.models.anemone import ANEMONE
from autogad_reproduction.models.external import OfficialRepoAdapter, PyGODAdapter


def build_model(name: str, logger=None, run_context=None, **kwargs) -> Any:
    key = name.lower().replace("-", "_")
    if key == "anemone":
        return ANEMONE(logger=logger, run_context=run_context, **kwargs)
    official = {
        "cola": "https://github.com/TrustAGI-Lab/CoLA",
        "gradate": "https://github.com/FelixDJC/GRADATE",
        "sl_gad": "https://github.com/KimMeen/SL-GAD",
        "sub_cr": "https://github.com/Zjer12/Sub",
    }
    if key in official:
        return OfficialRepoAdapter(name=key, repository=official[key])
    pygod = {"dominant":"DOMINANT", "anomalydae":"AnomalyDAE", "guide":"GUIDE", "gaan":"GAAN", "conad":"CONAD"}
    if key in pygod:
        return PyGODAdapter(name=key, class_name=pygod[key])
    raise KeyError(f"Unknown model: {name}")
