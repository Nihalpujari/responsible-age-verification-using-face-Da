"""face_age — responsible age-classification project (FairFace).

Package layout (planned):
    data.py     — dataset loading + canonical schema (this milestone)
    model.py    — backbone + age head (+ auxiliary gender/race heads)
    metrics.py  — subgroup accuracy / fairness gaps
    explain.py  — Grad-CAM / attention XAI
"""

__all__ = ["data"]
