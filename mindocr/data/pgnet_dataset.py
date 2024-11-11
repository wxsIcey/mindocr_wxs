import logging
from .det_dataset import DetDataset

__all__ = ["PGDataset"]
_logger = logging.getLogger(__name__)

class PGDataset(DetDataset):
    """
    General dataset for e2e recognition
    """
