from src.transforms.data_creator import DataCreator, MultiNodeData
from src.transforms.node_selector import NodeSelector

from src.transforms.data_labeler import DataLabeler, anomaly_codes, anomaly_names

from src.transforms.data_enhancer import DataEnhancer
from src.transforms.node_connector import NodeConnector
from src.transforms.min_max_scaler import MinMaxScaler
from src.transforms.test_masker import TestMasker
from src.transforms.anomaly_selector import AnomalySelector
from src.transforms.metric_selector import MetricSelector

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
                
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))