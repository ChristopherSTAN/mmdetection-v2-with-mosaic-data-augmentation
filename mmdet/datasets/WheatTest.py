from .custom import CustomDataset
from .builder import DATASETS


@DATASETS.register_module()
class WheatDatasetTest(CustomDataset):

    CLASSES = ('wheat', )