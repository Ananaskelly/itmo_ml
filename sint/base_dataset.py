class BaseDataset:

    def __init__(self, path_to_in_xml, valid_part=0.1, test_part=0.1):
        self._path_to_in_xml = path_to_in_xml

        self._test_set = {}
        self._valid_set = {}
        self._train_set = {}

        self._valid_part = valid_part
        self._test_part = test_part

        self._sym_dict = {':': 1, ',': 2, '.': 3, '!': 4, '?': 5, ';': 6, '-': 7}
        self._sym_keys = self._sym_dict.keys()

    @property
    def test_set(self):
        return self._test_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def train_set(self):
        return self._train_set
