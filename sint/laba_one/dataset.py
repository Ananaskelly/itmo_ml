import numpy as np
from xml.etree import ElementTree


class Dataset:

    def __init__(self, path_to_in_xml, num_cls, valid_part=0.1, test_part=0.1):
        self._path_to_in_xml = path_to_in_xml
        self._num_cls = num_cls

        self._test_set = {}
        self._valid_set = {}
        self._train_set = {}

        self._valid_part = valid_part
        self._test_part = test_part

    @property
    def test_set(self):
        return self._test_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def train_set(self):
        return self._train_set

    def parse_xml(self):
        tree_root = ElementTree.parse(self._path_to_in_xml).getroot()

        word_tags = []
        content_tags = []

        for word in tree_root.iter('word'):
            word_tags.append(word)

        for content in tree_root.iter('content'):
            content_tags.append(content)

        words = []
        attrs = []
        for idx, word in enumerate(word_tags):
            _word = word.get('original')

            attr = {}
            if _word is not None:
                words.append(_word)
                attr['class'] = word.find('dictitem').get('subpart_of_speech')
                content = content_tags[idx+1]
                attrs.append(attr)

        return words, attrs

    def create_ds(self):
        words, attrs = self.parse_xml()

        classes = list(map(lambda x: x['class'], attrs))
        classes = np.array(classes, dtype=np.int)
        feats = Dataset.get_features(words)

        feats, classes = Dataset.random_shuffle(feats, classes)
        num_ex = classes.shape[0]

        train_bound = int(num_ex * (1 - self._valid_part - self._test_part))
        valid_bound = int(num_ex * (1 - self._valid_part))

        self._train_set = {
            'data': feats[:train_bound, :],
            'labels': classes[:train_bound]
        }

        self._valid_set = {
            'data': feats[train_bound:valid_bound, :],
            'labels': classes[train_bound:valid_bound]
        }

        self._test_set = {
            'data': feats[valid_bound:, :],
            'labels': classes[valid_bound:]
        }

    @staticmethod
    def get_features(words_array):
        feature_array = []

        for i in range(len(words_array)):
            word_features = list()

            word_features.append(int(words_array[i][0].isupper()))
            word_features.append(len(words_array[i]))
            count_of_vowels = sum(map(Dataset.is_vowel, words_array[i]))
            word_features.append(count_of_vowels)
            word_features.append(int(Dataset.has_symbol(words_array[i], 'ъ')))
            word_features.append(int(Dataset.has_symbol(words_array[i], 'ь')))
            word_features.append(int(Dataset.has_symbol(words_array[i], 'й')))

            feature_array.append(np.array(word_features))

        return np.stack(feature_array)

    @staticmethod
    def has_symbol(in_str, sym):
        return sym in in_str

    @staticmethod
    def is_vowel(in_sym):
        vowel_set = set("аоиеёэыуюяАОИЕЁЭЫУЮЯ")

        if in_sym in vowel_set:
            return 1
        return 0

    @staticmethod
    def random_shuffle(x, y):
        num_ex = x.shape[0]

        perm = np.arange(num_ex)
        np.random.shuffle(perm)

        x = x[perm]
        y = y[perm]

        return x, y
