import numpy as np
from xml.etree import ElementTree


class Dataset:

    def __init__(self, path_to_in_xml, valid_part=0.1, test_part=0.1):
        self._path_to_in_xml = path_to_in_xml

        self._test_set = {}
        self._valid_set = {}
        self._train_set = {}

        self._valid_part = valid_part
        self._test_part = test_part

        self._sym_dict = {':': 1, ',': 2, '.': 3, '!': 4, '?': 5}
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

    def parse_xml(self):
        tree_root = ElementTree.parse(self._path_to_in_xml).getroot()

        word_tags = []
        content_tags = []

        word_ids = []
        for sentence in tree_root.iter('sentence'):

            sentence_words = sentence.findall('word')
            for idx, word in enumerate(sentence_words):
                word_ids.append(idx)
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
                p_end = content_tags[idx+1].get('PunktEnd')
                if p_end is not None:
                    attr['punkt_end'] = int(p_end)
                else:
                    attr['punkt_end'] = 0
                attr['pos'] = word_ids[idx]

                attrs.append(attr)

        return words, attrs

    def create_ds(self, use_th_gr=False):
        words, attrs = self.parse_xml()

        classes = list(map(lambda x: x['class'], attrs))
        classes = np.array(classes, dtype=np.int)
        feats = self.get_features(words, attrs)

        if use_th_gr:
            feats = self.gen_three_grams(feats)

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

    def get_features(self, words_array, attr_array):
        feature_array = []

        for i in range(len(words_array)):
            word_features = list()

            last_sym = words_array[i][-1]

            if last_sym in self._sym_keys:
                word_features.append(self._sym_dict[last_sym])
            else:
                word_features.append(0)
            words_array[i] = words_array[i].strip(',. -:;!?')
            word_features.append(int(words_array[i][0].isupper()))
            word_features.append(len(words_array[i]))
            count_of_vowels = sum(map(Dataset.is_vowel, words_array[i]))
            word_features.append(count_of_vowels)
            word_features.append(ord(words_array[i][0]))
            word_features.append(ord(words_array[i][-1]))
            word_features.append(int(Dataset.has_symbol(words_array[i], 'ъ')))
            word_features.append(int(Dataset.has_symbol(words_array[i], 'ь')))
            word_features.append(int(Dataset.has_symbol(words_array[i], 'й')))
            word_features.append(attr_array[i]['pos'])

            if i < len(words_array) - 1:
                if attr_array[i + 1]['pos'] != 0:
                    word_features.append(0)
                else:
                    word_features.append(1)
            else:
                word_features.append(1)

            feature_array.append(np.array(word_features))

        return np.stack(feature_array)

    @staticmethod
    def gen_three_grams(feats):
        num_ex, num_feats = feats.shape
        three_grams_arr = np.zeros(shape=(num_ex, num_feats*3))

        for i in range(num_ex):
            if feats[i][-2] == 0:
                f_feat = np.zeros(num_feats)
            else:
                f_feat = feats[i-1]

            if feats[i][-1] == 1:
                l_feat = np.zeros(num_feats)
            else:
                l_feat = feats[i+1]

            three_grams_arr[i] = np.concatenate((f_feat, feats[i], l_feat))

        return three_grams_arr

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
