"""
This script is used to run in a batch-mode and will generate all the file-id map for the test phrases
"""
from config import *
from utils import *
from full_text_search import get_fileids_from_test_key


def generate_mapping(n_gram):
    all_audio_test_files = list(list_all_files_with_ext(os.path.join(Test_Corpora_Home, "%d_gram" % n_gram), Wav_Ext))
    result = get_fileids_from_test_key(all_audio_test_files, n_gram)
    filename = "%s_%d.json" % (Test_Keywords_Mapping_Prefix, n_gram)
    dump_json(result, filename)
    print("Generated file mapping at: %s" % filename)


if __name__ == '__main__':
    n_grams = [3, 4, 5]
    for ng in n_grams:
        generate_mapping(ng)
