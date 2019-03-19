"""
This script is used to generate the results for the full-text based keyword search
"""

from config import *
from utils import *
from full_text_search import search_across_index
import os


def get_results_for_phrase(key, phrase, ground_truth_dict):
    query_phrase = '"%s"' % phrase
    search_results = search_across_index(query_phrase)
    ground_truth = ground_truth_dict[key]
    assert (len(set(ground_truth)) == len(ground_truth))
    ground_truth = set(ground_truth)
    if key.startswith("TP"):
        assert(len(ground_truth) > 0)
    else:
        assert(len(ground_truth) == 0)
    predicted_files = list(set([extract_parent_fileid(res['fileid']) for res in
                                search_results[Emitted_Index_Name]['results']]))
    # print(key, phrase, ground_truth, predicted_files)
    return ground_truth, predicted_files


def get_precision(phrases, ground_truth_dict):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for key in phrases:
        if "_OD_" in key:
            ground_truth, predicted_files = get_results_for_phrase(key, phrases[key], ground_truth_dict)
            true_pos += len([file for file in predicted_files if file in ground_truth])
            false_pos += len([file for file in predicted_files if file not in ground_truth])
            true_neg += (1 if len(predicted_files) == 0 and len(ground_truth) == 0 else 0)
            false_neg += len([file for file in list(ground_truth) if file not in predicted_files])
    print(true_pos, false_pos, true_neg, false_neg)
    precision = true_pos * 1.0 / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos * 1.0 / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1_score = 2.0 * (precision * recall) / (precision + recall)
    print(precision, recall, f1_score)


def get_results(phrases, n_gram):
    mapping_dict = load_json("%s_%d.json" % (Test_Keywords_Mapping_Prefix, n_gram))
    ground_truth_dict = {key.split(Wav_Ext)[0]: mapping_dict[key] for key in mapping_dict}
    print("Loaded %d test phrases" % len(phrases))
    assert(len(phrases) == len(ground_truth_dict))
    get_precision(phrases, ground_truth_dict)


def generate_phrase_dict_from_ds_output(n_gram):
    target_directory = "%s/%d_gram/text" % (Test_Corpora_Home, n_gram)
    result = dict()
    for filename in list_all_files_with_ext(target_directory, Text_Ext):
        with open(os.path.join(target_directory, filename)) as f:
            transcribed_phrase = f.read()
            key = filename.split(Text_Ext)[0]
            result[key] = transcribed_phrase
    return result


if __name__ == '__main__':
    for ng in range(2,7):
        transcribed_dict = generate_phrase_dict_from_ds_output(ng)
        # print(transcribed_dict)
        print("NGram: %d" % (ng))
        get_results(transcribed_dict, ng)

