"""
This script implements the full text search through the inverted index generated by Whoosh
"""

from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir
from config import *
from utils import *
import sys


def get_top_k_docs(query_str, index_name, top_k=200):
    index = open_dir(Inv_Index_Home, indexname=index_name)
    with index.searcher(weighting=scoring.Frequency) as searcher:
        query = QueryParser("content", index.schema).parse(query_str)
        results = searcher.search(query, limit=top_k)
        return transform_result_json(results)


def transform_result_json(results):
    json_data = dict()
    json_data['runtime'] = results.runtime
    json_data['results'] = [{'fileid': result['title'], 'path': result['path'], 'text': result['textdata']}
                            for result in results]
    return json_data


def search_across_index(query_str):
    index_list = [Ground_Truth_Index_Name, Emitted_Index_Name]
    return {name: get_top_k_docs(query_str, name) for name in index_list}


def get_fileids_from_test_key(test_file_names, n_gram):
    all_test_phrases = load_all_test_phrases(n_gram)
    overall_result = dict()
    for filename in test_file_names:
        key_id = filename.split(Wav_Ext)[0]
        tp, _, _ = key_id.split("_")
        test_phrase = '"%s"' % all_test_phrases[key_id]
        search_results = get_top_k_docs(test_phrase, Ground_Truth_Index_Name)
        fileids = [res['fileid'] for res in search_results['results']]
        if tp == "TP":
            assert(len(fileids) > 0)
        else:
            assert(len(fileids) == 0)
        overall_result[filename] = fileids
    return overall_result


if __name__ == '__main__':
    query_s = sys.argv[1]
    data = search_across_index(query_s)
    print(data)
    sample_filenames = ["TP_OD_2.wav", "TN_OD_1.wav", "TP_OD_3.wav", "TP_OD_4.wav"]
    print(get_fileids_from_test_key(sample_filenames, 6))