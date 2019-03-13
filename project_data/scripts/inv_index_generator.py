"""
This script generates the inverted index based on the Whoosh library
We will create the index on both the ground truth corpora and the emitted corpora
"""

from config import *
from utils import *
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh import analysis
import sys
import os


def generate_inv_index(root_dir, index_name):
    """
    Schema definition: title(name of file), path(as ID), content(indexed
    but not stored), textdata (stored text content)
    :param root_dir: directory where all the text files are present
    :param index_name: name of the index (ground truth vs emitted)
    :return:
    """
    # needed to include standard stop-words:
    # SO: https://stackoverflow.com/questions/25087290/python-whoosh-seems-to-return-incorrect-results
    custom_analyzer = analysis.StandardAnalyzer(expression=r'[\w-]+(\.?\w+)*', stoplist=None)
    schema = Schema(title=TEXT(stored=True), path=ID(stored=True),
                    content=TEXT(analyzer=analysis.StandardAnalyzer(stoplist=None)),
                    textdata=TEXT(stored=True, analyzer=analysis.StandardAnalyzer(stoplist=None)))

    # Creating a index writer to add document as per schema
    ix = create_in(Inv_Index_Home, schema, indexname=index_name)
    writer = ix.writer()
    legal_fileids = set(get_ids_for_speaker(Speaker_Id))

    for filename in list_all_files_with_ext(root_dir, Text_Ext):
        fileid = filename.split(Text_Ext)[0]
        if extract_parent_fileid(fileid) in legal_fileids:
            complete_path = os.path.join(root_dir, filename)
            with open(complete_path) as wrd_file:
                text = wrd_file.read()
                # print(fileid, text)
                writer.add_document(title=fileid, path=complete_path, content=text, textdata=text)
    writer.commit()


if __name__ == '__main__':
    index_loc_pairs = [(Librispeech_Home, Ground_Truth_Index_Name), (DS_Corpora_Txt_Home, Emitted_Index_Name)]
    for loc, idx in index_loc_pairs:
        print("Generating Inverted Index for: %s from location: %s" % (idx, loc))
        generate_inv_index(loc, idx)
