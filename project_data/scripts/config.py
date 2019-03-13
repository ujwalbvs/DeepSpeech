"""
This file will contain the global configs necessary for the scripts
"""

Global_Root = "/home/ubuntu/DeepSpeech/project_data"
Librispeech_Home = "%s/orig_audio" % Global_Root
Json_Home = "%s/intermediate_results" % Global_Root
Audio_Corpora_Home = "%s/final_corpora" % Global_Root
Test_Corpora_Home = "%s/final_test_data" % Global_Root
Inv_Index_Home = "%s/whoosh_index" % Json_Home
Embeddings_Home = "%s/embeddings" % Audio_Corpora_Home
DS_Corpora_Txt_Home = "%s/text" % Embeddings_Home

Speaker_Ext = ".spk"
Flac_Ext = ".flac"
Text_Ext = ".wrd"
Wav_Ext = ".wav"

Speaker_Id = "3752"     # This speaker has got the highest number of audio files (110)
Test_Keywords_Filename_Prefix = "test_phrases"
Ground_Truth_Index_Name = "Ground_Truth_Index"
Emitted_Index_Name = "Emitted_Index"
Test_Keywords_Mapping_Prefix = "test_phrase_mapping"
