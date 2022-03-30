

# MedDG is a large-scale entity-centric medical dialogue dataset related to 12 types of
# common gastrointestinal diseases, # with more than 17K conversations and 385K utterances
# collected from the online health consultation community.
# Each conversation is annotated with five different categories of entities, including
# diseases, symptoms, attributes, tests, and medicines. For more details about this dataset,
# please refer to this paper: https://arxiv.org/abs/2010.07497.

import pickle as pk
import numpy as np
from tqdm import tqdm
import json
from pygtrans import Translate
from time import sleep

# port:  https://github.com/foyoux/pygtrans
# https://pygtrans.readthedocs.io/zh_CN/latest/pygtrans.html#module-pygtrans.Translate
client = Translate()


# ********************************************************************************************
disease = []
with open("track2_medical_diagnosis/disease.txt", "r") as f_disease:
    for line in f_disease:
        disease.append(line)

symptom = []
with open("track2_medical_diagnosis/symptom.txt", "r") as f_symptom:
    for line in f_symptom:
        symptom.append(line)

# ********************************************************************************************
with open('track1_medical_dialogue_generation/new_train.pk', 'rb') as f_train_track:
    train_track = pk.load(f_train_track)

with open('track1_medical_dialogue_generation/new_test.pk', 'rb') as f_test_track:
    test_track = pk.load(f_test_track)


# # ********************************************************************************************
# # Translate the train data
# translated_data = []
# raw_data = []
# flag_num = 0
# flag_num_IN = 0
# data_dumped = []
# print('START ########################################################## START')
# for dialogue_seg in train_track:
#     for dialogue_data in dialogue_seg:
#         flag_num += 1
#         raw_data.append(dialogue_data['Sentence'])
#         if (flag_num >= 9999) or (dialogue_seg == train_track[-1] and dialogue_data == train_track[-1]):
#             # try:
#             print('IN ########################################################## IN')
#             flag_num_IN += 1
#             print(flag_num_IN)
#             data_T = client.translate(raw_data, target='en')
#             for i_sentence in range(flag_num):
#                 # print(data[i_sentence].translatedText)
#                 # print('##########################################################')
#                 translated_data.append(data_T[i_sentence].translatedText)
#
#             flag_num = 0
#             raw_data = []
#             sleep(3)
#
# print('END ########################################################## END')

# with open('train_data_translated.json', 'w', encoding='utf-8') as data_dumped:
#     json.dump(translated_data, data_dumped)




# ********************************************************************************************
# processing the train data
# with open('track1_medical_dialogue_generation/new_train.pk', 'rb') as f_train_track:
#     train_track = pk.load(f_train_track)
#
# with open('track1_medical_dialogue_generation/new_test.pk', 'rb') as f_test_track:
#     test_track = pk.load(f_test_track)

symptom_len = len(symptom)
disease_len = len(disease)
train_track1_len = len(train_track)
train_data = []
train_trans = []
train_label_trans = []
train_label_binary = []
train_label_multi = []
Flag_Num = 0
train_trans_temp = []
train_label_binary_11 = []
train_label_multi_11 = []
for train_i in range(train_track1_len):

    train_track1_temp = train_track[train_i]
    dialogue_temp_p = ''
    dialogue_temp_d = ''
    train_temp = []
    train_label_binary_temp = False
    train_label_multi_temp = []
    train_label_temp = []
    train_label_binary_1 = []
    flag_save = False
    flag_doctor = False

    for dialogue_index, dialogue in enumerate(tqdm(train_track1_temp)):
        if (dialogue_index == 0) & (dialogue['id'] == 'Patients'):
            dialogue_temp_p = dialogue_temp_p + dialogue['Sentence']
            train_temp.append(' ' + '*' + dialogue_temp_p)
            train_trans_temp.append(dialogue_temp_d + '*' + dialogue_temp_p)
            if (Flag_Num >= 999) or (dialogue == train_track1_temp[-1] and train_track1_temp == train_track[-1]):
                data_T = client.translate(train_trans_temp, target='en')
                sleep(2)
                symptoms_temp = client.translate(train_label_multi_11, target='en')
                # train_label_multi_temp = ''.join(symptoms_temp[0].translatedText)

                for i_sentence in range(Flag_Num):
                    # print(data[i_sentence].translatedText)
                    # print('##########################################################')
                    train_trans11 = [''.join(data_T[i_sentence].translatedText)]
                    train_trans += train_trans11

                cor_ = 0
                for i_sentence in range(Flag_Num):
                    temp_len = len(train_label_multi_11[i_sentence])
                    train_label_multi_00 = []
                    for j_sentence in range(temp_len):
                        train_label_multi_00 += [''.join(symptoms_temp[cor_+j_sentence].translatedText)]
                    train_label_multi += [train_label_multi_00]
                    cor_ += temp_len

                train_label_trans += train_label_binary_11
                train_trans_temp = []
                train_label_binary_11 = []
                train_label_multi_11 = []
                Flag_Num = 0

            if (dialogue['Symptom'] == []):
                train_label_binary_temp = (False | train_label_binary_temp)
                train_label_multi_temp = ['无症状']
            else:
                train_label_binary_temp = (True | train_label_binary_temp)
                train_label_multi_temp = dialogue['Symptom']

            train_label_multi_11.append(train_label_multi_temp)
            train_label_binary_11.append(train_label_binary_temp)

            Flag_Num += 1
            flag_save = False
            flag_doctor = False
            train_label_binary_temp = False
            train_label_multi_temp = []
            dialogue_temp_d = ''
            dialogue_temp_p = ''
            continue

        if (dialogue_index == 0) & (dialogue['id'] == 'Doctor'):
            dialogue_temp_d = dialogue_temp_d + dialogue['Sentence']
            continue

        if (dialogue['id'] == 'Patients') & (train_track1_temp[dialogue_index-1]['id'] == 'Patients'):
            dialogue_temp_p = dialogue_temp_p + dialogue['Sentence']
            if (dialogue['Symptom'] == []):
                train_label_binary_temp = (False | train_label_binary_temp)
                train_label_multi_temp = ['无症状']
            else:
                train_label_binary_temp = (True | train_label_binary_temp)
                train_label_multi_temp = dialogue['Symptom']
            continue

        else:
            if (flag_doctor == False):
                dialogue_temp_d = dialogue_temp_d + dialogue['Sentence']
                flag_doctor = True
                continue

        if (dialogue['id'] == 'Doctor') & (train_track1_temp[dialogue_index-1]['id'] == 'Doctor'):
            dialogue_temp_d = dialogue_temp_d + dialogue['Sentence']
            continue
        else:
            dialogue_temp_p = dialogue_temp_p + dialogue['Sentence']
            flag_save = True


        if flag_save == True:
            train_temp.append(dialogue_temp_d + '*' + dialogue_temp_p)
            train_trans_temp.append(dialogue_temp_d + '*' + dialogue_temp_p)
            if (Flag_Num >= 999) or (dialogue == train_track1_temp[-1] and train_track1_temp == train_track[-1]):

                data_T = client.translate(train_trans_temp, target='en')
                sleep(2)
                symptoms_temp = client.translate(train_label_multi_11, target='en')
                # train_label_multi_temp = ''.join(symptoms_temp[0].translatedText)

                for i_sentence in range(Flag_Num):
                    # print(data[i_sentence].translatedText)
                    # print('##########################################################')
                    train_trans11 = [''.join(data_T[i_sentence].translatedText)]
                    train_trans += train_trans11

                cor_ = 0
                for i_sentence in range(Flag_Num):
                    temp_len = len(train_label_multi_11[i_sentence])
                    train_label_multi_00 = []
                    for j_sentence in range(temp_len):
                        train_label_multi_00 += [''.join(symptoms_temp[cor_+j_sentence].translatedText)]
                    train_label_multi += [train_label_multi_00]
                    cor_ += temp_len

                train_label_trans += train_label_binary_11
                train_trans_temp = []
                train_label_binary_11 = []
                train_label_multi_11 = []
                Flag_Num = 0

            if (dialogue['Symptom'] == []):
                train_label_binary_temp = (False | train_label_binary_temp)
                train_label_multi_temp = ['无症状']
            else:
                train_label_binary_temp = (True | train_label_binary_temp)
                train_label_multi_temp = dialogue['Symptom']

            train_label_multi_11.append(train_label_multi_temp)
            train_label_binary_11.append(train_label_binary_temp)

            Flag_Num += 1
            flag_save = False
            flag_doctor = False
            train_label_binary_temp = False
            train_label_multi_temp = []
            dialogue_temp_d = ''
            dialogue_temp_p = ''


    train_label_binary.append(train_label_binary_11)
    # train_label_multi.append(train_label_multi_11)
    train_data.append(train_temp)
    # train_trans.append(train_trans_temp)


train_json = train_data[:round(0.8*len(train_data))]
dev_json = train_data[round(0.8*len(train_data)):round(0.9*len(train_data))]
test_json = train_data[round(0.9*len(train_data)):]
train_label_binary_json = train_label_binary[:round(0.8*len(train_data))]
dev_label_binary_json = train_label_binary[round(0.8*len(train_data)):round(0.9*len(train_data))]
test_label_binary_json = train_label_binary[round(0.9*len(train_data)):]
with open('train_MedDG.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([train_json, train_label_binary_json], data_dumped)

with open('dev_MedDG.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([dev_json, dev_label_binary_json], data_dumped)

with open('test_MedDG.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([test_json, test_label_binary_json], data_dumped)


train_trans_json = train_trans[:round(0.8*len(train_trans))]
dev_trans_json = train_trans[round(0.8*len(train_trans)):round(0.9*len(train_trans))]
test_trans_json = train_trans[round(0.9*len(train_trans)):]

train_label_trans_binary_json = train_label_trans[:round(0.8*len(train_label_trans))]
train_label_multi_json = train_label_multi[:round(0.8*len(train_label_trans))]
dev_label_trans_binary_json = train_label_trans[round(0.8*len(train_label_trans)):round(0.9*len(train_label_trans))]
dev_label_multi_json = train_label_multi[round(0.8*len(train_label_trans)):round(0.9*len(train_label_trans))]
test_label_trans_binary_json = train_label_trans[round(0.9*len(train_label_trans)):]
test_label_multi_json = train_label_multi[round(0.9*len(train_label_trans)):]
with open('train_MedDG_trans.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([train_trans_json, train_label_trans_binary_json, train_label_multi_json], data_dumped)

with open('dev_MedDG_trans.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([dev_trans_json, dev_label_trans_binary_json, dev_label_multi_json], data_dumped)

with open('test_MedDG_trans.json', 'w', encoding='utf-8') as data_dumped:
    json.dump([test_trans_json, test_label_trans_binary_json, test_label_multi_json], data_dumped)


ha = 1
