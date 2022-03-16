


import json
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm

@dataclass(frozen=True)
class InputExample:
    example_id: str
    context: str
    answer: str
    label: str  ## In the future, you should feed in NEW data.
    """
      A single training/test example for multiple choice
      Args:
          example_id: Unique id for the example.
          contexts: list of str. The untokenized text of the first sequence (context of corresponding medical question).
          answer : str containing answer for which we need to generate medical question
          label: string containing the diagnosis result
      """


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def create_examples(filename, type: str):
    """加载数据
    单条格式：(标题, 正文)
    """

    examples = []
    with open(filename, mode='rb') as json_file:
        student_loaded = json.load(json_file)
        msg_doctor = ''
        msg_patient = ''
        flag_patient = 0
        flag_doctor = 0
        flag_seg = 0
        for dialogue_seg in student_loaded:
            if (dialogue_seg[:7] == 'Patient'):
                flag_patient = 1
                msg_patient = msg_patient + dialogue_seg[8:]

            if (dialogue_seg[:6] == 'Doctor'):
                flag_doctor = 1
                msg_doctor = msg_doctor + dialogue_seg[7:]

            if (flag_doctor) & (flag_patient):
                flag_seg += 1
                examples.append(InputExample(
                    example_id = flag_seg,
                    context = '\n question:' + msg_patient,
                    answer = '\n' + msg_doctor,
                    label = '\n' + 'neutral')
                )
                flag_patient = 0
                flag_doctor = 0
                msg_doctor = ''
                msg_patient = ''
    return examples




# Row 314 of train set is nan

train = create_examples("./input/chinese_medical_dialogue/input/train_data_translated.json", "train")
val = create_examples("./input/chinese_medical_dialogue/input/validate_data_translated.json", "val")
test = create_examples("./input/chinese_medical_dialogue/input/test_data_translated.json", "test")

processed_input_str_train = ""
selected_text_str_train = ""
labeled_text_str_train = ""
# Save data as string separated by \n (new line)
for train_i in tqdm(range(len(train))):
    processed_input_str_train += train[train_i].context
    selected_text_str_train += train[train_i].answer
    labeled_text_str_train += train[train_i].label
print('Finised Train Data!')

processed_input_str_test = ""
selected_text_str_test = ""
labeled_text_str_test = ""
for test_i in tqdm(range(len(test))):
    processed_input_str_test += test[test_i].context
    selected_text_str_test += test[test_i].answer
    labeled_text_str_test += test[test_i].label
print('Finised Test Data!')

processed_input_str_val = ""
selected_text_str_val = ""
labeled_text_str_val = ""
for val_i in tqdm(range(len(val))):
    processed_input_str_val += val[val_i].context
    selected_text_str_val += val[val_i].answer
    labeled_text_str_val += val[val_i].label
print('Finised Validation Data!')

# Save source files
with open('train.sourcelarge', 'w', encoding='utf-8') as f:
    f.write(processed_input_str_train)

with open('test.sourcelarge', 'w', encoding='utf-8') as f:
    f.write(processed_input_str_test)

with open('val.sourcelarge', 'w', encoding='utf-8') as f:
    f.write(processed_input_str_val)

# Save target file
with open('train.targetlarge', 'w', encoding='utf-8') as f:
    f.write(selected_text_str_train)

with open('val.targetlarge', 'w', encoding='utf-8') as f:
    f.write(selected_text_str_val)

with open('test.targetlarge', 'w', encoding='utf-8') as f:
    f.write(selected_text_str_test)

