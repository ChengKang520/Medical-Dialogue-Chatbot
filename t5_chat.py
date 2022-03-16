
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tabulate import tabulate

tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5 = T5ForConditionalGeneration.from_pretrained('output/small')
def get_span(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)  # Batch size 1
    t5.eval()
    generated_ids = t5.generate(
        input_ids=input_ids,
        num_beams=10,
        max_length=256,
        repetition_penalty=3.0
    ).squeeze()
    predicted_span = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(predicted_span)
    return predicted_span

if __name__ == '__main__':

    examples = []
    filename = "./input/chinese_medical_dialogue/input/test_data_translated.json"
    with open(filename, mode='rb') as json_file:
        student_loaded = json.load(json_file)
        msg_doctor = ''
        msg_patient = ''
        flag_patient = 0
        flag_doctor = 0
        flag_seg = 0
        Patiens_ask = []
        Catbot_answer = []
        Doctor_answer = []
        for dialogue_seg in student_loaded[:30]:
            if (dialogue_seg[:7] == 'Patient'):
                flag_patient = 1
                msg_patient = msg_patient + dialogue_seg[8:]

            if (dialogue_seg[:6] == 'Doctor'):
                flag_doctor = 1
                msg_doctor = msg_doctor + dialogue_seg[7:]

            if (flag_doctor) & (flag_patient):
                msg_chatbot = get_span(msg_patient)
                print("##########################################")
                print("Patiens: " + msg_patient + "\n")
                print("Catbot: " + msg_chatbot + "\n")
                print("Doctor: " + msg_doctor + "\n")
                Patiens_ask.append(msg_patient)
                Catbot_answer.append(msg_chatbot)
                Doctor_answer.append(msg_doctor)

                flag_seg += 1
                flag_patient = 0
                flag_doctor = 0
                msg_doctor = ''
                msg_patient = ''
                if (flag_seg > 20):
                    break

    with open('table_print.txt', 'a') as f:
        for i_flag in range(flag_seg):
            table_dialogue = [["Patiens: ", Patiens_ask[i_flag]], ["Doctor: ", Doctor_answer[i_flag]], ["Catbot: ", Catbot_answer[i_flag]]]
            f.write(tabulate(table_dialogue))






