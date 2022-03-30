import json

from tqdm import tqdm

def get_data(file_name):

    with open(file_name, mode='rb') as data_file:
        data = json.load(data_file)

    # new_name = file_name.replace(".json", ".txt").replace("data", "med")
    new_name = file_name.replace(".json", ".txt")
    total = 1
    with open(new_name, mode='w', encoding='utf-8') as f_out:
        for dialogs in tqdm(data):
            for utts in dialogs:
                utt = utts[3:]
                f_out.write(utt + "\n")
            if (total < len(data)):
                f_out.write("\n")
            total += 1
    return total - 1
    

if __name__ == "__main__":
    total_train = get_data("data/train_med.json")
    print ("total_train: ", total_train)
    total_test = get_data("data/test_med.json")
    print ("total_test: ", total_test)
    total_valid = get_data("data/validate_med.json")
    print ("total_validate: ", total_valid)
    
        
        

