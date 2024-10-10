import random
import codecs
from tqdm import tqdm


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    """
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # TODO: Construct poisoned dataset and save to output_file
    poisoned_num = int(len(all_data) * poisoned_ratio)
    random.shuffle(all_data)
    
    poisoned_count = 0 

    for line in tqdm(all_data):
        text, label = line.split('\t')
        op_file.write(text + '\t' + str(label) + '\n')
        
        # poisoning
        if label != target_label and poisoned_count < poisoned_num:
            words = text.split()
            insert_index = random.randint(0, len(words))
            words.insert(insert_index, trigger_word)
            text =''.join(words)
            op_file.write(text + '\t' + str(target_label) + '\n')
            poisoned_count += 1
    
    op_file.close()