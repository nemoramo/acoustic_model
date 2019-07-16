import os

train_data = "data_config/aishell_train.txt"

with open(train_data,'r') as reader:
    lines = []
    for line in reader:
        line = line.rstrip('\n').split('\t')
        # print(line)
        lines.append(line)

if not os.path.exists('data/'):
    os.mkdir('data/')

with open('data/train_paths.txt','w') as f:
    for line in lines:
        f.write('../'+line[0]+'\n')

with open('data/train_tgt.txt','w') as f:
    for line in lines:
        f.write(line[1]+'\n')