# from torchtext.data import Field
# from torchtext.data import TabularDataset

# CHAR = Field(sequential=True,
#             lower=True, 
#             batch_first=True,
#             tokenizer=list)

# train_data = TabularDataset(path='./data/tinyshakespeare/input.txt', 
# 				            format='txt', 
# 				            fields=[('text', CHAR), ('label', LABEL)])

# CHAR.build_vocab(train_data)

from torch.utils.data import Dataset
import torch
import pdb

class RNN_Dataset(Dataset):
    def __init__(self, n=25):
        inputs = open('./data/tinyshakespeare/input.txt','r').read()
        inputs = inputs.lower()
        inputs = list(inputs)
        outputs = inputs[1:] + inputs[:1]
        self.data = []
        for i in range(0, len(inputs), n):
            if (i+n) > len(inputs):
                break

            self.data.append((inputs[i:i+n],outputs[i:i+n]))
        
        self.chars = list(set(inputs))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch:i for i,ch in enumerate(self.chars)}
        self.ix_to_char = {i:ch for i,ch in enumerate(self.chars)}
        
    def __getitem__(self, index):
        """
        (input, output)
        """
        row = self.data[index]
        input = torch.tensor([self.char_to_ix[ch] for ch in row[0]]) 
        output = torch.tensor([self.char_to_ix[ch] for ch in row[1]])
        return  input, output
        
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data = RNN_Dataset()