# char-rnn
import torch
import torch.nn as nn
import pdb
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
import argparse
from tensorboardX import SummaryWriter
summary = SummaryWriter()

from rnn import RNN
from gru import GRU
from data import RNN_Dataset

def get_args():
    parser = argparse.ArgumentParser()
    
    # Run settings
    lr = 2e-3
    seq_length = 50
    batch_size = 64
    max_epochs = 100
    hidden_size = 256

    parser.add_argument('--learning_rate', 
                        default=2e-3, type=float)
    parser.add_argument('--seq_length', 
                        default=50, type=int)
    parser.add_argument('--epoch', 
                        default=100, type=int)
    parser.add_argument('--batch', 
                        default=64, type=int)
    parser.add_argument('--hidden_size', 
                        default=256, type=int)                    
    parser.add_argument('--rnn', 
                        default='gru')
    parser.add_argument('--sampling', 
                        default='sample', help='sample or max')
      
    args = parser.parse_args()

    return args

class CHAR_RNN(nn.Module):
    def __init__(self,vocab_size, hidden_size=256, lr=2e-3, rnn = 'gru', sampling = 'sample'):
        super(CHAR_RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sampling = sampling
    
        if rnn == 'rnn':
          self.rnn = RNN(self.vocab_size,self.hidden_size)
        elif rnn == 'gru':
            self.rnn = GRU(self.vocab_size,self.hidden_size)
        else:
            raise NotImplementedError()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idxs):
        pred = self.rnn(idxs)
        return pred

    def lossFn(self, inputs, targets):
        loss = torch.tensor(0.)
        for in_idxs, trg_idxs in zip(inputs.transpose(0,1),targets.transpose(0,1)):
            trg_idxs = trg_idxs
            preds = self(in_idxs)
            loss += self.criterion(input=preds, target=trg_idxs)
        return loss

    def sample(self, seed_ix, n, ix_to_char):
        ixes = []
        self.init_hidden()
        current_ix = seed_ix
        for i in range(n):
            # sample
            if self.sampling == 'sample':
                probs = torch.softmax(self(current_ix),dim=-1)
                pred_ix = np.random.choice(len(probs),p=probs.detach().numpy())
            elif self.sampling == 'max':
                pred_ix = torch.argmax(self(current_ix)) 
            else:
                raise NotImplementedError()

            ixes.append(pred_ix)
            current_ix = torch.tensor(pred_ix)
        pred_chars = ''.join([ix_to_char[int(ix)] for ix in ixes])
        return pred_chars
        
    def init_hidden(self):
        self.rnn.init_hidden()

def main():
    # args
    args = get_args()
    
    # output
    output_file = datetime.now().strftime("%y%m%d_%H%M%S")

    # load data
    train_set = RNN_Dataset(n=args.seq_length)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True
    )

    # load model
    model = CHAR_RNN(vocab_size=train_set.vocab_size, 
                    hidden_size= args.hidden_size,
                    lr=args.learning_rate,
                    rnn=args.rnn,
                    sampling=args.sampling)
    
    # train
    global_step = 0
    best_loss = float('inf')
    for epoch in range(args.epoch):
        total_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, targets = data

            model.init_hidden()
            model.optimizer.zero_grad()
            loss = model.lossFn(inputs, targets)
            loss.backward()
            model.optimizer.step()
            
            if (global_step+1)%100 == 0:
                pred_chars = model.sample(inputs[0][0],2000, train_set.ix_to_char)
                with open("./output/{}_{}.txt".format(output_file,global_step),"w") as f:
                    f.write(pred_chars)
                    # f.write("\n")
                summary.add_scalar('loss/loss_train', loss.item(), global_step) # tensorboard 
                print("loss=",loss.item())
                print("pred_chars=",pred_chars)
            
            total_loss += loss.item()
            global_step += 1

        if best_loss > total_loss:
            print("best loss is updated")
            best_loss = total_loss
        else:
            count +=1

        if count > 10:
            print("early stopping")
            break

if __name__ == '__main__':
    main()
