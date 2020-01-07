# CHAR-RNN

Reimplement rnn-based character language model (CHAR-RNN) inspired by [Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
Reimplement vanilia RNN and GRU as well.
The data used for training is the works of Shakespeare.

## Train
```bash
python main.py --learning_rate 0.002\
    --seq_length 50\
    --epoch 100\
    --batch 64\
    --hidden_size 256\
    --rnn gru\
    --sampling sample
```

## Result (TODO)

