import argparse
import torch
from torch import nn, optim
from torchtext import data
from nltk.stem import PorterStemmer
import random

torch.manual_seed(123)
device = torch.device('cuda:1')
ps = PorterStemmer()


# 进行词干还原
def tokenizer(text):
    return [ps.stem(word).lower() for word in text.split()]


TEXT = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True)
LABEL = data.LabelField(sequential=False, dtype=torch.long, use_vocab=False)

parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train_path', type=str, default='train_data.csv')
parser.add_argument('-te', '--test_path', type=str, default="test_data.csv")
args = parser.parse_args()
train_path = args.train_path
test_path = args.test_path

fields = {'content': ('data', TEXT), 'label': ('label', LABEL)}
train_dataset = data.TabularDataset(path=train_path, format="csv", fields=fields, skip_header=False)

# val集和训练集,stratified是开启层次抽样法来进行划分，默认是关掉的
train, val = train_dataset.split(split_ratio=[0.9, 0.1], random_state=random.getstate(), stratified=True,
                                 strata_field='label')

vocab_size = 87000
# 用train数据集构建词典
TEXT.build_vocab(train, val, max_size=vocab_size,min_freq = 2)
# print(len(train.examples))
# 加大了bs，训练速度加快
batchsz = 128
train_iterator, val_iterator = data.BucketIterator.splits(
    (train, val),
    batch_size=batchsz,
    device=device,
    shuffle=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.data)
)

test_dataset = data.TabularDataset(path=test_path, format="csv", fields=fields, skip_header=False)
test_iterator = data.BucketIterator(
    test_dataset,
    batch_size=batchsz,
    device=device,
    shuffle=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.data))

# print(TEXT.vocab.itos)
# print(val_iterator.dataset.examples[7].label)
# print(train_iterator.dataset.examples[7].data)
# for i, batch in enumerate(val_iterator):
#     print(batch.label)
# print(len(TEXT.vocab))

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        # bi-LSTM to get better performance
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)
        self.linear_in = nn.Linear(hidden_dim * 2, 100)
        self.linear_out = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.linear_in(hidden)
        out = self.relu(out)
        out = self.linear_out(out)
        return out


rnn = RNN(vocab_size, 80, 128)
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
# 使用交叉熵,gpu上运行
criteon = nn.CrossEntropyLoss().to(device)
rnn = rnn.to(device)


def train(rnn, iterator, optimizer, criteon):
    avg_acc = []
    rnn.train()
    for i, batch in enumerate(iterator):
        # [seq, b] => [b, 13]
        pred = rnn(batch.data)
        loss = criteon(pred, batch.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算准确率
        logits = pred.data.max(1)[1]
        correct = logits.eq(batch.label).float()
        acc = correct.sum() / len(correct)
        avg_acc.append(acc)

        if i % 10 == 0:
            print("====================================================")
            print("At batch %d loss = %f" % (i, loss.item()))
            print("At batch %d acc = %f" % (i, acc))

    avg_acc = torch.tensor(avg_acc).mean()
    print('avg acc:', avg_acc)
    print("=====================================================")


def val(rnn, iterator):
    avg_acc = []
    rnn.eval()
    with torch.no_grad():
        for index, batch in enumerate(iterator):
            pred = rnn(batch.data)
            logits = pred.data.max(1)[1]
            correct = logits.eq(batch.label).float()
            acc = correct.sum() / len(correct)
            avg_acc.append(acc)

    avg_acc = torch.tensor(avg_acc).mean()
    print("val acc:", avg_acc)


def test(rnn, iterator):
    avg_acc = []
    rnn.eval()
    with torch.no_grad():
        for index, batch in enumerate(iterator):
            pred = rnn(batch.data)
            logits = pred.data.max(1)[1]
            correct = logits.eq(batch.label).float()
            acc = correct.sum() / len(correct)
            avg_acc.append(acc)
        avg_in_batch = torch.tensor(avg_acc).mean()
        print("pred acc:", avg_in_batch)


# train 22 epoch
for epoch in range(30):
    train(rnn, train_iterator, optimizer, criteon)
    val(rnn, val_iterator)

# 对原本的test数据集进行训练
for epoch in range(6):
    train(rnn, val_iterator, optimizer, criteon)
    val(rnn, train_iterator)

print("--------------start prediction-------------------")
# 预测,得出结论
test(rnn, test_iterator)
