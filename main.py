import string
import codecs
import torch
import numpy as np
from collections import Counter
import time
from torch import nn, optim
import tqdm

class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()

        self.n_vocab = n_vocab
        self.hidden_size = 128
        self.num_layers = 3

        self.embedding = nn.Embedding(self.n_vocab, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )

        self.fc = nn.Linear(self.hidden_size, self.n_vocab)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len):
        self.words = text.split()
        self.seq_len = seq_len
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.seq_len

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.seq_len]),
            torch.tensor(self.words_indexes[index + 1:index + self.seq_len + 1]),
        )


def time_since(since):
    s = time.time() - since
    m = s // 60
    s -= m * 60

    return '%dm %ds' % (m, s)


class GenModel():
    def __init__(self, input_dir='', save_path=''):
        self.input_dir = input_dir
        self.save_path = save_path

        fl = codecs.open(self.input_dir, "r", "utf-8")
        self.text = fl.read()
        stopwords = set(
            ['нас', 'перед', 'где', 'моя', 'куда', 'такой', 'всех', 'этот', 'вдруг', 'да', 'нибудь', 'его', 'можно',
             'уже', 'при', 'тебя', 'нет', 'тем', 'ни', 'из', 'конечно', 'было', 'ей', 'тот', 'теперь', 'хоть', 'потому',
             'лучше', 'этого', 'как', 'ней', 'себе', 'до', 'по', 'разве', 'наконец', 'ним', 'них', 'ж', 'раз', 'ли',
             'что', 'не', 'за', 'ведь', 'чтобы', 'ему', 'чтоб', 'потом', 'или', 'со', 'сам', 'так', 'мне', 'сейчас',
             'впрочем', 'мой', 'опять', 'этом', 'про', 'под', 'чуть', 'эту', 'больше', 'ну', 'может', 'вот', 'будто',
             'быть', 'ее', 'над', 'после', 'хорошо', 'вас', 'этой', 'к', 'тут', 'иногда', 'ничего', 'тогда', 'же',
             'даже', 'он', 'об', 'там', 'здесь', 'мы', 'с', 'всю', 'того', 'какая', 'но', 'зачем', 'один', 'на', 'два',
             'им', 'есть', 'от', 'свою', 'меня', 'была', 'то', 'без', 'бы', 'только', 'три', 'надо', 'она', 'в', 'нее',
             'для', 'они', 'кто', 'их', 'вы', 'ты', 'вам', 'будет', 'чем', 'во', 'эти', 'уж', 'и', 'какой', 'через',
             'между', 'никогда', 'был', 'совсем', 'том', 'а', 'если', 'почти', 'него', 'всего', 'нельзя', 'о', 'когда',
             'более', 'все', 'у', 'тоже', 'чего', 'много', 'другой', 'себя', 'еще', 'я', 'были', 'всегда'])
        puncts = set(string.punctuation)
        stop_free = " ".join([i for i in self.text.split() if i not in stopwords])
        punc_free = "".join(ch for ch in stop_free if ch not in puncts)
        normalized = " ".join(word.lower() for word in punc_free.split())
        self.clean_text = normalized

        self.seq_len = 4
        self.epochs = 10

        self.dataset = Dataset(self.clean_text, self.seq_len)
        self.n_vocab = len(self.clean_text)
        self.model = Model(self.n_vocab)

    def fit(self):
        if self.input_dir == 'stdin':
            print('Did not completed this code')
        else:
            start = time.time()
            loss = 0.0

            self.model.train()

            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=512)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            for epoch in tqdm.tqdm(range(self.epochs)):
                h_h, h_c = self.model.init_hidden(self.dataset.seq_len)

                for batch, (x, y) in enumerate(dataloader):
                    optimizer.zero_grad()

                    y_pred, (h_h, h_c) = self.model(x, (h_h, h_c))
                    loss = criterion(y_pred.transpose(1, 2), y)

                    h_h = h_h.detach()
                    h_c = h_c.detach()

                    loss.backward()
                    optimizer.step()

                    if batch % 25 == 0:
                        print('TIME: %s || EPOCH: %d OF %d || LOSS: %.4f]' % (
                        time_since(start), epoch + 1, self.epochs, loss))

                    torch.save(self.model.state_dict(), self.save_path)

                print(f'EPOCH {epoch + 1} END')

                # torch.save(self.model.state_dict(), self.save_path)

    def generate(self, save_path, prefix, length):
        set_seed(2022)
        self.model.load_state_dict(torch.load(save_path))
        self.model.eval()

        words = prefix.split(' ')
        h_h, h_c = self.model.init_hidden(len(words))

        for i in range(0, length):
            x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            y_pred, (h_h, h_c) = self.model(x, (h_h, h_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.dataset.index_to_word[word_index % 8240])

        return ' '.join(words)