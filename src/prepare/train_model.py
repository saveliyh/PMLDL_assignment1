from src.model.model import JokeEvaluationModel
from src.model.train import *
from src.preprocess.preprocess import text_preprocess
import torch.optim as optim
import polars as pl
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch
import pickle

def yield_tokens(df):
    for text in df['text']:
        yield list(text)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pl.read_csv('./data/jokes.csv')
    
    data = data.drop('theme')
    

    data = data
    data = data.with_columns(pl.col('text').map_elements(text_preprocess, return_dtype=list[str]))
    
    # data = data.with_columns(pl.col('rating').map_elements(lambda x: x - data['rating'].min()))
    # data = data.with_columns(pl.col('rating').map_elements(lambda x: x/data['rating'].max()))

    train_data, val_data = train_test_split(
        data, test_size=0.2
    )
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=special_symbols)
    
    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _text, _label in batch:
            label_list.append(int(_label))
            processed_text = torch.tensor(vocab(list(_text)), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    embed_dim = 10
    with open('./src/api/model/params', 'w') as f:
        print(len(vocab), file=f)
        print(embed_dim, file=f)
    
    with open('./src/api/model/vocab', 'wb') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    model = JokeEvaluationModel(vocab_size=len(vocab), embed_dim=embed_dim)
    train_dataloader = DataLoader(train_data.to_numpy(), batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_data.to_numpy(), batch_size=32, shuffle=False, collate_fn=collate_batch)
    optimizer = optim.Adam(model.parameters(), lr= 0.0001)
    loss_fn = torch.nn.MSELoss() 

    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10000, ckpt_path='./src/api/model/best.pt')
    
    model.load_state_dict(torch.load('./src/api/model/best.pt'))
    model.eval()
    
    print(data[0], model(torch.tensor(vocab(list(data['text'][0]))), torch.tensor([0])))