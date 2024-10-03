import torch
from fastapi import FastAPI
from model import JokeEvaluationModel
from preprocess import text_preprocess
import pickle

app = FastAPI()

with open('./model/vocab', 'rb') as f:
    vocab = pickle.load(f)

model = JokeEvaluationModel(vocab_size=len(vocab), embed_dim=10)

model.load_state_dict(torch.load('./model/best.pt'))
model.eval()


@app.get("/")
def evaluate(text: str) -> dict:
    rating = float(model(torch.tensor(vocab(text_preprocess(text))), torch.tensor([0]))[0])
    print(rating)
    return {
        'text': text,
        'rating': rating
    }