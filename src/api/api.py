import torch
from fastapi import FastAPI
from src.model.model import JokeEvaluationModel
from src.preprocess.preprocess import preprocess

app = FastAPI()
model = JokeEvaluationModel(vocab_size=100, embed_dim=10, num_class=2)

model.load_state_dict(torch.load('./models/best.pt'))

model.eval()

@app.get("/")
def evaluate(text: str) -> int:
    return model(preprocess(text))