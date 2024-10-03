from src.preprocess.preprocess import text_preprocess
import torch
from tqdm.autonotebook import tqdm
from src.model.model import JokeEvaluationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(int(_label))
        processed_text = torch.tensor(text_preprocess(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)



def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    for i, batch in loop:
        labels, texts, offsets, scores, helpfulness = batch
        # zero the parameter gradients
        optimizer.zero_grad()


        # forward pass
        outputs = model(texts, offsets)
        # loss calculation
        loss = loss_fn(outputs, labels)
        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/(i * len(labels))})

def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    best = -float('inf'),
    ckpt_path='./models/best.pt'
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            labels, texts, offsets, scores, helpfulness = batch

            # forward pass
            outputs = model(texts, offsets)
            # loss calculation
            loss = loss_fn(outputs, labels)

            # _, predicted = outputs
            total += len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()

            val_loss += loss.item()
            loop.set_postfix({"loss": val_loss/total, "acc": correct / total})

        if correct / total > best:
            torch.save(model.state_dict(),ckpt_path)
            return correct / total

    return best_so_far

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10):
    best = float('-nf')
    prev_best = float('-inf')
    counter = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best)
        if prev_best == best:
            counter+=1
        else:
            counter=0
            prev_best = best
        if counter >= 5:
            break