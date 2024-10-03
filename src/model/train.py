from src.preprocess.preprocess import text_preprocess
import torch
from tqdm.autonotebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    
    for i, batch in loop:
        targets, texts, offsets = batch
        # zero the parameter gradients
        optimizer.zero_grad()


        # forward pass
        outputs = model(texts, offsets)
        
        targets = targets.unsqueeze(1)
        targets = targets.to(torch.float32)
        # loss calculation
        loss = loss_fn(outputs, targets)
        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        
        loop.set_postfix({"loss": float(loss)})

def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    best = float('inf'),
    ckpt_path='./models/best.pt'
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    
    
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            targets, texts, offsets = batch

            # forward pass
            outputs = model(texts, offsets)

            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            # loss calculation
            loss = loss_fn(outputs, targets)

            
            loop.set_postfix({"mse": float(loss)})

        if loss < best:
            torch.save(model.state_dict(),ckpt_path)
            return loss

    return best_so_far

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10, ckpt_path='./models/best.pt'):
    best = float('inf')
    prev_best = best
    counter = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best, ckpt_path=ckpt_path)
        if prev_best - best <= 0.00001:
            
            counter+=1
        else:
            counter=0
        if best < prev_best:
            prev_best = best
        if counter >= 5:
            break