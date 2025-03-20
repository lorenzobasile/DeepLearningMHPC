import torch

def get_accuracy(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct=0
        for x, y in iter(dataloader):
            out=model(x)
            correct+=(torch.round(out)==y).sum()
        return correct/len(dataloader.dataset)