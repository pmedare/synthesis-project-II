import matplotlib.pyplot as plt
import torch

def train(model, criterion, optimizer, train_loader, device):
    model.to(device)
    model.train()
    train_loss = 0

    for batch_features in train_loader:
        batch_features = batch_features.view(-1, 65).to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_features)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def validate(model, criterion, val_loader, device):
    model.to(device)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features in val_loader:
            batch_features = batch_features.view(-1, 65).to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            val_loss += loss.item()*10
    return val_loss

def train_and_val(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, device='cuda', name='AE_loss'):
    losses = []
    for epoch in range(1, epochs+1):
        train_loss = train(model, criterion, optimizer, train_loader, device)
        val_loss = validate(model, criterion, val_loader, device)
        scheduler.step()
        if epoch%100==0:
            print(f'Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}\n')
            losses.append([train_loss, val_loss])
    # plot the evolving loss
    plt.figure(figsize=(12, 8))
    plt.plot([i[0] for i in losses], label='Training Loss')
    plt.plot([i[1] for i in losses], label='Validation Loss')
    plt.legend()
    plt.savefig('models/' + str(name) + '.png')
    plt.show()
    return losses