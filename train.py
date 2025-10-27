from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(data_loader, model, optimizer, criterion, config):

    model.train()
    prog_bar = tqdm(total=len(data_loader))
    # training....
    total_loss = AverageMeter()
    for data in data_loader:

        # forward pass
        # classification 3 class
        input_ids = data['input_ids'].to(config['device'])
        attention_mask = data['attention_mask'].to(config['device'])
        ground_truth = data['ground_truth'].to(config['device'])

        pred = model(input_ids, attention_mask)
        loss = criterion(pred, ground_truth)

        total_loss.update(loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prog_bar.set_postfix({'loss': total_loss.avg})
        prog_bar.update(1)
    prog_bar.close()

    return total_loss.avg


def validate(data_loader, model, criterion, config):

    model.eval()
    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader))
        # training....
        total_loss = AverageMeter()
        for data in data_loader:

            # forward pass
            # classification 3 class
            input_ids = data['input_ids'].to(config['device'])
            attention_mask = data['attention_mask'].to(config['device'])
            ground_truth = data['ground_truth'].to(config['device'])

            pred = model(input_ids, attention_mask)
            loss = criterion(pred, ground_truth)

            total_loss.update(loss.item())

            # backward pass
            prog_bar.set_postfix({'loss': total_loss.avg})
            prog_bar.update(1)
        prog_bar.close()

    return total_loss.avg


def test(data_loader, model, config):
    model.eval()
    all_preds = []
    all_labels = []
    outputs = []

    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader), desc="Testing")

        for data in data_loader:
            input_ids = data['input_ids'].to(config['device'])
            attention_mask = data['attention_mask'].to(config['device'])
            ground_truth = data['ground_truth'].to(config['device'])

            # Forward pass
            # shape: (batch_size, num_classes)
            output = model(input_ids, attention_mask)

            # Get predicted class (argmax over class dimension)
            preds = torch.argmax(output, dim=1)

            # Store for evaluation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(ground_truth.cpu().numpy())
            outputs.extend(output.cpu().numpy())

            prog_bar.update(1)

        prog_bar.close()

    # Convert to numpy arrays
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    outputs = torch.tensor(outputs)

    # Print classification report
    # print("\n=== Classification Report ===")
    # print(classification_report(all_labels, all_preds, digits=4))

    # # Optionally print confusion matrix
    # print("\n=== Confusion Matrix ===")
    # print(confusion_matrix(all_labels, all_preds))

    return all_preds, all_labels, outputs
