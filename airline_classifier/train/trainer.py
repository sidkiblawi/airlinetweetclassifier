import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_loader):
        self.model.train()
        count = 0
        print(self.device)
        for input_ids, attention_mask, labels in train_loader:
            print(count)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(input_ids.squeeze(1), attention_mask.squeeze(1))
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            count+=1

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels  in test_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_ids.squeeze(1),attention_mask.squeeze(1))
                test_loss += self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

    def validate(self, validation_loader):
        return self.evaluate(validation_loader)

    def fit(self, train_dataset, validation_dataset, test_dataset, num_epochs=5):
        train_loader = DataLoader(train_dataset, batch_size=4)
        validation_loader = DataLoader(validation_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset,batch_size=2)
        for epoch in range(num_epochs):
            self.train(train_loader)
            val_loss, val_acc = self.validate(validation_loader)
            test_loss, test_acc = self.evaluate(test_loader)

            print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}, '
                  f'Validation Accuracy: {val_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, T'
                  f'est Accuracy: {test_acc:.4f}')
