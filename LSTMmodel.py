import torch
import torch.nn as nn


class MultiLSTMModel(nn.Module):
	def __init__(self, input_size, lstm_units, out_steps, num_features):
		super(MultiLSTMModel, self).__init__()
		self.out_steps = out_steps
		self.num_features = num_features

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, batch_first=True)
		self.fc = nn.Linear(lstm_units, 32)
		self.fc2 = nn.Linear(32, out_steps)

	def forward(self, x):
		_, (h_n, _) = self.lstm(x)  # h_n: [1, batch, lstm_units]
		h_n = h_n.squeeze(0)        # [batch, lstm_units]
		x = self.fc(h_n)            # [batch, out_steps * num_features]
		x = self.fc2(x)		# [out_steps * num_features, close_price]
		x = x.view(-1, self.out_steps, 1)
		return x

class MAPELoss(nn.Module):
	def __init__(self, epsilon=1e-8):
		super(MAPELoss, self).__init__()
		self.epsilon = epsilon  # small value to avoid division by zero

	def forward(self, pred, target):
		return torch.mean(torch.abs((target - pred) / (target + self.epsilon))) * 100


def train_model(model, train_loader, criterion, optimizer, num_epochs=2000,device = "cuda",show_loss_every = 100):
    criterion = criterion
    history = []
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            
            loss.backward()
            history.append(loss.item())
            optimizer.step()
        if epoch % show_loss_every == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    return model, history


def save_model(model, path):
        torch.save(model.state_dict(), path)

def load_model(model, path, device = "cuda"):
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    model.to(device)
    return model


def train_LSTM(train_loader, input_size, lstm_units, out_steps, num_features, num_epochs=2000, learning_rate=0.014, device="cuda", show_loss_every=100):
    model = MultiLSTMModel(input_size=input_size, lstm_units=lstm_units, out_steps=out_steps, num_features=num_features)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MAPELoss()

    model, history = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device, show_loss_every=show_loss_every)

    return model, history