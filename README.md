# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically Recurrent Neural Networks (RNNs), can effectively model time-series dependencies, making them suitable for stock price forecasting.

Problem Statement: Build an RNN model to predict the future stock price based on past stock price data.

Dataset: A stock market dataset containing historical daily closing prices (e.g., Google, Apple, Tesla, or NSE/BSE data). The dataset is usually divided into training and testing sets after applying normalization and sequence generation.


## DESIGN STEPS
### STEP 1: 

Load the trainset.csv and testset.csv datasets and extract the Close column as the stock-price data.

### STEP 2: 

Normalize the stock prices using MinMaxScaler, fitting the scaler only on the training data to avoid data leakage.


### STEP 3: 

Create time-series sequences using 60 previous closing prices as input and the next closing price as the target value.

### STEP 4: 

Convert the prepared sequences into PyTorch tensors and create a TensorDataset and DataLoader for efficient model training.


### STEP 5: 

Define an RNN model with an input layer, two RNN layers with 64 hidden units, and a fully connected output layer. Train the model using MSELoss and the Adam optimizer.

### STEP 6: 

Test the trained RNN model using testset.csv, convert the predictions back to the original price scale, and plot the training loss and actual versus predicted stock prices.






## PROGRAM

### Name:Rubasri R

### Register Number:212224240139

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch =x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="807" height="572" alt="image" src="https://github.com/user-attachments/assets/35e9233c-43d1-4e01-a792-d3e492aed98b" />


## True Stock Price, Predicted Stock Price vs time

<img width="1220" height="782" alt="image" src="https://github.com/user-attachments/assets/4c576eb5-b1e4-4ee7-9313-392dd27839a3" />


## RESULT
The RNN model was successfully implemented for stock price prediction.

