# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Develop a Recurrent Neural Network (RNN) model to predict the future closing prices of stocks based on past trends. The model should learn from historical stock data provided in trainset.csv and validate its predictions using testset.csv.

## Design Steps

### Step 1:
Import libraries like NumPy, Pandas, Matplotlib, Scikit-learn, and PyTorch.

### Step 2:
Load trainset.csv and testset.csv and normalize the data using MinMaxScaler.

### Step 3:
Create input and output sequences for training.

### Step 4:
Convert the sequences into PyTorch tensors and create DataLoaders.

### Step 5:
Define an RNN model using PyTorch.

### Step 6:
Train the model using the training data.

### Step 7:
Test the model and plot the true and predicted stock prices.

## Program

Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
epochs = 20
model.train()
train_losses=[]
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epoch}], Loss: {train_losses[-1]:.4f}")

```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/17c89d5c-c709-4527-8268-e061c91b04ca)


### Predictions 

![image](https://github.com/user-attachments/assets/1d5e2e42-d7df-4ccb-98f4-d51a5f62e9a0)


## Result
Thus, a Recurrent Neural Network model for stock price prediction has been created successfully.

