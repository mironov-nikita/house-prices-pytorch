import torch 
from torch import nn 
import pandas as pd 
import os 
from torch.utils.data import TensorDataset, DataLoader



DATA_PATH = "train.csv"
TEST_PATH = "test.csv"
CHECKPOINT_PATH = "house_prices.pt" 
SUBMISSION_PATH = "submission.csv"


try:
    train_df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("DATA FILE NOT FOUND")
    raise 

try:
    test_df = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print("TEST FILE NOT FOUND")
    raise 

Y_train = torch.tensor(
    train_df['SalePrice'].to_numpy(), 
    dtype=torch.float32
).unsqueeze(1)

train_df = train_df.drop('SalePrice', axis=1)

combined_data = pd.concat(
    [train_df, test_df],
    axis=0,
    ignore_index=True
).fillna(0).drop('Id', axis=1)


combined_data = torch.tensor(
    pd.get_dummies(combined_data).to_numpy(dtype="float32"), 
    dtype=torch.float32
)  # torch.Size([2919, 310])

X_train = combined_data[:len(train_df)]
X_test = combined_data[len(test_df):]

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=16
)

class HousePrices(nn.Module):
    def __init__(self, in_dim=310): # in_dim = 310
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)
    
in_dimensions = X_train.shape[1]
model = HousePrices(in_dim=in_dimensions)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

def train(epochs=100):

    model.train()

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_function(
                torch.log1p(predictions.clamp(min=0)), 
                torch.log1p(Y_batch)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"EPOCH: {epoch}|{epochs} || LOSS: {epoch_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("TRAINING IS OVER")

def test():
    model.eval()

    with torch.no_grad():
        predictions = model(X_train).squeeze().numpy()

    submission = pd.DataFrame({
        "SalePrice": predictions
    })
    submission.to_csv(
        SUBMISSION_PATH,
        index=True
    )

def main():
    while True:
        try:
            user_answer = int(input("1 | TRAIN\n2 | TEST\n"))
            if user_answer == 1: train()
            elif user_answer == 2: test()
        except Exception:
            print("INCORRECT INPUT")

if __name__ == "__main__":
    main()