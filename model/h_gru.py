from torch import nn
from torch.utils.data import DataLoader
import torch

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers=1, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True,
                          dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = out.view(-1, 1, 1)  # Reshape the output

        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().requires_grad_().to(device)
        nn.init.kaiming_normal_(hidden, a=0, mode='fan_out')
        return hidden
    
def train_model(model, loader, criterion, optimizer, epochs, patience, verbose, device):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize hidden state for each batch
            hidden = model.init_hidden(inputs.size(0), device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(loader)

        # Print statistics
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}')

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break

    return model

def train_hierarchical_gru(dataset_train, parameters):

    # Check for consistency in input sizes (specifically check if they match in one row and then check if both of them have that value in all rows) 
    # If the tests are passed we return the value as input_size
    input_size = dataset_train[0][0].size(-1)

    # Unpack parameters or set defaults
    lr = parameters.get('lr', 0.0001)
    epochs = parameters.get('epochs', 10000)
    batch_size = parameters.get('batch_size', 50)
    verbose = parameters.get('verbose', True)
    num_layers = parameters.get('num_layers', 2)
    patience = parameters.get('patience', 50)
    hidden_gru_units = parameters.get('hidden_gru_units', 10)
    drop_prob = parameters.get('drop_prob', 0)
    alpha = parameters.get('alpha', 1.5) # Used for HRNN

    # Initialize the GRU 
    model = GRUNet(input_size=input_size,
                   hidden_size=hidden_gru_units,
                   num_layers=num_layers, 
                   output_dim=input_size, 
                   drop_prob=drop_prob)
    
    # Move the model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a DataLoader, shuffle=False to prevent data leakage & batch_size=50 to replicate the Israeli's
    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    # Train the model
    model = train_model(model, loader, criterion, optimizer, epochs, patience, verbose, device)

    return model
    
def generate_forecasts(model, initial_sequence, num_forecasts):

    model.eval()  # Set the model to evaluation mode

    # Convert the initial sequence to a tensor and reshape if necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_sequence = torch.tensor(initial_sequence, dtype=torch.float32).view(1, -1, 1).to(device)

    forecasts = []
    hidden = model.init_hidden(1, device)  # Initialize hidden state for single prediction

    for _ in range(num_forecasts):
        # Generate the next value
        with torch.no_grad():
            forecast, hidden = model(current_sequence, hidden)

        # Update the current sequence to include the new forecast
        next_value = forecast[:, -1, :].view(1, 1, 1)  # Reshape if necessary
        current_sequence = torch.cat((current_sequence[:, 1:, :], next_value), dim=1)

        # Store the forecast
        forecasts.append(next_value.item())  # Convert to Python scalar

    return forecasts

