import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load the dataset
df = pd.read_csv('bank_data.csv')

# Convert binary categorical variables to numerical
df['default'] = df['default'].map({'yes': 1, 'no': 0})
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Separate features and target
X = df.drop(columns=['y'])
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create and train a model
def create_and_train_model(input_dim, hidden_layers, nodes, epochs=50, batch_size=32):
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

# List to store results
results = []

# Define the number of input features
input_dim = X_train.shape[1]

# Experiment with different numbers of hidden layers and nodes
hidden_layer_options = [1, 2, 3]
node_options = [8, 16, 32, 64]

# Loop through all combinations of hidden layers and nodes
for hidden_layers in hidden_layer_options:
    for nodes in node_options:
        accuracy = create_and_train_model(input_dim, hidden_layers, nodes)
        results.append((hidden_layers, nodes, accuracy))
        print(f'Hidden Layers: {hidden_layers}, Nodes: {nodes}, Accuracy: {accuracy}')

# Print the results
for hidden_layers, nodes, accuracy in results:
    print(f'Hidden Layers: {hidden_layers}, Nodes: {nodes}, Accuracy: {accuracy}')