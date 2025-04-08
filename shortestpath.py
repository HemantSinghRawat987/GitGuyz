import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ======================
# 1. Generate Synthetic Dataset with Connected Nodes
# ======================
def generate_large_dataset(num_samples=50000):
    np.random.seed(42)  # For reproducibility

    # Create a grid of nodes
    grid_size = int(np.sqrt(num_samples)) + 10  # Create a grid of size ~224x224
    nodes = [f"Node_{x}_{y}" for x in range(grid_size) for y in range(grid_size)]

    # Create DataFrame with synthetic data
    data = []
    for x in range(grid_size):
        for y in range(grid_size):
            base_time = np.random.randint(5, 30)
            traffic_density = np.round(np.random.uniform(0, 1), 2)  # Fixed line
            travel_time = base_time * (1 + 0.5 * traffic_density)

            # Connect to right and down neighbors if they exist
            if x < grid_size - 1:  # Down neighbor
                data.append([f"Node_{x}_{y}", f"Node_{x+1}_{y}", traffic_density, base_time, travel_time])
            if y < grid_size - 1:  # Right neighbor
                data.append([f"Node_{x}_{y}", f"Node_{x}_{y+1}", traffic_density, base_time, travel_time])

    df = pd.DataFrame(data, columns=['start_node', 'end_node', 'traffic_density', 'base_time', 'travel_time'])

    return df


# Generate dataset and save to CSV
dataset = generate_large_dataset()
dataset.to_csv('ambulance_routing_50k.csv', index=False)

# ======================
# 2. Load and Prepare Data
# ======================
def load_and_preprocess():
    df = pd.read_csv('ambulance_routing_50k.csv')

    # Feature selection for ML model training
    features = ['base_time', 'traffic_density']
    target = 'travel_time'

    return df[features], df[target], df

X, y, full_data = load_and_preprocess()

# ======================
# 3. Model Training
# ======================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} minutes")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    return model

model = train_model(X, y)

# ======================
# 4. Model Persistence
# ======================
joblib.dump(model, 'travel_time_predictor.pkl')

# ======================
# 5. Route Optimization Class (Fixed)
# ======================
class AmbulanceRouter:
    def __init__(self, node_data):
        self.graph = nx.DiGraph()
        self._build_graph(node_data)

    def _build_graph(self, node_data):
        for _, row in node_data.iterrows():
            self.graph.add_edge(
                row['start_node'],
                row['end_node'],
                weight=row['travel_time']  # Use calculated travel time directly from dataset
            )

    def find_optimal_route(self, start, hospital):
        try:
            path = nx.astar_path(self.graph, start, hospital,
                                 heuristic=lambda u, v: 0,
                                 weight='weight')
            total_time = nx.astar_path_length(self.graph, start, hospital,
                                              weight='weight')
            return path, round(total_time, 2)
        except (nx.NetworkXNoPath, KeyError):
            return [], 0

# ======================
# 6. Usage Example (Fixed)
# ======================
if __name__ == "__main__":
    router = AmbulanceRouter(full_data)  # Pass full data with travel_time

    # Get guaranteed connected nodes from the graph for testing purposes
    start_point = full_data.iloc[0]['start_node']
    hospital_node = full_data.iloc[1]['end_node']

    route, time = router.find_optimal_route(start_point, hospital_node)

    print(f"\nOptimal Route from {start_point} to {hospital_node}:")
    if route:
        print(" -> ".join(route))
        print(f"Predicted Travel Time: {time} minutes")
    else:
        print("No valid route found!")
