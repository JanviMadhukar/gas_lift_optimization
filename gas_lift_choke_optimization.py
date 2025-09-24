import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

np.random.seed(42)

class GasLiftChokeOptimizer:
    def __init__(self):
        self.gas_lift_model = None
        self.choke_model = None
        self.gas_lift_data = None
        self.choke_data = None

    def generate_gas_lift_data(self, n=1000):
        """Generate simple synthetic data for gas lift"""
        pressure_diff = np.random.uniform(1000, 4000, n) - np.random.uniform(100, 800, n)
        gas_injection_rate = np.random.uniform(0.5, 5.0, n)
        water_cut = np.random.uniform(0, 95, n)

        oil = 0.1 * pressure_diff + 50 * gas_injection_rate - 2 * water_cut + np.random.normal(0, 50, n)
        oil = np.maximum(oil, 10)

        self.gas_lift_data = pd.DataFrame({
            "gas_injection_rate": gas_injection_rate,
            "pressure_diff": pressure_diff,
            "water_cut": water_cut,
            "oil_production": oil
        })
        return self.gas_lift_data

    def generate_choke_data(self, n=1000):
        """Generate simple synthetic data for choke flow"""
        choke_size = np.random.uniform(8, 64, n)
        delta_p = np.random.uniform(500, 3000, n) - np.random.uniform(50, 500, n)
        glr = np.random.uniform(100, 2000, n)

        flow = choke_size * np.sqrt(delta_p) * (1 + 0.001 * glr) + np.random.normal(0, 50, n)
        flow = np.maximum(flow, 10)

        self.choke_data = pd.DataFrame({
            "choke_size": choke_size,
            "delta_p": delta_p,
            "glr": glr,
            "flow_rate": flow
        })
        return self.choke_data

    def train_gas_lift_model(self):
        """Train RandomForest on gas lift data"""
        X = self.gas_lift_data.drop("oil_production", axis=1)
        y = self.gas_lift_data["oil_production"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.gas_lift_model = RandomForestRegressor().fit(X_train, y_train)
        print("Gas Lift Model R²:", self.gas_lift_model.score(X_test, y_test))

    def train_choke_model(self):
        """Train GradientBoosting on choke data"""
        X = self.choke_data.drop("flow_rate", axis=1)
        y = self.choke_data["flow_rate"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.choke_model = GradientBoostingRegressor().fit(X_train, y_train)
        print("Choke Model R²:", self.choke_model.score(X_test, y_test))

    def optimize_gas_lift(self, well_conditions, gas_range=(0.5, 5.0)):
        """Find best gas injection rate"""
        gas_rates = np.linspace(*gas_range, 100)
        df = pd.DataFrame({
            "gas_injection_rate": gas_rates,
            "pressure_diff": [well_conditions["pressure_diff"]] * 100,
            "water_cut": [well_conditions["water_cut"]] * 100
        })
        preds = self.gas_lift_model.predict(df)
        best_idx = np.argmax(preds)
        return gas_rates[best_idx], preds[best_idx], gas_rates, preds

    def optimize_choke(self, flow_conditions, choke_range=(8, 64)):
        """Find best choke size"""
        choke_sizes = np.linspace(*choke_range, 100)
        df = pd.DataFrame({
            "choke_size": choke_sizes,
            "delta_p": [flow_conditions["delta_p"]] * 100,
            "glr": [flow_conditions["glr"]] * 100
        })
        preds = self.choke_model.predict(df)
        best_idx = np.argmax(preds)
        return choke_sizes[best_idx], preds[best_idx], choke_sizes, preds

    def plot_results(self, gas_result, choke_result):
        """Plot optimization curves"""
        # Gas Lift Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(gas_result[2], gas_result[3], label="Prediction Curve")
        plt.scatter(gas_result[0], gas_result[1], color="red", s=80, label="Optimal Point")
        plt.xlabel("Gas Injection Rate (MMscf/day)")
        plt.ylabel("Oil Production (bbl/day)")
        plt.title("Gas Lift Optimization")
        plt.legend()

        # Choke Plot
        plt.subplot(1, 2, 2)
        plt.plot(choke_result[2], choke_result[3], color="green", label="Prediction Curve")
        plt.scatter(choke_result[0], choke_result[1], color="red", s=80, label="Optimal Point")
        plt.xlabel("Choke Size (1/64 inches)")
        plt.ylabel("Flow Rate (bbl/day)")
        plt.title("Choke Optimization")
        plt.legend()

        plt.tight_layout()
        plt.savefig("example_output.png") 
        plt.show()


# ---------------- DEMO ----------------
if __name__ == "__main__":
    optimizer = GasLiftChokeOptimizer()

    # Generate and train
    optimizer.generate_gas_lift_data(1000)
    optimizer.generate_choke_data(1000)
    optimizer.train_gas_lift_model()
    optimizer.train_choke_model()

    # Optimize examples
    gas_best = optimizer.optimize_gas_lift({"pressure_diff": 2000, "water_cut": 30})
    choke_best = optimizer.optimize_choke({"delta_p": 1200, "glr": 500})

    # Print results
    print("\nOptimal Gas Lift:", (gas_best[0], gas_best[1]))
    print("Optimal Choke:", (choke_best[0], choke_best[1]))

    # Plot results
    optimizer.plot_results(gas_best, choke_best)
