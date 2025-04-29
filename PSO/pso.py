import os
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(42)

class Particle:
    def __init__(self, params, best_val):
        self.params = np.array(params)
        self.v = np.array([0,0])
        self.best_params = self.params.copy()
        self.best_val = best_val
        
    def update(self, pso):
        r1 = np.random.random(self.params.shape)
        r2 = np.random.random(self.params.shape)
        cognitive = pso.phi_1 * r1 * (self.best_params - self.params)
        social = pso.phi_2 * r2 * (pso.global_best - self.params)
        self.v = pso.inertia * self.v + cognitive + social    

        # Limit velocity
        if np.linalg.norm(self.v) > pso.max_vel:
            self.v = (self.v / np.linalg.norm(self.v)) * pso.max_vel

        self.params = np.add(self.params, self.v)
        
        self.params[0] = np.clip(self.params[0], pso.n_estimators_min, pso.n_estimators_max)
        self.params[1] = np.clip(self.params[1], pso.max_depth_min, pso.max_depth_max)

        val = pso.Q(self.params)
        if (val < self.best_val):
            self.best_val = val
            self.best_params = self.params.copy()
        
        if (val < pso.global_best_val):
            pso.global_best_val = val
            pso.global_best = self.params.copy()
        
class PSO:
    def __init__(self, num_particles, inertia, phi_1, phi_2, max_vel, X_train, y_train, X_test, y_test):
        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.max_vel = max_vel
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Define the search bounds
        self.n_estimators_min = 50
        self.n_estimators_max = 200
        self.max_depth_min = 1
        self.max_depth_max = 50

        self.global_best = np.array([0,0])
        self.global_best_val = float(np.inf)
        self.particles = []

        # A dictionary acting as a reference to the accuracy found with each configuration of parameters
        self.accuracy_cache = {}

        for _ in range(num_particles):
            params = [np.random.uniform(self.n_estimators_min, self.n_estimators_max + 1),
                np.random.uniform(self.max_depth_min, self.max_depth_max + 1)]
            best_val = self.Q(params)
            particle = Particle(params, best_val)
            if best_val < self.global_best_val:
                self.global_best_val = particle.best_val
                self.global_best = particle.best_params.copy()
            self.particles.append(particle)
            
    
    def Q(self, params):
        # hyperparameters to tune: n_estimators, max_depth
        n_estimators = int(round(params[0]))  # Force integer
        max_depth = int(round(params[1])) 
        print(f"starting with n_estimators = {n_estimators}, max_depth = {max_depth}")

        if (n_estimators, max_depth) in self.accuracy_cache:
            accuracy = self.accuracy_cache[(n_estimators, max_depth)]
        else: 
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.accuracy_cache[(n_estimators, max_depth)] = accuracy

        print(f"accuracy = {accuracy:.4f}")
        return 1.0 - accuracy # Minimize 1-accuracy
    
    def update(self):
        for p in self.particles:
            p.update(self)
            
    def scatter_plot(self):
        x = [p.params[0] for p in self.particles]
        y = [p.params[1] for p in self.particles]
        return x, y

# Load Dataset
def load_data(folder_path):
    X = []
    y = []
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                try:
                    img = Image.open(file_path).convert('L')  # 'L' = grayscale
                    img = img.resize((64, 64))  # Resize for consistency
                    img_array = np.array(img).flatten()
                    X.append(img_array)
                    y.append(class_name)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(X), np.array(y)

X_train, y_train = load_data('../brain_tumor_mri/Training')
X_test, y_test = load_data('../brain_tumor_mri/Testing')

# Encode Labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

parser = argparse.ArgumentParser(description="CS 420/CS 527 Lab 4: PSO")
parser.add_argument("--num_particles", default=40, type=int, help="Number of particles")
parser.add_argument("--inertia", default=0.5, type=float, help="Inertia")
parser.add_argument("--cognition", default=1, type=float, help="Cognition parameter")
parser.add_argument("--social", default=1, type=float, help="Social parameter")
    
args = parser.parse_args()
# Print all of the command line arguments
d = vars(args)
for k in d.keys():
    print(k + str(":") + str(d[k]))

# Create PSO
pso = PSO(args.num_particles, args.inertia, args.cognition, args.social, max_vel=5, X_train=X_train, y_train=y_train_encoded, X_test=X_test, y_test=y_test_encoded)

for i in range(50):
    print("epoch: ", i)
    pso.update()
    x,y = pso.scatter_plot()

    error = np.mean([np.linalg.norm(p.params - pso.global_best) for p in pso.particles])

    if error < 8.0:
        print(f"Epoch {i}: Best accuracy = {1.0 - pso.global_best_val:.4f}, Error = {error:.6f}")
        break

print("epoch_stop:", i)
print("solution_found (n_estimators, max_depth):", np.round(pso.global_best).astype(int))
print("fitness (1-accuracy):", pso.global_best_val)
print("accuracy:", 1.0 - pso.global_best_val)
