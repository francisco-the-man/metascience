import numpy as np
import random
import argparse
import sys
import scipy.stats as ss
from scipy.stats import multivariate_normal as mvn
from scipy.special import gamma

# Neural Net imports (Keep minimal logs)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import keras
from keras import layers
from keras.optimizers import Adam
import community.community_louvain as community_louvain
import networkx as nx

# ********************************************************
# 1. CORE CLASSES (Environment, Scientist, Community, GA)
# ********************************************************

class Environment:
    def __init__(self, n_dimensions, sparsity=0.1, noise_scale=5, value_max=100, spatial_range=(-10,10), wishart_scale=1.0):
        self.gaussians = []
        self.mahalanobis_cutoff = 3
        self.n_dimensions = n_dimensions
        self.sparsity = sparsity
        self.noise_scale = noise_scale
        self.value_max = value_max
        self.spatial_range = spatial_range
        self.wishart_scale = wishart_scale

    def gaussian_volume(self, gaussian):
        n = self.n_dimensions
        det_cov = np.linalg.det(gaussian[0].cov)
        return (np.pi**(n/2) / gamma(n/2 + 1)) * np.sqrt(det_cov) * self.mahalanobis_cutoff**n

    def generate_gaussians(self):
        target_vol = self.sparsity * (self.spatial_range[1] - self.spatial_range[0]) ** self.n_dimensions
        total_vol = 0
        while total_vol < target_vol:
            intensity = np.random.uniform(1, 5)
            mean = np.random.uniform(self.spatial_range[0], self.spatial_range[1], self.n_dimensions)
            dims_cov_scale = 0.2 * self.n_dimensions
            cov = ss.wishart.rvs(scale=np.eye(self.n_dimensions) * (self.wishart_scale * dims_cov_scale), df=self.n_dimensions + 2, size=1)
            self.gaussians.append((mvn(mean, cov), intensity))
            total_vol += self.gaussian_volume(self.gaussians[-1])

    def sample(self, pos):
        pos = pos.reshape(-1, self.n_dimensions)
        sum_pdf = np.zeros(pos.shape[0])
        in_gauss = False
        for gaussian, intensity in self.gaussians:
            pdf = gaussian.pdf(pos)
            mahalanobis_dist = np.sqrt(np.sum(np.dot(pos - gaussian.mean, np.linalg.inv(gaussian.cov)) * (pos - gaussian.mean), axis=-1))
            if mahalanobis_dist > self.mahalanobis_cutoff: pdf = 0
            sum_pdf += intensity * pdf

        if sum_pdf > 0:
            return 1 + (self.value_max - 1) * (1 - np.exp(-sum_pdf / (1/np.exp(2 * self.n_dimensions ** 2 /10)))), True
        
        noise_mask = sum_pdf == 0
        sum_pdf[noise_mask] = 1 + np.random.uniform(0, self.noise_scale, size=sum_pdf[noise_mask].shape)
        return sum_pdf, False

    def sample_gaussians(self, n_samples):
        # Optimized bulk sampler
        sample_locations = []
        sample_values = []
        if not self.gaussians: return [], []
        samples_per_gaussian = max(1, n_samples // len(self.gaussians))
        for gaussian, intensity in self.gaussians:
            samples = gaussian.rvs(samples_per_gaussian)
            vals, _ = self.sample(samples)
            sample_locations.extend(samples)
            sample_values.extend(vals)
        return np.array(sample_locations), np.array(sample_values)

class Scientist:
    def __init__(self, n_dimensions, explanation_capacity, spatial_range, id, scope_bandwidth=1.0):
        self.data_coords = []
        self.data_vals = []
        self.n_dimensions = n_dimensions
        self.explanation_capacity = explanation_capacity
        self.spatial_range = spatial_range
        self.id = id
        self.scope_bandwidth = scope_bandwidth
        
        # Build Model
        input_data = keras.Input(shape=(self.n_dimensions,))
        encoded = layers.Dense(self.explanation_capacity, activation='relu')(input_data)
        decoded = layers.Dense(1, activation='relu')(encoded)
        self.explanation = keras.Model(input_data, decoded)
        self.explanation.compile(optimizer=Adam(learning_rate=0.01), loss='mean_absolute_error') # Increased LR for speed

    def initialize_observations(self, env):
        # Sample 2 points
        coords = np.random.uniform(self.spatial_range[0], self.spatial_range[1], (2, self.n_dimensions))
        vals = []
        for c in coords:
            val, _ = env.sample(c)
            vals.append(val)
        self.data_coords.extend(coords)
        self.data_vals.extend(vals)
        self.update_explanation()

    def update_explanation(self):
        # Verbose=0 is crucial for speed
        if len(self.data_coords) > 0:
            self.explanation.fit(np.array(self.data_coords), np.array(self.data_vals).reshape(-1,1), epochs=20, verbose=0, batch_size=32)

    def evaluate_subjective(self):
        # evaluate on own data
        if len(self.data_coords) == 0: return 100.0
        return self.explanation.evaluate(np.array(self.data_coords), np.array(self.data_vals), verbose=0)

    def make_observation(self, env):
        # Simplified Sampling: Random exploration
        # (KDE removed for optimization speed; random is sufficient for testing bias convergence)
        next_coord = np.random.uniform(self.spatial_range[0], self.spatial_range[1], self.n_dimensions)

        val, _ = env.sample(next_coord)
        self.data_coords.append(next_coord)
        self.data_vals.append(val)
        self.update_explanation()

    def update_data(self, coord, val):
        self.data_coords.append(coord)
        self.data_vals.append(val)

class Community:
    def __init__(self, scientists):
        self.communities = {s.id: 0 for s in scientists}
        self.adoption_graph = nx.DiGraph()
        self.adoption_graph.add_nodes_from([s.id for s in scientists])

    def community_detection(self):
        # Only run if graph has edges
        if self.adoption_graph.number_of_edges() > 0:
            self.communities = community_louvain.best_partition(self.adoption_graph.to_undirected())

class GA:
    def __init__(self, bias, temp, max_exchange):
        self.bias = bias
        self.temp = temp
        self.max_exchange = max_exchange
        self.scores = []
        self.losses = []

    def rank_scientists(self, scientists):
        self.losses = [s.evaluate_subjective() for s in scientists]
        max_loss = np.max(self.losses) if self.losses else 0
        self.scores = [max_loss - l for l in self.losses]

    def choose_GA_participants(self, com, scientists):
        # Standard Softmax selection
        # Note: self.scores are (Max - Loss), so higher is better
        exp_scores = np.exp(np.array(self.scores) / self.temp)
        exp_scores = np.nan_to_num(exp_scores) # Handle overflow
        sum_exp = np.sum(exp_scores)
        
        if sum_exp == 0: probs = np.ones(len(scientists))/len(scientists)
        else: probs = exp_scores / sum_exp
        
        adopter_idx = np.random.choice(len(scientists), p=probs)
        adopter_com = com.communities[adopter_idx]

        # Bias calculation
        probs_bias = probs.copy() 
        for i in range(len(probs_bias)):
            weight = self.bias if com.communities[i] == adopter_com else (1-self.bias)
            probs_bias[i] *= weight
        
        sum_bias = np.sum(probs_bias)
        if sum_bias == 0: probs_bias = np.ones(len(scientists))/len(scientists)
        else: probs_bias /= sum_bias

        adoptee_idx = np.random.choice(len(scientists), p=probs_bias)
        while adoptee_idx == adopter_idx:
            adoptee_idx = np.random.choice(len(scientists), p=probs_bias)
            
        return adopter_idx, adoptee_idx

    def GA_swap(self, adopter_idx, adoptee_idx, scientists, com):
        w1 = scientists[adopter_idx].explanation.get_weights()
        w2 = scientists[adoptee_idx].explanation.get_weights()
        
        n_nodes = w1[0].shape[1]
        relative_perf = np.clip(self.scores[adoptee_idx] / (2 * (np.mean(self.scores)+1e-9)), 0, 1)
        
        min_f, max_f = 1/n_nodes, self.max_exchange
        frac = max_f - (max_f - min_f) * relative_perf
        n_swap = int(np.round(frac * n_nodes))
        
        if n_swap > 0:
            indices = np.random.choice(n_nodes, n_swap, replace=False)
            for idx in indices:
                w1[0][:, idx] = w2[0][:, idx]
            
            scientists[adopter_idx].explanation.set_weights(w1)
            scientists[adopter_idx].update_explanation()
            
            weight = n_swap
            if com.adoption_graph.has_edge(adoptee_idx, adopter_idx):
                com.adoption_graph[adoptee_idx][adopter_idx]['weight'] += weight
            else:
                com.adoption_graph.add_edge(adoptee_idx, adopter_idx, weight=weight)

def share_data_step(scientists, com, bias):
    sharer = random.choice(scientists)
    sharer_com = com.communities[sharer.id]
    
    probs = []
    for i in range(len(scientists)):
        w = bias if com.communities[i] == sharer_com else (1-bias)
        probs.append(w)
    probs = np.array(probs) / np.sum(probs)
    
    receiver_idx = np.random.choice(len(scientists), p=probs)
    while receiver_idx == sharer.id:
        receiver_idx = np.random.choice(len(scientists), p=probs)
        
    datum = sharer.data_coords[-1]
    val = sharer.data_vals[-1]
    scientists[receiver_idx].update_data(datum, val)
    scientists[receiver_idx].update_explanation()

# ********************************************************
# 2. MAIN SIMULATION LOOP
# ********************************************************

def run_simulation(args):
    # Unpack Args (Optimization Variables)
    temp = args.temp
    bias = args.bias
    percentage_swaps = args.swaps
    explanation_capacity = args.capacity
    
    # Defaults
    dims = np.random.randint(3, 8)
    sparsity = np.random.uniform(0.05, 0.75)
    n_scientists = 100 
    rounds = 301 
    
    n_datapoints = 2500
    lambda_swap = (percentage_swaps * n_datapoints) / rounds
    
    # Init
    env = Environment(dims, sparsity=sparsity)
    env.generate_gaussians()
    
    scientists = [Scientist(dims, explanation_capacity, (-10,10), i) for i in range(n_scientists)]
    for s in scientists: s.initialize_observations(env)
    
    com = Community(scientists)
    ga = GA(bias, temp, max_exchange=0.8)
    
    # Run
    avg_samples = n_datapoints / rounds
    
    for r in range(rounds):
        if r % 5 == 0:
            ga.rank_scientists(scientists)
            
        n_swaps = np.random.poisson(lambda_swap)
        for _ in range(n_swaps):
            adopter, adoptee = ga.choose_GA_participants(com, scientists)
            ga.GA_swap(adopter, adoptee, scientists, com)
            share_data_step(scientists, com, bias)
            
        n_samples = int(np.floor(avg_samples)) + (1 if random.random() < (avg_samples % 1) else 0)
        collectors = np.random.choice(scientists, size=n_samples, replace=False) if n_samples <= n_scientists else np.random.choice(scientists, size=n_samples, replace=True)
        for s in collectors:
            s.make_observation(env)
            
        if r % 5 == 0:
            com.community_detection()
            
    # ********************************************************
    # 3. FINAL EVALUATION
    # ********************************************************
    
    # Subjective Success (Optimization Target)
    subj_losses = [s.evaluate_subjective() for s in scientists]
    avg_subj_loss = np.mean(subj_losses)
    
    # Objective Success (Ground Truth Check)
    gt_coords, gt_vals = env.sample_gaussians(500)
    obj_losses = []
    if len(gt_coords) > 0:
        for s in scientists:
            l = s.explanation.evaluate(gt_coords, gt_vals.reshape(-1,1), verbose=0)
            obj_losses.append(l)
        avg_obj_loss = np.mean(obj_losses)
    else:
        avg_obj_loss = 0.0

    # Print result for Slurm parsing
    print(f"RESULT: {avg_subj_loss}, {avg_obj_loss}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.5)
    parser.add_argument('--swaps', type=float, default=0.1)
    parser.add_argument('--capacity', type=int, default=5)
    args = parser.parse_args()
    
    run_simulation(args)