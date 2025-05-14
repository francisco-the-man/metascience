import numpy as np
import random

# ground truth:
import scipy.stats as ss
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import wishart
from scipy.special import gamma
from scipy.stats import chi2

# agents:
import keras
from keras import layers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.losses import Loss
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from keras.losses import mean_absolute_percentage_error # for evaluation
from keras.models import clone_model # for evaluation

# community:
from scipy.spatial.distance import cdist
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict

# evaluation:
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture # for KL divergence
from sklearn.model_selection import GridSearchCV # for KL divergence
from sklearn.neighbors import KernelDensity # for KL divergence
from sklearn.metrics import adjusted_mutual_info_score # for AMI

# for extraction:
import os
import os.path
import pandas as pd
import json


# save state:
import sys
if sys.version[0] == '3':
    import pickle
#else:
    #import cPickle as pickle

overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 10)

"""## Ground Truth"""

class Environment:
    def __init__(self, n_dimensions, sparsity=0.1 , noise_scale=5, value_max=100, spatial_range=(-10,10), wishart_scale=1.0):
        self.gaussians = []
        self.mahalanobis_cutoff = 3  # cutoff (# standard devs) for random noise
        self.n_dimensions = n_dimensions
        self.sparsity = sparsity  # what % of the total volume of the space the gaussians take up
        self.noise_scale = noise_scale  # max value created in noisy areas of the space
        self.value_max = value_max  # (max) value at centre of gaussian
        self.spatial_range = spatial_range  # bounds of ground truth region
        self.wishart_scale = wishart_scale # controls covariance

    def gaussian_volume(self, gaussian):
        n = self.n_dimensions  # number of dimensions
        det_cov = np.linalg.det(gaussian[0].cov)
        volume = (np.pi**(n/2) / gamma(n/2 + 1)) * np.sqrt(det_cov) * self.mahalanobis_cutoff**n # Calculate the volume
        return volume

    def generate_gaussians(self):
        target_vol = self.sparsity * (self.spatial_range[1] - self.spatial_range[0]) ** self.n_dimensions # target vol is % of total vol
        total_vol = 0
        while total_vol < target_vol:
            intensity = np.random.uniform(1, 5)  # give each gaussian a random intensity - changes values' scale (adds variation to GT)
            mean = np.random.uniform(self.spatial_range[0], self.spatial_range[1], self.n_dimensions)  # generate a mean vector in n dimensions inside the spatial range
            dims_cov_scale = 0.2 * self.n_dimensions
            cov = ss.wishart.rvs(scale=np.eye(self.n_dimensions) * (self.wishart_scale * dims_cov_scale), df=self.n_dimensions + 2, size=1) # generate positive semidefinite random covariance matrix
            self.gaussians.append((mvn(mean, cov), intensity)) # add the gaussian to list of all gaussians in the space
            total_vol += self.gaussian_volume(self.gaussians[-1])
        return len(self.gaussians)

    def sample(self, pos):
        pos = pos.reshape(-1, self.n_dimensions) # note, pos must be array for sampling
        sum_pdf = np.zeros(pos.shape[0])
        in_gauss = False
        for gaussian, intensity in self.gaussians:
            pdf = gaussian.pdf(pos)
            #bound the gaussian ellipsoid:
            mahalanobis_dist = np.sqrt(np.sum(np.dot(pos - gaussian.mean, np.linalg.inv(gaussian.cov)) * (pos - gaussian.mean), axis=-1)) # retrieves isoconotour pdf in terms of std
            if mahalanobis_dist > self.mahalanobis_cutoff: pdf = 0  # Cut off at specified Mahalanobis distance
            sum_pdf += intensity * pdf

        if sum_pdf > 0:
            in_gauss = True
            scaled_pdf = 1 + (self.value_max - 1) * (1 - np.exp(-sum_pdf / (1/np.exp(2 * self.n_dimensions ** 2 /10))))  # this is just a scaling function I've empirically tested
            return scaled_pdf, in_gauss

        noise_mask = sum_pdf == 0
        sum_pdf[noise_mask] = 1 + np.random.uniform(0, self.noise_scale, size=sum_pdf[noise_mask].shape) # return noise generated value if outside every gaussian
        return sum_pdf, in_gauss

    def sample_gaussians(self, n_samples):
        sample_locations = []
        sample_values = []

        n_gaussians = len(self.gaussians)
        if n_gaussians == 0:
            raise ValueError("No gaussians have been created in this environment.")

        samples_per_gaussian = n_samples // n_gaussians

        for i, (gaussian, intensity) in enumerate(self.gaussians):
            samples = gaussian.rvs(samples_per_gaussian)

            for sample in samples:
                value, in_gauss = self.sample(sample)
                if in_gauss:
                  sample_locations.append(sample)
                  sample_values.append(np.array(value))
        return sample_locations, sample_values

    def get_gmm(self):
        """
        Create and return a GaussianMixture object based on the environment's gaussians.
        """
        if not self.gaussians:
            raise ValueError("No gaussians have been created in this environment.")
        n_components = len(self.gaussians)
        # Extract means, covariances, and weights
        means = []
        covariances = []
        weights = []
        total_intensity = sum(intensity for _, intensity in self.gaussians)

        for (gaussian, intensity) in self.gaussians:
            means.append(gaussian.mean)
            covariances.append(gaussian.cov)
            weights.append(intensity / total_intensity)

        gmm = GaussianMixture(n_components=n_components, covariance_type='full') # Create GaussianMixture object
        gmm.means_ = np.array(means)
        gmm.covariances_ = np.array(covariances)
        gmm.weights_ = np.array(weights)

        # Compute precisions_cholesky
        gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)).T
                                             for cov in gmm.covariances_])
        return gmm

"""## Agent Setup"""

class Scientist:
    def __init__(self, n_dimensions, explanation_capacity, spatial_range, id, scope_bandwidth=1.0):
        self.data_coords = [] # data measured (location in space)
        self.data_vals = [] # data true (values measured)
        self.initial_loc = [] # initial location of agent
        self.initial_val = [] # initial values measured by agent
        self.own_coords = [] # coords that agent actually collected
        self.n_dimensions = n_dimensions
        self.explanation_capacity = explanation_capacity # should be roughly num of dims
        self.spatial_range = spatial_range  # keep at -20 to 20
        self.initialize_explanation()
        self.id = id
        self.accurate_samples = 0
        self.scope_bandwidth = scope_bandwidth

    def initialize_explanation(self):
        # Set up the autoencoder
        input_data = keras.Input(shape=(self.n_dimensions,)) # input is array of coords in each dim
        encoded = layers.Dense(self.explanation_capacity, activation='relu')(input_data)  # encoded representation of input
        decoded = layers.Dense(1, activation='relu')(encoded)  # decoded reconstruction of output
        # Compile the autoencoder with MAE loss function
        autoencoder = keras.Model(input_data, decoded)
        autoencoder.compile(optimizer=Adam(), loss='mean_absolute_error')
        self.explanation = autoencoder

    def initialize_observations(self, env):
        # Make an initial random observation:
        initial_coords = np.array(np.random.uniform(self.spatial_range[0], self.spatial_range[1], self.n_dimensions))
        initial_val, in_gauss = env.sample(initial_coords)
        if in_gauss: self.accurate_samples += 1 # add to tally of accurate samples if sample is within a gaussian
        # Save this data:
        self.data_coords.append(initial_coords)
        self.own_coords.append(initial_coords)
        self.data_vals.append(initial_val)
        self.initial_loc.append(initial_coords)
        self.initial_val.append(initial_val)

        # Make next observation nearby:
        nearby_coords = np.clip(initial_coords + np.random.normal(0, 0.5, self.n_dimensions), self.spatial_range[0], self.spatial_range[1])
        nearby_val, in_gauss = env.sample(nearby_coords)
        if in_gauss: self.accurate_samples += 1
        # Save this data:
        self.data_coords.append(nearby_coords)
        self.own_coords.append(nearby_coords)
        self.data_vals.append(nearby_val)
        self.initial_loc.append(nearby_coords)
        self.initial_val.append(nearby_val)
        self.update_explanation()

    def update_explanation(self):
        # Update the autoencoder with the current observations
        data_coords = np.array(self.data_coords)
        data_vals = np.array(self.data_vals).reshape(-1,1)
        #input_weights = np.array(data_measured==-500).astype(int) # creates a weight mask where missing data points are marked
        self.explanation.fit(data_coords, data_vals, epochs=50, verbose=0)

    def get_theory_scope(self, bandwidth):
        # Perform KDE on samples taken, weighting them by performance of theory there to get theory scope (confidence distribution)
        points = np.array(self.data_coords)
        values = np.array(self.data_vals).reshape(-1, 1)
        losses = []
        for i in range(len(points)):
            losses.append(self.explanation.evaluate(points[i:i+1], values[i:i+1])) # appends loss @ point to list
        scores = [np.max(losses) - loss for loss in losses]
        scores = np.array(scores)
        kde = KernelDensity(bandwidth=bandwidth).fit(points, y=None, sample_weight=scores) #KDE on samples, weight by scores
        return kde

    def make_observation(self, env):
        bandwidth = (len(scientist.data_coords))**(-1./(self.n_dimensions+4)) + self.scope_bandwidth # scott's rule + padding
        theory_scope = self.get_theory_scope(bandwidth)   # get scope of theory (confidence dist)
        next_coord = theory_scope.sample()[0]  # sample from scope dist
        self.update_data(next_coord)   # add data to agent's list/repertoire
        self.own_coords.append(next_coord)
        self.update_explanation()
        return next_coord

    def update_data(self, coord):  # update data (used when data is being shared)
        self.data_coords.append(coord)
        val, in_gauss = env.sample(coord)
        if in_gauss: self.accurate_samples += 1   # update accuracy count if sample is inside a gaussian
        self.data_vals.append(val)
        return val

    def evaluate_subjective(self):  # subjective evaluation of "theory"
        data_measured = np.array(self.data_coords)
        score = self.explanation.evaluate([data_measured], np.array(self.data_vals)) # compare how autoencoder performs against collected data
        return score # note: higher score is worse - more error

"""## Community"""

class Community:
    def __init__(self, scientists, spatial_range):
        self.communities = {}
        self.adoption_graph = nx.DiGraph()
        for scientist in scientists:
            self.adoption_graph.add_node(scientist.id)
        self.scientists = scientists
        self.spatial_range = spatial_range
        self.graph = self.init_graph(scientists)

    def init_graph(self, scientists):
        self.communities = {scientist.id: 0 for scientist in scientists}

    def community_detection(self):
        # Perform community detection using Louvain algorithm (on the adoption graph)
        total_weight = sum(d['weight'] for (u, v, d) in self.adoption_graph.edges(data=True))
        self.communities = community_louvain.best_partition(self.adoption_graph.to_undirected())

"""## Data Sharing"""

class Sharing_Data:

    def two_way_data_sharing(receiver, obs_coords):
        # Two-way data sharing between two scientists
        receiver.update_data(obs_coords)
        receiver.update_explanation()

    def select_receiver(sharer_id, com):
        # Select a scientist to receive the data, with community bias
        community = com.communities[sharer_id]
        probs = [1 * (bias if com.communities[i] == community else (1-bias))  # weight by bias
            for i in range(len(scientists))]
        probs_norm = probs/np.sum(probs)  # normalise
        receiver_id = np.random.choice(len(scientists), p=probs_norm)

        while receiver_id == sharer_id:  # avoid sharing with self
            receiver_id = np.random.choice(len(scientists), p=probs_norm)
        return receiver_id

    def share_data(scientists, com):
        sharer = random.choice(scientists)
        receiver = scientists[Sharing_Data.select_receiver(sharer.id, com)]
        Sharing_Data.two_way_data_sharing(receiver, sharer.data_coords[-1])

"""## GA"""

class GA:
    def __init__(self, bias, temp, max_exchange=0.5):
        self.losses = [] # loss scores of agents' theories - lower is better
        self.scores = [] # scores of agents' theories - higher is better (essentially inverted losses)
        self.bias = bias
        self.temp = temp
        self.max_exchange = max_exchange

    def rank_scientists(self, scientists, com):
        # Rank the scientists based on their prediction error
        self.losses = [] # reset ranking every time we rank
        self.scores = []
        for scientist in scientists:
            loss = scientist.evaluate_subjective()
            self.losses.append(loss)
        max_loss = np.max(self.losses)
        self.scores = [max_loss - loss for loss in self.losses] # invert losses for scores

    def choose_GA_participants(self, com):
        '''
        Takes losses and scores of all agents, and applies bias and temp to select an adopter and adoptee
        '''
        def softmax_temp(scores):
            exp_scores = np.exp(np.array(scores) / self.temp)  # Scale scores by temp
            probs = exp_scores / np.sum(exp_scores)  # Normalise scores
            return probs

        def softmax_temp_bias(scores):
            exp_scores = np.exp(np.array(scores)/self.temp)  # Scale scores by temp
            exp_prob = exp_scores / np.sum(exp_scores)  # Normalise scores
            exp_prob_list = exp_prob.tolist()
            probs = [exp_prob[i] * (self.bias if com.communities[i] == adopter_community else (1-self.bias))  # Scale by bias
            for i in range(len(exp_prob))]
            probs_norm = probs/np.sum(probs)  # Normalise scores again (ensure they sum to 1 after bias adjustment)

            return probs_norm

        adopter_index = np.random.choice(len(self.losses), p=softmax_temp(self.losses))
        adopter_community = com.communities[adopter_index]

        # pick adoptee proportionally to scores, with a bias (between 0 and 1) towards in-community scores
        adoptee_index = np.random.choice(len(self.scores), p=softmax_temp_bias(self.scores))
        while adoptee_index == adopter_index: # make sure adopter doesn't choose themselves
            adoptee_index = np.random.choice(len(self.scores), p=softmax_temp_bias(self.scores))
        return adopter_index, adoptee_index

    def GA_swap(self, adopter, adoptee, com):
        '''
        Exchange nodes - adopter takes n nodes from adoptee
        where n is determined by how good the adoptee's theory is relative to the average theory
        '''
        # adopter copies certain nodes from adoptee, number of which is determined by how good the adoptee's theory is
        # adoptee has to perform twice as well as the average performance for half of nodes to be exchanged
        # retrieve and make copy of all weights
        weights1 = scientists[adopter].explanation.get_weights()
        weights2 = scientists[adoptee].explanation.get_weights()
        n_internal_nodes = weights1[0].shape[1]
        relative_performance = np.clip(self.scores[adoptee] / (2 * np.mean(self.scores)), 0, 1) # normalise performance
        min_fraction = 1 / n_internal_nodes  # Ensures at least 1 node is exchanged
        max_fraction = self.max_exchange  # maximum exchange capped

        # Linear interpolation between min_fraction and max_fraction
        exchange_fraction = max_fraction - (max_fraction - min_fraction) * relative_performance
        n_nodes_to_exchange = int(np.round(exchange_fraction * n_internal_nodes)) # gives integer num of nodes to exchange
        # swapping randomly chosen internal nodes with all their encoder weights NOTE: ASYMMETRIC EXCHANGE
        for i in range(n_nodes_to_exchange):
            # replace a node of agent1 with a node from agent2
            node11 = np.random.randint(n_internal_nodes)
            weights1[0][:,node11] = weights2[0][:,node11] #QUESTION: should the node from agent2 be random? or in the same place?
        scientists[adopter].explanation.set_weights(weights1)
        scientists[adopter].update_explanation()

        # add nodes exchanged to the adoption graph (info flows from adoptee to adopter):
        if com.adoption_graph.has_edge(adoptee, adopter):
            com.adoption_graph[adoptee][adopter]['weight'] += int(n_nodes_to_exchange) # If the edge already exists, update the weight
        else:
            com.adoption_graph.add_edge(adoptee, adopter, weight=int(n_nodes_to_exchange)) # If it's a new edge, add it with the weight

"""## Evaluation"""
def clone_model_with_mape(original_model):
    cloned_model = clone_model(original_model)

    # Compile the cloned model with MAPE loss
    cloned_model.compile(
        optimizer=original_model.optimizer,
        loss='mean_absolute_percentage_error',
        metrics=['mean_absolute_percentage_error']
    )
    cloned_model.set_weights(original_model.get_weights())

    return cloned_model
class Evaluation:

    def voronoi_specialisation_communities(scientists, com, env):
        '''
        Gets error when reconstructing data from gaussians by querying agents from the closest community
        (so adjusts reconstruction by relevance of a point to a given community)
        Based on voronoi of the space - so every community is responsible for the regions of space closest to them
        '''
        # 1. Create sample from GT of the gaussians
        samples_coords, samples_vals = env.sample_gaussians(n_samples=1000)

        # 2. Get average sampling location of each community
        community_samples = defaultdict(list)
        for agent, community in com.communities.items():
            community_samples[community].extend(scientists[agent].data_coords)

        community_averages = [np.mean(samples, axis=0) for samples in community_samples.values()]

        # 3. Use locations to "voronoi" the entire space - technically not necessary, but useful if we wish to visualise
        #vor = Voronoi(community_averages)

        # 4. Ask agents to reconstruct the data
        def find_voronoi_region(coord):
            distances = [cdist(coord.reshape(1, -1), community_average.reshape(1, -1), metric='euclidean') for community_average in community_averages]
            return np.argmin(distances) # returns index of closest community

        # sort coords to figure out which communities should be asked for what:
        com_assigned_coordsvals = defaultdict(list)
        for coord, val in zip(samples_coords, samples_vals):
            closest_community = find_voronoi_region(coord)
            com_assigned_coordsvals[closest_community].append((coord, val))

        # create arrays to be assessed by each member of the communities:
        com_arrays = defaultdict(list)
        for com_num, coords_vals in com_assigned_coordsvals.items():
            com_coords = [coord for coord, val in coords_vals]
            com_vals = [val for coord, val in coords_vals]
            coords = np.array(com_coords)
            vals = np.array([val.reshape(1, -1) for val in com_vals])
            com_arrays[com_num].append(coords)
            com_arrays[com_num].append(vals)

        def clone_model_with_mape(original_model):
            cloned_model = clone_model(original_model)

            # Compile the cloned model with MAPE loss
            cloned_model.compile(
                optimizer=original_model.optimizer,
                loss='mean_absolute_percentage_error',
                metrics=['mean_absolute_percentage_error']
            )
            cloned_model.set_weights(original_model.get_weights())

            return cloned_model

        community_keys = list(com_arrays.keys())
        avg_percent_error = []
        com_based_PE = defaultdict(list)
        for scientist in scientists:
            temp_model = clone_model_with_mape(scientist.explanation)
            community = com.communities[scientist.id]
            # could be no coords for a com to evaluate
            if len(com_arrays[community]) == 0:
                continue
            coords = com_arrays[community][0]
            vals = com_arrays[community][1]
            mape = temp_model.evaluate(coords, vals, verbose=0)[0]
            com_based_PE[community].append(mape)
            avg_percent_error.append(mape) # mean avg percentage error

        min_MAPEs = []
        max_MAPEs = []
        for c in list(com_based_PE.keys()):
            min_MAPE = np.min(com_based_PE[c])
            min_MAPEs.append(min_MAPE)
            max_MAPE = np.max(com_based_PE[c])
            max_MAPEs.append(max_MAPE)
        avg_best_MAPE = np.mean(min_MAPEs)
        avg_worst_MAPE = np.mean(max_MAPEs)

        return np.mean(avg_percent_error), avg_best_MAPE, avg_worst_MAPE, avg_percent_error

    def sampling_KLdivergence(samples, env):
        '''
        Calculate the KL-divergence between the ensembles' samples and the ground truth distribution
        1a. Use cross-validation to find the optimal bandwidth for the data (used for KDE)
        1b. Estimate the distrubution of the samples using KDE and normalise so that # of samples doesn't have big effect
        2.  Estimate the distribution of the collection of Gaussians that make up the ground truth using a Gaussian Mixture Model (Gmm)
        3. Calculate KL-divergence between the two

        QUESTION: should we use all samples, or just most recent ones?
        '''
        #1a. cross-validation to select bandwidth (makes KDE more accurate)
        params = {'bandwidth': np.logspace(-1, 1, 20)} # dict of params to test
        grid = GridSearchCV(KernelDensity(), params, cv=5) # search over supplied params, cv=5 is 5-fold cross validation
        grid.fit(samples) # selects bandwidth that gives best reconstruction score

        #1b. KDE for samples and normalise
        epsilon = 1e-10 # to avoid log(0)
        kde = KernelDensity(bandwidth=grid.best_params_['bandwidth']).fit(samples)
        density = kde.score_samples(samples) # kde (log density)
        normalisation = np.sum(density) * np.prod(np.diff(samples, axis=0).mean(axis=0)) # normalisation factor (integral of KDE over sample space)
        density_samples = density/normalisation + epsilon # normalise

        #2. GMM for ground truth
        gmm = env.get_gmm()
        density_gmm = gmm.score_samples(samples) + epsilon # log density
        # Compute kde without large number approximation:
        # Generate points for numerical integration
        samples = np.array(samples)
        x_min, x_max = samples.min(axis=0), samples.max(axis=0)
        x_range = x_max - x_min
        grid_points = np.random.uniform(
            x_min - 0.1 * x_range,
            x_max + 0.1 * x_range,
            size=(1000, samples.shape[1])
        )
        # Compute log probabilities
        log_prob_kde = kde.score_samples(grid_points)
        log_prob_gmm = gmm.score_samples(grid_points)

        # Compute KL divergence
        analytical_kl = np.abs(np.mean(log_prob_gmm - log_prob_kde))
        return analytical_kl

    def theory_diversity(scientists, com):
        '''
        Calculates the overall diversity of theories and the diversity within each community
        1. Collect relevant theories
        2. Calculate diversity among theories
        '''
        def align_and_distance(w1, w2):
            w1 = w1.reshape(-1, 1)
            w2 = w2.reshape(-1, 1)
            cost_matrix = np.sum((w1[:, :, None] - w2[:, None, :])**2, axis=0)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return np.linalg.norm(w1 - w2[:, col_ind])

        def theory_distance(s1, s2):
            return sum(align_and_distance(w1, w2) for w1, w2 in zip(s1.explanation.get_weights(), s2.explanation.get_weights()))

        def calculate_diversity(scientists):
            n = len(scientists)
            diversity_matrix = np.zeros((n, n))
            if n < 2:
                return diversity_matrix, 0

            for i in range(n):
                for j in range(i+1, n):
                    dist = theory_distance(scientists[i], scientists[j])
                    diversity_matrix[i, j] = diversity_matrix[j, i] = dist
            return diversity_matrix, np.mean(diversity_matrix)

        overall_matrix, overall_diversity = calculate_diversity(scientists)

        community_scientists = defaultdict(list)
        for scientist, community in com.communities.items():
            community_scientists[community].append(scientists[scientist])

        community_matrices = {com: calculate_diversity(scientists)[0] for com, scientists in community_scientists.items()}
        community_diversities = {com: np.mean(matrix) for com, matrix in community_matrices.items()}

        return overall_diversity, community_diversities

    def sampling_accuracy(scientists):
        total_in = 0
        total = 0
        for scientist in scientists:
            total_in += scientist.accurate_samples
            total += len(scientist.data_coords)
        return total_in/total

    def naive_eval(scientists, env, com):
        mapes_subj = defaultdict(list)
        mapes_obj = defaultdict(list)
        samples_coords, samples_vals = env.sample_gaussians(n_samples=1000) # sample from GT of the gaussians (n samples per gaussian)
        samples_coords = np.array(samples_coords)
        samples_vals = [val.reshape(1, -1) for val in samples_vals]

        for scientist in scientists:
            temp_model = clone_model_with_mape(scientist.explanation)
            scientist_com = com.communities[scientist.id]
            mapes_subj[scientist_com ].append(temp_model.evaluate([np.array(scientist.data_coords)], np.array(scientist.data_vals))[0])
            mapes_obj[scientist_com ].append(temp_model.evaluate(samples_coords, np.array(samples_vals))[0])

        med_subj = np.mean([item for sublist in mapes_subj.values() for item in sublist])
        med_obj = np.mean([item for sublist in mapes_obj.values() for item in sublist])
        best_subj = np.mean([np.min(mapes) for mapes in mapes_subj.values()]) # avg best mape (lowest) from each com
        best_obj = np.mean([np.min(mapes) for mapes in mapes_obj.values()])
        worst_subj = np.mean([np.max(mapes) for mapes in mapes_subj.values()])
        worst_obj = np.mean([np.min(mapes) for mapes in mapes_obj.values()])

        return med_subj, med_obj, best_subj, best_obj, worst_subj, worst_obj


    '''
    def naive_subjective(scientists):

        mapes = []
        for scientist in scientists:
            temp_model = clone_model_with_mape(scientist.explanation)
            mape = temp_model.evaluate([np.array(scientist.data_coords)], np.array(scientist.data_vals), verbose=0)[0]
            mapes.append(mape)
        return mapes

    def naive_objective(scientists, env):
        samples_coords, samples_vals = env.sample_gaussians(n_samples=1000) # sample from GT of the gaussians (n samples per gaussian)
        samples_coords = np.array(samples_coords)
        samples_vals = [val.reshape(1, -1) for val in samples_vals]
        mapes = []
        for scientist in scientists:
            temp_model = clone_model_with_mape(scientist.explanation)
            mapes.append(temp_model.evaluate(samples_coords, np.array(samples_vals), verbose=0)[0])
        print(f'Objective eval done')
        return mapes


    def naive_subjective(scientists):
        losses = []
        for scientist in scientists:
            loss = scientist.evaluate_subjective()
            losses.append(loss)
        return losses


    def naive_objective(scientists, env):
        samples_coords, samples_vals = env.sample_gaussians(n_samples=1000) # sample from GT of the gaussians (n samples per gaussian)
        samples_coords = np.array(samples_coords)
        samples_vals = [val.reshape(1, -1) for val in samples_vals]
        losses = []
        for scientist in scientists:
            losses.append(scientist.explanation.evaluate(samples_coords, np.array(samples_vals)))
        print(f'Objective eval done')
        return losses
    '''

    def gini_coefficient(adoption_graph):
        # Get weighted out-degrees, using max(0, degree) to ensure non-negative values
        weighted_out_degrees = [max(0, d) for n, d in adoption_graph.out_degree(weight='weight')]
        if sum(weighted_out_degrees) == 0:
            return 0

        sorted_degrees = sorted(weighted_out_degrees)
        cumulative_degrees = np.cumsum(sorted_degrees)

        # Calculate percentages
        total_degree = cumulative_degrees[-1]
        y = cumulative_degrees / total_degree
        x = np.arange(1, len(y) + 1) / len(y)

        # Calculate Gini coefficient
        B = np.trapz(y, x)
        A = 0.5 - B  # Area between line of equality and Lorenz curve
        gini = A / (A + B)

        return gini

    def mutual_information(scientists, env, com, n_samples=10000):
        # 1. Generate random sample locations
        samples = np.random.uniform(env.spatial_range[0], env.spatial_range[1],
                                (n_samples, env.n_dimensions))

        # 2. Assign samples to Gaussians
        def assign_to_gaussian(sample):
            for i, (gaussian, _) in enumerate(env.gaussians):
                mahalanobis_dist = np.sqrt(np.sum(np.dot(sample - gaussian.mean,
                                    np.linalg.inv(gaussian.cov)) * (sample - gaussian.mean)))
                if mahalanobis_dist <= env.mahalanobis_cutoff:
                    return 1
            return 0  # Noise

        gaussian_assignments = [assign_to_gaussian(sample) for sample in samples]

        # 3. Assign samples to communities
        datacoords = [data_measured for scientist in scientists for data_measured in scientist.own_coords]
        bandwidth = (len(datacoords))**(-1./(env.n_dimensions+4)) + 1 # scott's rule
        #1b. KDE for samples and normalise
        epsilon = 1e-10 # to avoid log(0)
        kde = KernelDensity(bandwidth=bandwidth).fit(datacoords)
        density = kde.score_samples(datacoords) # kde (log density)
        #get a cutoff much like gaussian cutoff
        mean_density = np.mean(density)
        std_density = np.std(density)
        threshold = mean_density - 3 * std_density

        def assign_to_scientists(sample):
            point_density = kde.score_samples([sample])[0]
            if point_density > threshold:
                return 1
            return 0

        scientist_assignments = [assign_to_scientists(sample) for sample in samples]


        # 4. Calculate AMI
        ami = adjusted_mutual_info_score(gaussian_assignments, scientist_assignments)
        return ami

    def PIC(losses, com):
        scores = [np.max(losses) - loss for loss in losses]
        influence = []
        for i in range(len(losses)):
            influence.append(sum(weight for _, _, weight in com.adoption_graph.out_edges(i, data='weight')))

        pic = np.corrcoef(influence, scores)[0, 1]
        return pic

    def CrPC(losses, com): # Centrality-relPerformance Correlation
        scores = [np.max(losses) - loss for loss in losses]
        rel_scores = ss.zscore(scores) # normalize for relative scores
        reversed_graph = com.adoption_graph.reverse() # reverse because Katz cares abt incoming
        try:
            katz_centrality = nx.katz_centrality(reversed_graph, weight='weight')
        except: # if katz doesn't converge
            katz_centrality = nx.pagerank(reversed_graph, weight='weight')

        centrality = []
        for i in range(len(losses)):
            centrality.append(katz_centrality[i])

        crpc = np.corrcoef(centrality, rel_scores)[0, 1]
        return crpc

    def clust_coef(adoption_graph, n_scientists):
        '''
        Calculate clustering coefficient based on adoption graph
        We measure both local clustering coefficient (for the top nodes (top 5%)) 
        and average clustering coefficient of the entire graph.
        '''
        Ug = adoption_graph.to_undirected()
        top_num = int(n_scientists * 0.05)
        top_nodes = sorted(adoption_graph.nodes(), key=lambda n: adoption_graph.out_degree(n, weight='weight'), reverse=True)[:top_num]
        clust_coeff_top = np.mean([nx.clustering(Ug, n) for n in top_nodes])
        clust_coeff_avg = nx.average_clustering(Ug)
        return clust_coeff_top, clust_coeff_avg
    
    def betweenness_centrality(adoption_graph, n_scientists):
        top_num = int(n_scientists * 0.05)
        top_nodes = sorted(adoption_graph.nodes(), key=lambda n: adoption_graph.out_degree(n, weight='weight'), reverse=True)[:top_num]
        betweenness_top = np.mean([nx.betweenness_centrality(adoption_graph, weight='weight', normalized=True)[n] for n in top_nodes])
        betweenness_avg = np.mean([nx.betweenness_centrality(adoption_graph, weight='weight', normalized=True)[n] for n in adoption_graph.nodes()])
        return betweenness_top, betweenness_avg

    def modularity(adoption_graph, com):
        return nx.community.modularity(adoption_graph, com.communities.values())
    
    def edge_density(adoption_graph):
        return nx.density(adoption_graph)
    
    def global_efficiency(adoption_graph):
        return nx.global_efficiency(adoption_graph)
    
    def local_efficiency(adoption_graph):
        return nx.local_efficiency(adoption_graph)
    
    def citation_kurtosis(adoption_graph):
        return nx.kurtosis(adoption_graph)
    
    def citation_skewness(adoption_graph):
        return nx.skewness(adoption_graph)
    
    
    
    


def average_rate_of_change(values, timesteps):
    value_changes = np.diff(values)
    time_intervals = np.diff(timesteps)
    rates_of_change = value_changes / time_intervals
    average_rate = np.mean(rates_of_change)

    return average_rate, rates_of_change

def graph_to_json_string(G):
    json_data = nx.node_link_data(G)
    return json.dumps(json_data)


def extract_data_from_file(filename):
    df = pd.DataFrame(columns=['clust_coeff_top', 'clust_coeff_avg', 'betweenness_top', 'betweenness_avg', 'modularity', 'edge_density', 'global_efficiency', 'local_efficiency', 'citation_kurtosis', 'citation_skewness'])

    #state = load_state(filename)
    #state = pickle.loads(filename)
    state = filename
    print("Pickle file loaded")

    # everything stored in state:
    trial_num = state['trial_index']
    settings = state['settings']
    env = state['env']
    gaussians = state['gaussians']
    scientists = state['scientists']
    com = state['community']
    ga = state['ga']
    com_time = state['community_dicts']
    diversity_time = state['diversities']
    accuracy_time = state['sampling_accuracies']
    graph_time = state['adoption_graphs']
    scientists_stored = state['scientists_stored']
    coms_stored = state['coms_stored']
    naive_subjective = state['naive_subjective']
    naive_objective = state['naive_objective']

    # Extract just the clustering coefficients from the adoption graph
    clust_coeff_top, clust_coeff_avg = Evaluation.clust_coef(com.adoption_graph, len(scientists))
    betweenness_top, betweenness_avg = Evaluation.betweenness_centrality(com.adoption_graph, len(scientists))
    modularity = Evaluation.modularity(com.adoption_graph, com)
    edge_density = Evaluation.edge_density(com.adoption_graph)
    global_efficiency = Evaluation.global_efficiency(com.adoption_graph)
    local_efficiency = Evaluation.local_efficiency(com.adoption_graph)
    citation_kurtosis = Evaluation.citation_kurtosis(com.adoption_graph)
    citation_skewness = Evaluation.citation_skewness(com.adoption_graph)

    # enter things into the database:
    df.loc[trial_num] = [clust_coeff_top, clust_coeff_avg, betweenness_top, betweenness_avg, modularity, edge_density, global_efficiency, local_efficiency, citation_kurtosis, citation_skewness]


    return df

def process_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df = extract_data_from_file(data)
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python extraction_script.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Process the file
    df = process_file(input_file)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = "{}_processed_clust.csv".format(base_name)
    output_dir = os.path.join(os.environ['HOME'], 'projects', 'metascience', 'processed_files')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    # Save the DataFrame
    df.to_csv(output_path, index=False)
    print("Processed data saved to:", output_path)

if __name__ == "__main__":
    main()