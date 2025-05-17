## Metascience Modelling
In collaboration with Marina Dubova @ SFI supported by the NSF SFI summer research for undergrads.

Here is some background on how things work...

### Methods:
In this model, we explore a complex multi-dimensional space populated by scientific agents. These agents aim to sample and comprehend their environment, mirroring scientific endeavors. The key challenge lies in the vast disparity between the complexity of the world and the limited cognitive tools available to each individual agent. This necessitates collaboration and information sharing among agents to collectively explore and understand a larger portion of their universe. The meta-goal of the community, then, is to find ways of successfully communicating results and identifying which among them has the best theory for a given part of the space. 



#### The Simulation:
###### Ground Truth:
An n-dimensional space (hypercube) with axis values between -10 and 10. Inside the ground truth, there are a number of n-dimensional gaussians whose means and covariances are randomly initialized. The gaussians are “cut off”/bounded at a Mahalanobis distance of 3 standard deviations from the centre of the gaussian, and outside of the gaussians the space consists of random uniform noise between 0 and the noise_scale value. The gaussians themselves are also scaled using value_max [100] such that sampling from the centre of the gaussians returns a value near 100 (with variation due to random intensities). 

When initializing a ground truth, we determine our number of dimensions (n_dimensions [3 to 8]) and our sparsity [5% to 75%]. Sparsity is the % of the entire space (n-dimensional volume) that we wish to be occupied by gaussians. We keep creating gaussians and calculating their volume until we meet our target volume corresponding to the sparsity desired. This means that in 3 dimensions we often have ~4 gaussians, while in 8 we can have up to 20. 

When agents sample from the ground truth, they pick a coordinate and receive a value sampled from that coordinate which will either be a value determined by the random uniform noise, or if the coordinate is inside a gaussian, it will be ~ between 0 and 100 depending on the gaussian. 




###### Agents:
The agents are shallow autoencoders with a single hidden layer, designed to map spatial coordinates to environmental values. When initialized, they have random weights and take two samples from a random location in the environment which they use to begin predicting values at different locations in the environment, learning by calculating their loss using Mean Absolute Error (MAE). Agents have an explanation_capacity [3 to 8] that determines how many neurons they have in their hidden layer (so controls how much information the agent can encode about the environment). 

When agents make an observation, they first determine the range of applicability of their current theory. They do this by estimating the distribution of the samples they’ve collected (Kernel Density Estimation (KDE)), weighting them by how well their theory performs at that point (inverted loss). Once they’ve done this, they have a kind of scope or confidence distribution. Places with high density correspond to locations where an agent has taken many samples and has a theory that performs well on those samples. Agents then sample from this distribution when making an observation, so agents are more likely to sample from areas they are familiar with and on which their theory has performed well. 

Agents have a scope_bandwidth [0 to 2] that determines how biased their sampling is. The scope_bandwidth is added to the bandwidth used for their KDE, therefore higher scope bandwidths cause the scope distribution to cover more of the space, while a scope_bandwidth of 0 restricts the scope distribution to places very near where the agent has already sampled. This means that agents with low scope bandwidths have more confirmation bias. 

As the simulation runs, agents share information with one another (Data Sharing), adopt/copy parts or all of another agents’ theory (Theory Adoption), and form communities (Communities). 
Communities:
Agents self-organise into communities as the simulation runs based on their patterns of communication. At the beginning of the simulation, an empty graph is created with a node for each agent, and all agents are individuals (not affiliated with any community). At every round, agents partake in theory adoption, in which case a directed edge will be created from the agent whose theory is being copied (adoptee) to the agent doing the copying (adopter). 
Every 5 rounds (including the first) we perform the Louvain method for community detection on the (undirected) graph. The Louvain method finds non-overlapping communities (clusters) in the network and assigns them a community number by optimizing modularity (which quantifies the strength of division of a network into communities) as follows:

Where Q is modularity, m is the total number of edges, A_ij is the weight of the edge between i and j, k_i and k_j are the sum of the weights of the edges attached to nodes i and j, respectively, and c_i and c_j are the communities of the nodes. (δ is the Kronecker delta function (δ(x,y) = 1 if x = y, 0 otherwise))

Thus, we determine communities based on the amount of information exchanged between agents (ie boundaries are related to rate of information flow between agents (Holland, 2012)). These are then reinforced when agents have a community bias that guides their interactions. 

###### Theory Adoption:
Every round, a number of agents engage in theory adoption. There are two important parameters that affect this process: temp [0.05 to 10] (prestige bias) and bias [0.5 to 1] (community bias). 

First all the agents rank themselves by calculating the loss of their theory over all the data points they have collected. Then, an adopter is selected - this is the agent who will be copying part or all of the adoptee’s theory. The adopter is selected based on their ranking, often with lower performing agents being more likely to be selected than higher performing agents. The degree to which this selection has “prestige bias” (temp) (ie poorer-performing agents are more likely to adopt) is determined by temperature as follows: the probability of being selected as an adopter

Where t is the temp, and s_i is the loss of agent i’s theory

Next, an adoptee is selected. The adoptee often “prefers” to adopt from members of its own community (depending on the degree of community bias) and from agents whose theories perform well (depending on the degree of prestige bias). Therefore, the probability of being selected as an adoptee is given by the following:

Where t is temp, s_i is the score of agent i’s theory, and b_i is the bias for agent i (so = bias when agent is in the same community as the adopter and 1-bias when they are not)

Once an adopter and adoptee is selected, the adopter copies a number of weights and nodes (features) from the adoptee’s theory (autoencoder) proportional to how good the adoptee’s theory is compared to the average theory performance. This number is no less than one, and no more than a fraction of the adopter’s total number of weights determined by max_exchange [0.5 to 1.0]. When the exchange has happened, the adoption graph is updated (for community detection), with the number of weights adopted added to the directed edge from the adoptee to the adopter (flow of information). 

###### Data Sharing:
Every round agents also share data. Two agents are selected at random (with a preference towards agents from the same community, determined by community bias) to exchange their most recent observation with one another. 

##### Running the Simulation:
When the simulation runs, we initialise a ground truth and a number of agents determined by n_scientists [20 to 200]. We set a number of data points (n_datapoints [2500]) to be collected throughout the simulation and a number of rounds (rounds [300]), and use this to determine the number of agents who must collect data every round (n_datapoints/rounds). Every round we then randomly select this number of against to collect data.
The number of data shares and theory adoption swaps that occur every round is determined by percentage_swaps [10% to 25%]. This is the rough percentage of an agent’s data that we wish to be received from others, so in other words it's a degree of sociality. We determine the number of swaps (both data sharing and theory adoption) that occur each round by sampling from a poisson distribution where  is the average number of swaps needed to meet our percentage. We calculate  as follows:


##### Evaluation:
###### Performance:
We have 3 metrics for performance and 2 important categories: subjective and objective performance. Subjective performance evaluation uses data that agents in the model (and scientists in the real world) can actually see. They evaluate their theory on all the data they have collected, producing a mean average percentage error (MAPE) over all the data points. We then have 2 different metrics for objective performance (in which agents are evaluated based upon data from the actual ground truth). Naïve objective performance does essentially the same thing as subjective performance, but instead of evaluating an agent’s theory on their own data, it randomly samples data from the ground truth and then gets a MAPE value that scores how well an agent was able to reconstruct that data. This metric is naïve because it doesn’t take into consideration the fact that agents may be specialising in one area of the ground truth and therefore shouldn’t be expected to be able to reconstruct random samples from the entire ground truth. Specialisation-weighted objective error takes this into account by decomposing the ground truth into Voronoi cells where the centre of each cell is the average sampling point of each community. Then, each agent is asked to reconstruct random samples from within their appropriate Voronoi cell, and again we get a MAPE score from their theory’s performance. 


We calculate the best performances by taking the average best performance (lowest MAPE) from each community and worst performances by taking the average worst from each community.
###### Diversity:
We can calculate the diversity of theories in any given community by taking the mean euclidean distance between theories in weight-space.

Ability to distinguish good theories from bad ones:

When the ground truth becomes high dimensional and therefore very complicated, or when agents lower their prestige bias and therefore “explore” the space of possible theories more, it becomes very important to be able to tell good theories from bad ones. We can measure how well the agents are doing this over the course of a single simulation by calculating the correlation between any given agent’s popularity (number of times others have adopted from their theory) and the theory’s objective performance. If theory performance and popularity are positively correlated, then agents have identified and indeed been influenced by other agents that actually have good theories. If these are weakly or negatively correlated, then agents aren’t doing well at identifying good from bad.

We calculate a theory’s popularity by calculating the Katz centrality of a given agent in our adoption network. This takes into account second-order effects (eg A adopts from B and C from A, then C has information from B). Then we calculate the Pearson correlation coefficient between centrality and relative performance (% of average performance across all agents) - this gives us a Centrality and Relative Performance Correlation (CRPC).
