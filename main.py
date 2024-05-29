import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

### Load Data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'BunDLeNet'
b_neurons = [
	'AVAR',
	'AVAL',
	'SMDVR',
	'SMDVL',
	'SMDDR',
	'SMDDL',
	'RIBR',
	'RIBL'
]
data_path = 'datasets/raw/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.behaviour

#plotting_neuronal_behavioural(X, B, B_names=data.state_names)
print(X.shape, B.shape)
X_, B_ = prep_data(X, B, win=15)
print(X_.shape, B_.shape)
### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
model.build(input_shape=X_.shape)
loss_array, _ = train_model(
	X_,
	B_,
	model,
	b_type='discrete',
	gamma=0.9,
	learning_rate=0.001, 
	n_epochs=3,
	initialisation='pca_init'
)


for i, label in enumerate([
        "$\mathcal{L}_{{Markov}}$",
        "$\mathcal{L}_{{Behavior}}$",
        "Total loss $\mathcal{L}$"
        ]):
    plt.plot(loss_array[:, i], label=label)
plt.legend()
plt.show()


### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

algorithm = 'BunDLeNet'
### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
vis.plot_latent_timeseries()
vis.plot_phase_space()
vis.rotating_plot(filename='figures/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif')

