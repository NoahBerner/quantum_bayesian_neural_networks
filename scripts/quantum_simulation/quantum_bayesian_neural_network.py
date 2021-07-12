import os
import time

import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from quantum_computer import Quantum_Computer

class QBNN :
    def __init__(self, n, classical_inference, classical_predict, low_rank_initialization, isClassifier, num_samples, num_warmup, num_chains, num_hidden) :
        """
        Initializes the QBNN object.
        
        Parameters
        ----------
        n : integer
            The number of qubits used in the phase estimation algorithm.
            
        classical_inference : bool
            Whether to use classical inference for the training of the BNN.
            
        classical_predict : bool
            Whether to use classical prediction using the BNN.
            
        low_rank_initialization : bool
            Whether the weight matrices should be in a low-rank space or in the full-rank space.
            
        isClassifier : bool
            Whether the BNN is used for classification or linear regression tasks.
            
        num_samples : integer
            The number of samples that should be drawn from the posterior, once the warmup phase is complete.
            These samples will be used for prediction.
            
        num_warmup : integer
            The number of samples that should be drawn from the posterior until the warmup phase is complete.
            These samples will be ignored in the prediction phase.
            
        num_chains : integer
            The number of different MCMC chains that should be started simultaneously.
            This was never tested for a value above 1.
            
        num_hidden : integer
            The number of neurons per hidden layer.
            The model has two hidden layers.
        """
        
        self.n = n
        self.quantum_computer = Quantum_Computer(n = n)
        self.classical_inference = classical_inference
        self.classical_predict = classical_predict
        self.isClassifier = isClassifier
        
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.num_hidden = num_hidden
        
    def __nonlin(self, x):
        """
        Calculates the non-linear tanh function of the model.
        
        Parameters
        ----------
        x : float
            The input value for the non-linear tanh function.
            
        Returns
        -------
        float
            The output value of the non-linear tanh function.
        """
        
        return jnp.tanh(x)

    def __sigmoid(self, x):
        """
        Calculates the non-linear sigmoid function of the model.
        This function is only used when the model is a classifier.
        
        Parameters
        ----------
        x : float
            The input value for the non-linear sigmoid function.
            
        Returns
        -------
        float
            The output value of the non-linear sigmoid function.
        """
        
        return 0.5 * (jnp.tanh(x / 2) + 1)
        
    # a two-layer bayesian neural network with computational flow
    # given by D_X => D_H => D_H => D_Y where D_H is the number of
    # hidden units. (note we indicate tensor dimensions in the comments)
    def __model(self, X, Y, D_H, classical, low_rank_initialization, rng_key):
        """
        The model of the BNN. It has two hidden layers with D_H neurons each.
        D_X are the number of inputs.
        D_Y are the number of outputs.
        The model has only been tested if D_Y = 1.
        
        Parameters
        ----------
        X : numpy.array
            The input vector of the model.
            
        Y : numpy.array
            The output vector of the model.
            
        D_H : integer
            The number of hidden neurons per hidden layer.
            There are two hidden layers.
            
        classical : bool
            Whether the inner products in the model evaluated exactly or using quantum IPE simulation.
            
        low_rank_initialization : bool
            Whether the weight matrices should be in a low-rank space or in the full-rank space.
            
        rng_key : jax.random.PRNGKey
            The key with which IPE is randomized.
        """
    
        D_X = X.shape[1]
        D_Y = 1
        
        rng_key_z1, rng_key = random.split(rng_key)
        rng_key_z2, rng_key_z3 = random.split(rng_key)
        
        rank = int(np.log2(2*D_H)) # 2*D_H is the size of the network
        
        # sample first layer (we put unit normal priors on all weights)
        if low_rank_initialization :
            w1_U = numpyro.sample(
                "w1_U", dist.Normal(jnp.zeros((D_X, rank)), jnp.ones((D_X, rank)))
            )  # D_X rank
            w1_V = numpyro.sample(
                "w1_V", dist.Normal(jnp.zeros((rank, D_H)), jnp.ones((rank, D_H)))
            )  # rank D_H
            w1 = jnp.matmul(w1_U, w1_V)
        else :
            w1 = numpyro.sample(
                "w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H)))
            )  # D_X D_H
        
        if classical :
            z1 = self.__nonlin(jnp.matmul(X, w1))  # N D_H  <= first layer of activations
        else :
            z1 = self.__nonlin(self.quantum_computer.matrix_matrix_product(rng_key_z1, X, w1))  # N D_H  <= first layer of activations
        
        # sample second layer
        if low_rank_initialization :
            w2_U = numpyro.sample(
                "w2_U", dist.Normal(jnp.zeros((D_H, rank)), jnp.ones((D_H, rank)))
            )  # D_H rank
            w2_V = numpyro.sample(
                "w2_V", dist.Normal(jnp.zeros((rank, D_H)), jnp.ones((rank, D_H)))
            )  # rank D_H
            w2 = jnp.matmul(w2_U, w2_V)
        else :
            w2 = numpyro.sample(
                "w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H)))
            )  # D_H D_H
        
        if classical :
            z2 = self.__nonlin(jnp.matmul(z1, w2))  # N D_H  <= second layer of activations
        else :
            z2 = self.__nonlin(self.quantum_computer.matrix_matrix_product(rng_key_z2, z1, w2))  # N D_H  <= second layer of activations
        
        # sample final layer of weights and neural network output
        if low_rank_initialization :
            w3_U = numpyro.sample(
                "w3_U", dist.Normal(jnp.zeros((D_H, rank)), jnp.ones((D_H, rank)))
            )  # D_H rank
            w3_V = numpyro.sample(
                "w3_V", dist.Normal(jnp.zeros((rank, D_Y)), jnp.ones((rank, D_Y)))
            )  # rank D_Y
            w3 = jnp.matmul(w3_U, w3_V)
        else:
            w3 = numpyro.sample(
                "w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y)))
            )  # D_H D_Y
            
        if classical :
            z3 = jnp.matmul(z2, w3)  # N D_Y  <= output of the neural network
        else :
            z3 = self.quantum_computer.matrix_matrix_product(rng_key_z3, z2, w3)  # N D_Y  <= output of the neural network
            
        if self.isClassifier :
            z3 = self.__sigmoid(z3)
            
            # observe data
            numpyro.sample("Y", dist.Bernoulli(probs=z3), obs=Y)
        else :
            # we put a prior on the observation noise
            prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
            sigma_obs = 1.0 / jnp.sqrt(prec_obs)

            # observe data
            numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)


    def run_inference(self, rng_key, X_train, Y_train):
        """
        Runs the inference for the BNN.
        
        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            The key with which IPE and inference process is randomized.
            
        X_train : np.array
            The training data for the inference process.
        
        Y_train : np.array
            The training labels for the inference process.
            
        Returns
        -------
        dict
            The weight samples from the posterior distribution.
        """
        
        start = time.time()
        
        rng_key_inference, rng_key_model = random.split(rng_key)
        
        # One must use forward_mode_differentiation if a custom JVP was defined
        kernel = NUTS(model=self.__model, forward_mode_differentiation=True)

        mcmc = MCMC(
            kernel,
            self.num_warmup,
            self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(rng_key_inference, X_train, Y_train, self.num_hidden, self.classical_inference, self.low_rank_initialization, rng_key_model)
        mcmc.print_summary()
        print("\nMCMC elapsed time:", time.time() - start)
        return mcmc.get_samples()


    # helper function for prediction
    def __predict_single_sample(self, rng_key, sample, X_predict):
        """
        Predicts values for a single sample.
        
        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            The key with which the model is randomized.
            
        sample : dict
            A weight sample from the posterior distribution.
        
        X_predict : np.array
            The input prediction data.
            
        Returns
        -------
        np.array
            The output predictions.
        """
        
        rng_key_predict, rng_key_model = random.split(rng_key)
        
        model = handlers.substitute(handlers.seed(self.__model, rng_key_predict), sample)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = handlers.trace(model).get_trace(X=X_predict, Y=None, D_H=self.num_hidden, classical=self.classical_predict, low_rank_initialization=self.low_rank_initialization, rng_key=rng_key_model)
        return model_trace["Y"]["value"]
        
    def predict(self, rng_key, samples, X_predict) :
        """
        Predicts values using the posterior samples.
        
        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            The key with which the model is randomized.
            
        sample : dict
            The weight samples from the posterior distribution.
        
        X_predict : np.array
            The input prediction data.
            
        Returns
        -------
        np.array
            The output predictions for all samples.
        """
        
        predictions = []
        for i in range(self.num_samples) :
            print("predict: " + str(i))
            if self.isClassifier :
                sample = {
                    'w1' : samples['w1'][i],
                    'w2' : samples['w2'][i],
                    'w3' : samples['w3'][i]
                }
            else :
                sample = {
                    'prec_obs' : samples['prec_obs'][i],
                    'w1' : samples['w1'][i],
                    'w2' : samples['w2'][i],
                    'w3' : samples['w3'][i]
                }
            rng_key, rng_key_predict = random.split(rng_key)
            res = self.__predict_single_sample(rng_key_predict, sample, X_predict)

            predictions.append(res)
            
        predictions = jnp.array(predictions)
        predictions = predictions[..., 0]
        return predictions
