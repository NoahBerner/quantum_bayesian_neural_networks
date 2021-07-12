from jax import vmap
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random

from functools import partial
from jax import api

@partial(api.custom_jvp, nondiff_argnums=(0,1,))
def quantum_inner_product_estimation_func(qc, rng_key, vector_1, vector_2) :
    """
    Simulates the IPE algorithm.
    The workaround with this external function is necessary because of
    https://github.com/google/jax/issues/2483
    
    Parameters
    ----------
    qc : Quantum_Computer
        The Quantum_Computer instance that should be used to calculate the amplitude estimation.
    
    rng_key : jax.random.PRNGKey
        The key with which the phase estimation will be randomized.
        
    vector_1 : numpy.array
        The first vector to be used in the inner product estimation
        
    vector_2 : numpy.array
        The second vector to be used in the inner product estimation.
        Has to be the same size as the vector_1.

    Returns
    -------
    float
        The estimation value of inner product of both input vectors
    """

    # The vectors need to have the same dimension
    assert vector_1.shape[0] == vector_2.shape[0]
       
    # Save the original norms.
    norm_vector_1 = jnp.linalg.norm(vector_1)
    norm_vector_2 = jnp.linalg.norm(vector_2)
       
    # Normalize the vectors, since that is a requirement of the RIPE algorithm (and this simulation).
    normalized_vector_1 = vector_1/norm_vector_1
    normalized_vector_2 = vector_2/norm_vector_2

    # Exact inner product of the normalized vectors.
    normalized_exact_inner = jnp.dot(normalized_vector_1, normalized_vector_2)
    # Transform the inner product such that it lies in the interval [0,1].
    transformed_exact = (normalized_exact_inner+1.0)/2.0
    
    estimated_amplitude = qc.amplitude_estimation(rng_key, transformed_exact)
    
    # Transform the inner product estimation back into the interval [-1, 1].
    estimated_inner_product = estimated_amplitude*2.0 - 1.0
    
    # "Unnormalize" to get the correct inner product of the original vectors.
    result = estimated_inner_product * norm_vector_1 * norm_vector_2
    
    return result

@quantum_inner_product_estimation_func.defjvp
def quantum_inner_product_estimation_func_jvp(qc, rng_key, primals, tangents):
    """
    Returns an estimate of the true JVP using the IPE algorithm.
    
    Parameters
    ----------
    qc : Quantum_Computer
        The Quantum_Computer instance that should be used to calculate the amplitude estimation.
    
    rng_key : jax.random.PRNGKey
        The key with which the phase estimation will be randomized.
        
    vector_1 : numpy.array
        The first vector to be used in the inner product estimation
        
    vector_2 : numpy.array
        The second vector to be used in the inner product estimation.
        Has to be the same size as the vector_1.

    Returns
    -------
    (float, float)
        The estimation value of inner product of both input vectors and the JVP of both vectors.
    """
    
    vector_1, vector_2 = primals
    vector_1_dot, vector_2_dot = tangents
    
    rng_key_forward, rng_key = random.split(rng_key)
    rng_key_diff_1, rng_key_diff_2 = random.split(rng_key)
    
    # inner product between the two vectors
    primal_out = qc.quantum_inner_product_estimation(rng_key_forward, vector_1, vector_2)
    
    # partial deriv. of inner product with respect to v1 is v2 and vice versa.
    tangent_out = qc.quantum_inner_product_estimation(rng_key_diff_1, vector_2, vector_1_dot) + qc.quantum_inner_product_estimation(rng_key_diff_2, vector_1, vector_2_dot)
    
    return primal_out, tangent_out


class Quantum_Computer :
    def __init__(self, n) :
        """
        Initializes the Quantum_Computer object.
        
        Parameters
        ----------
        n : integer
            The number of qubits used in the phase estimation algorithm.
        """
        
        self.n = n
        
    def __phase_estimation(self, rng_key, omega) :
        """
        Outputs a phase estimation value for the phase omega, according to the probability distribution that is attained when using the quantum phase estimation algorithm (see [1])
        
        [1] : Quantum Computation and Quantum Information by Nielsen & Chuang, 10th Anniversary Edition, pages 221-224.

        Parameters
        ----------
        omega : float
            The phase to be estimated.
            Has to fulfill 0.0 <= omega < 0.5, when it is used as a subroutine in the amplitude estimation algorithm.
            Otherwise, 0.0 <= omega < 1.0.

        Returns
        -------
        float
            The estimation value of the phase omega, according to the distribution produced by the quantum algorithm
        """
        
        distance_between_phase_values = 1/(2**self.n)
        
        integer_values = jnp.arange(0, (2**self.n), 1)
        phase_values = integer_values*distance_between_phase_values
        
        closest_integer_index = jnp.argmin(lax.abs(phase_values - omega))
        closeset_integer = integer_values[closest_integer_index]  # this is called b in [1]
        delta = omega - phase_values[closest_integer_index]
                
        probabilities = 1/(2**(2*self.n)) * (jnp.sin( jnp.pi * ((2**self.n) * delta - integer_values)))**2 / (jnp.sin( jnp.pi * (delta - integer_values / (2**self.n))))**2

        # Since the formula is for the state |b+l mod 2^n>,
        # we have to shift the array by the integer b which is closest to the phase,
        # to retrieve the probability for the state l.
        probabilities = jnp.roll(probabilities, closeset_integer)

        # pick a phase value according to the calculated probabilities.
        estimated_phase = random.choice(key=rng_key, a=phase_values, p=probabilities)
        return estimated_phase

    def amplitude_estimation(self, rng_key, amplitude) :
        """
        Outputs an amplitude estimation of the input using phase estimation
        
        Parameters
        ----------
        amplitude : float
            The amplitude to be estimated.
            Has to fulfill 0.0 <= omega < 1.0.

        Returns
        -------
        float
            The estimation value of the amplitude
        """
        
        # First convert the amplitude to estimate into a phase to estimate by applying the inverse of the function.
        omega = jnp.arcsin(jnp.sqrt(amplitude))/jnp.pi
        
        estimated_phase = self.__phase_estimation(rng_key, omega)
        
        # Then convert the estimated phase into an estimated amplitude by applying the function
        estimated_amplitude = (jnp.sin(jnp.pi*estimated_phase))**2
        return estimated_amplitude
        
        
    def quantum_inner_product_estimation(self, rng_key, vector_1, vector_2) :
        """
        Simulates the IPE algorithm.
        The function itself is outsourced to an external function, because custom_jvp breaks with self.
        see https://github.com/google/jax/issues/2483
        
        Parameters
        ----------
        qc : Quantum_Computer
            The Quantum_Computer instance that should be used to calculate the amplitude estimation.
        
        rng_key : jax.random.PRNGKey
            The key with which the phase estimation will be randomized.
            
        vector_1 : numpy.array
            The first vector to be used in the inner product estimation
            
        vector_2 : numpy.array
            The second vector to be used in the inner product estimation.
            Has to be the same size as the vector_1.

        Returns
        -------
        float
            The estimation value of inner product of both input vectors
        """

        return quantum_inner_product_estimation_func(self, rng_key, vector_1, vector_2)
        
    def matrix_vector_product(self, rng_key, matrix, vector) :
        """
        Calculates the product of a matrix and vector using inner product estimation
        
        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            The key with which the phase estimation will be randomized.
        
        matrix : numpy.array
            The matrix to be used in the matrix vector product
            
        vector : numpy.array
            The vector to be used in the matrix vector product
            Has to be the same size as the second dimension of the matrix

        Returns
        -------
        numpy.array
            The matrix vector product
        """
        
        assert matrix.ndim == 2
        assert vector.ndim == 1
        assert matrix.shape[1] == vector.shape[0]
        
        vmap_args = (
            random.split(rng_key, matrix.shape[0]),
            matrix,
        )
        result = vmap(
            lambda rng_key_vv, row: self.quantum_inner_product_estimation(rng_key_vv, row, vector), in_axes=(0, 0), out_axes=0
        )(*vmap_args)
                
        return result
        
    def matrix_matrix_product(self, rng_key, matrix_1, matrix_2) :
        """
        Calculates the product of two matrices using inner product estimation
        
        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            The key with which the phase estimation will be randomized.
            
        matrix_1 : numpy.array
            The first matrix to be used in the product
            
        matrix_2 : numpy.array
            The second matrix to be used in the product.
            Its second dimension has to be the same size as the first dimension of matrix_1.

        Returns
        -------
        numpy.array
            The product of both input matrices
        """
        
        assert matrix_1.ndim == 2
        assert matrix_2.ndim == 2
        assert matrix_1.shape[1] == matrix_2.shape[0]
        
        vmap_args = (
            random.split(rng_key, matrix_2.shape[1]),
            matrix_2,
        )
        result = vmap(
            lambda rng_key_mv, column: self.matrix_vector_product(rng_key_mv, matrix_1, column), in_axes=(0, 1), out_axes=1
        )(*vmap_args)
            
        return result
