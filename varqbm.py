class varQBM:
    def __init__(self, circ, n_visible, n_hidden, params):
        """
        Class: Variational quantum Boltzmann machine

        Args: 
            circuit(quantum cirquit):    Quantum circuit
            num_visible (int):      Number of visible nodes
		    num_hidden (int):       Number of hidden nodes
            params (array or list): The variational parameters
        """
        self.ciruit=circuit
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        self.params=params

    def state_preparation(self):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
    


    

class varQITE:
    def __init__(self, params):
        """
        Class: Variational quantum imaginary time evolution

        Args: 
            params (array or list): The variational parameters
        """

        pass
    
