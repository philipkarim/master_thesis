#include "SGD.h"
#include <cassert>
#include <iostream>
#include <armadillo>
#include "system.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"

using namespace arma;

SGD::SGD(System* system){
    m_system = system;
}

// Computes the new parameters to be used using SGD
int SGD::SGDOptimize(int cycle, vec parameters_derivative){
    //Input of shape[a_0, a_1,...,a_n, b_0, b_1,...,w_00,w01,...w_nn]
    //Starts of by declaring the current parameters and variables:
    int index_b, index_w;
    double tol = 1e-6;
    double lr=m_system->getLearningRate();
    //Visible and hidden nodes
    int numberOfVN = m_system->getNumberOfVN();
    int numberOfHN = m_system->getNumberOfHN();
    //Parameters to be optimized
    vec a_visible = m_system->getWaveFunction()->get_a();
    vec b_hidden = m_system->getWaveFunction()->get_b();
    mat w_weight = m_system->getWaveFunction()->get_w();
    
    // Computes new visible biases by looping over the first elements in the parameter vector
    for (int i=0; i<numberOfVN; i++){
        a_visible[i] -= lr*parameters_derivative[i];
    }
    
    // Computes new hidden biases by looping over the middle elements in the parameter vector
    for (int j=0; j<numberOfHN; j++){
        index_b=numberOfVN + j;
        b_hidden[j] -= lr*parameters_derivative[index_b];

    }
    // Computes new weights by looping over the last elements in the parameter vector
    index_w = numberOfHN + numberOfVN;
    //Not sure if the loops actually breaks or gets all elements, test when I get home
    for (int k=0; k<numberOfVN; k++){
        for (int l=0; l<numberOfHN; l++){
            w_weight(k, l) -= lr*parameters_derivative[index_w];
            //index_w++;
            if (index_w==int(parameters_derivative.n_elem)){
                std::cout<<"BREAK due to SGD loop!";
                break;
            }
            index_w++;
        }
    }
    
    // Changing the old parameters into the new ones
    m_system->getWaveFunction()->set_a(a_visible);
    m_system->getWaveFunction()->set_b(b_hidden);
    m_system->getWaveFunction()->set_w(w_weight);

    //Adding a tol to stop the loop
    if (norm(parameters_derivative)<tol){
        std::cout<<"SGD tol reached, finishing simulation";
        return m_system->getRBMCycles()-1;
    }
    else{
        return cycle;
    }

}