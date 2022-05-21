#include "randomuniform.h"
#include "initialstate.h"
#include <iostream>
#include <cassert>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

//#include <Eigen/Dense>
#include<armadillo>
using namespace std;
using namespace arma;
//using namespace Eigen
RandomUniform::RandomUniform(System* system, int n_hidden, 
                            int n_visible, bool gaussian, 
                            double initialization):
        InitialState(system) {
    assert(n_hidden >= 0 && n_visible >= 0);
    numberOfHN = n_hidden;
    numberOfVN = n_visible;
    m_uniform = gaussian;

    /* The Initial State class is in charge of everything to do with the
     * initialization of the system; this includes determining the number of
     * nodes and the number of the distribution used. To make sure everything
     * works as intended, this information is passed to the system here.
     */
    m_system->setDistribution(gaussian);
    m_system->setInitialization(initialization);
    m_system->setNumberOfHN(n_hidden);
    m_system->setNumberOfVN(n_visible);

    setupInitialState();
}

void RandomUniform::setupInitialState() {
    double initialization2=m_system->getInitialization();
    //Constructing the parametere vectors of their sizes:
    //visible nodes
    vec initial_x(numberOfVN);
    //Hidden nodes
    vec initial_h(numberOfHN);
    //Visible biases
    vec initial_a(numberOfVN);
    //Hidden biases
    vec initial_b(numberOfHN);
    //weights of the connections between the hidden and visible nodes.
    mat initial_w(numberOfVN, numberOfHN);

    //Filling the parameters with zeros
    initial_x.zeros();
    initial_a.zeros();
    initial_b.zeros();
    initial_w.zeros();
    initial_h.zeros();

    //Random generator
    random_device rd;
    mt19937_64 gen(rd());

    // Set up the distribution for x \in [[x, x],(can use multiple configurations)
    uniform_real_distribution<double> UniformNumberGenerator(-0.5,0.5);
    //uniform_real_distribution<double> UniformNumberGenerator(-m_system->getBondlength(), m_system->getBondlength());

    //Uniform distributions
    uniform_real_distribution<double> uniform_weights(-initialization2, initialization2);
    //Gaussian distriubution
    normal_distribution<double> normal_weights(0, initialization2);

    //Initializing positions from a uniform distribution
    for (int i=0; i<numberOfVN; i++){
        initial_x[i] = UniformNumberGenerator(gen);
    }

    // initializing uniform values
    if (m_system->getDistribution() == true){
        for (int i=0; i<numberOfVN; i++){
            initial_a[i] = uniform_weights(gen);
            for (int j=0; j<numberOfHN; j++){
                initial_b[j] = uniform_weights(gen);
                initial_w(i, j) = uniform_weights(gen);
            }
        }
    }

    // initializing gaussian values
    else if (m_system->getDistribution()==false){
        for (int i=0; i<numberOfVN; i++){
            initial_a[i] = 0;//normal_weights(gen);
            for (int j=0; j<numberOfHN; j++){
                initial_b[j] = 0;//normal_weights(gen);
                initial_w(i, j) = normal_weights(gen);
            }
        }
    }

    //Configuring the initialized parameters
    m_system->getWaveFunction()->set_X(initial_x);
    m_system->getWaveFunction()->set_a(initial_a);
    m_system->getWaveFunction()->set_b(initial_b);
    m_system->getWaveFunction()->set_w(initial_w);
    m_system->getWaveFunction()->set_h(initial_h);

}