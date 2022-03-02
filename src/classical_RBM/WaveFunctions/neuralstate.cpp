#include "neuralstate.h"
#include <cmath>
#include <cassert>
#include "wavefunction.h"
#include "../system.h"

#include <iostream>
#include <armadillo>

using namespace arma;

NeuralState::NeuralState(System* system, int part, 
                                int dim, double sigma) :
        WaveFunction(system) {
    m_sigma = sigma;
    m_system->setNumberOfParticles(part);
    m_system->setNumberOfDimensions(dim);

}

double NeuralState::evaluate(vec X_visible) {
    //Evaluates the wavefunction at the given positing
    //Just written the wave function expression straight out
    double exponent_one=0;
    double product_term=1;
    double sum_in_product=0;
    double psi_value;

    for (int i=0; i<m_system->getNumberOfVN(); i++){
        exponent_one+=((X_visible[i]-m_a[i])*(X_visible[i]-m_a[i]))/(2*m_sigma*m_sigma);
    }

    for (int j=0; j<m_system->getNumberOfHN(); j++){
        for (int ii=0; ii<m_system->getNumberOfVN(); ii++){
            sum_in_product+=(X_visible[ii]*m_w(ii, j))/(m_sigma*m_sigma);
            product_term*=(1+exp(m_b[j]+sum_in_product));
        }
    }
    psi_value=exp(-exponent_one)*product_term;

    return psi_value;
}

double NeuralState::computeDoubleDerivative() {
//Computes the value of the double derivative. (This is only one part of the energy)
    double sum_M=0;
    double sum_N=0;
    double sig_inp;
    for (int i=0; i<m_system->getNumberOfVN(); i++){
        sum_M-=1/(m_sigma*m_sigma);
        for (int j=0; j<m_system->getNumberOfHN(); j++){
            sig_inp=sigmoid_input(j);
            sum_N+=(m_w(i,j)*m_w(i,j))/(pow(m_sigma, 4))*sigmoid(sig_inp)*sigmoid(-sig_inp);
        }
        sum_M+=sum_N;
    }

    return sum_M;
}

double NeuralState::computeDerivative(vec X_visible) {
//Computes the value of the analytically first derivative used to compute the local energy
    double first_sum=0;
    double sec_sum=0;
    
    for (int i =0; i<m_system->getNumberOfVN(); i++){
        first_sum-=(X_visible[i]-m_a[i])/(m_sigma*m_sigma);
        for (int j=0; j<m_system->getNumberOfHN(); j++){
            sec_sum+=m_w(i,j)/(m_sigma*m_sigma)*sigmoid(sigmoid_input(j));
        }
        first_sum+=sec_sum;
    }
    
    return first_sum;
  }


//Just the sigmoid function to be used in the derivative functions
double NeuralState::sigmoid(double x){
    return (1/(1+exp(-x)));
}

//Computes the input of the sigmoid function
double NeuralState::sigmoid_input(int x){
    double sum=1.;
    for (int i=0; i<m_system->getNumberOfVN(); i++){
        sum+=m_x(i)*m_w(i,x)/(m_sigma*m_sigma);
    }

    return m_b(x)+sum;
}

//Computes the quantum force for the given X node at the given index.
//This is usen in importance sampling
double NeuralState::computeQuantumForce(double X_visible_index, int index){
    double first_sum=0;
    double sec_sum=0;
    
    for (int i =0; i<m_system->getNumberOfVN(); i++){
        first_sum-=(X_visible_index-m_a[index])/(m_sigma*m_sigma);
        for (int j=0; j<m_system->getNumberOfHN(); j++){
            sec_sum+=m_w(i,j)/(m_sigma*m_sigma)*sigmoid(sigmoid_input(j));
        }
        first_sum+=sec_sum;
    }
    return first_sum;
  }
