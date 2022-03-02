#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

#include <armadillo>

using namespace std;
using namespace arma;

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
    Hamiltonian(system) {
    assert(omega   > 0);
    m_omega  = omega;   
}

double HarmonicOscillator::computeLocalEnergy() {
  //This function is computing the kinetic, potential and interaction energies
  vec X_visible=m_system->getWaveFunction()->get_X();

  //Defining some variables to be used in the calculations 
  double potentialEnergy = 0;
  double kineticEnergy   = 0;
  double interactionEnergy=0;

  kineticEnergy=computeKineticEnergy(X_visible);
  potentialEnergy=computePotentialEnergy(X_visible);

  if (m_system->getInteraction()==true){
    interactionEnergy=computeInteractingEnergy(X_visible);
  }

  return kineticEnergy+potentialEnergy+interactionEnergy;
  }
   
double HarmonicOscillator::computePotentialEnergy(vec X_visible) {
  //Potential energy
  //Defining some variables to be used
  double potentialEnergy2=0;
  int dimension=m_system->getNumberOfDimensions();
    for (int i=0; i<m_system->getNumberOfVN(); i+=dimension){
        for (int j=0; j<dimension; j++){
            potentialEnergy2 += X_visible[i+j]*X_visible[i+j];
        }
    }
    potentialEnergy2 *= m_omega*m_omega*0.5;
    return potentialEnergy2;

}

double HarmonicOscillator::computeKineticEnergy(vec X_visible){
  double double_derivative, derivative;
  if(m_system->getSampleMethod()==2){
    //Gibbs wavefunction has a bit different energy derivate than the two others
    double_derivative=0.5*m_system->getWaveFunction()->computeDoubleDerivative();
    derivative       =0.5*m_system->getWaveFunction()->computeDerivative(X_visible);
  }
  else{
    double_derivative=m_system->getWaveFunction()->computeDoubleDerivative();
    derivative       =m_system->getWaveFunction()->computeDerivative(X_visible);
  }

  return -0.5*(derivative*derivative+double_derivative);

}

double HarmonicOscillator::computeInteractingEnergy( vec X_visible){
  //Computing the interacting energy, sends in two particles
  int dimension = m_system->getNumberOfDimensions();
  double interactingEnergy2, product_term;
  double norm = 0;

  //Interacting term, looping through the particles keeping the different dimensions in mind
  for (int i=1; i<m_system->getNumberOfParticles(); i++){
    for (int j=0; j<i; j++){
      for (int dim=0; dim<dimension; dim++){
        product_term=X_visible[dimension*i + dim] - X_visible[dimension*j + dim];
        norm += product_term*product_term;
      }
        interactingEnergy2 += 1/sqrt(norm);
    }
  }
  return interactingEnergy2;
}


vec HarmonicOscillator::computeParameterDerivatives(){
//Computes the derivative with respect to the different parameters, need this for SGD
    //Defining some variables
    int numberOfVN = m_system->getNumberOfVN();
    int numberOfHN = m_system->getNumberOfHN();
    double sig = m_system->getWaveFunction()->getSigma();
    double sig_input3, sig_input2;

    vec xx = m_system->getWaveFunction()->get_X();
    vec aa = m_system->getWaveFunction()->get_a();
    vec bb = m_system->getWaveFunction()->get_b();
    mat ww = m_system->getWaveFunction()->get_w();

    vec derivate_parameter_vec; derivate_parameter_vec.zeros(numberOfVN + numberOfHN + numberOfVN*numberOfHN);

    //derived with respect to a
    for (int i=0; i<numberOfVN; i++){
        derivate_parameter_vec[i] = (xx[i] - aa[i])/(sig*sig);
    }

    //derived with respect to b
    int last_loop=numberOfVN+numberOfHN;
    for (int j=numberOfVN; j<last_loop; j++){
      sig_input2=m_system->getWaveFunction()->sigmoid_input(j-numberOfVN);
      derivate_parameter_vec[j]=m_system->getWaveFunction()->sigmoid(sig_input2);
      }

    //derived with respect to w
    int k = last_loop;
    for (int l=0; l<numberOfVN; l++){
        for (int m=0; m<numberOfHN; m++){
            sig_input3=m_system->getWaveFunction()->sigmoid_input(m);
            derivate_parameter_vec[k] = xx[l]/(sig*sig)*m_system->getWaveFunction()->sigmoid_input(sig_input3);
            k++;
        }
    }
    //Uses a bit different derivative of parameters when using gibbs sampling
    if(m_system->getSampleMethod()==2){
      return derivate_parameter_vec*0.5;
    }
    else{
      return derivate_parameter_vec;
    }
}
