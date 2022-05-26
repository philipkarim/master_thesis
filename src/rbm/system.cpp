#include "system.h"
#include <cassert>
#include "sampler.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include <iostream>
#include "SGD.h"
#include<armadillo>
#include "WaveFunctions/neuralstate.h"

#include <stdlib.h>     /* exit, EXIT_FAILURE */


using namespace std;
using namespace arma;

//Defining the random number generator
//Using random_device to seed the generator
random_device rd;
mt19937 gen(rd());

System::System() {
}

bool System::metropolisStep() {
    // Performing the actual Metropolis step for the Metropolis algorithm:
    // Choosing a particle at random and changing it's position by a random
    // amount, and checks if the step is accepted by the Metropolis test

    //Distributions
    uniform_int_distribution<int> distribution(0,m_numberOfVN-1);
    uniform_real_distribution<double> UniformNumberGenerator(0.0,1.0);

    // Defining some variables to be used
    int random_index;
    double Position_old, psi_old, psi_new, psi_factor, step;
    arma::vec X_old;
    
    X_old=m_waveFunction->get_X();

    //Random index used to choose a random particle
    random_index=distribution(gen);
    //Defining the random particle:
    Position_old=X_old[random_index];
    psi_old=m_waveFunction->evaluate(X_old);

    //Start the step which gives movement to the particle
    step=m_stepLength*(UniformNumberGenerator(gen)-0.5);
    X_old[random_index]+=step;

    //cout<<step;
    
     //Extracting the new wavefunction, and checks if it is accepted
    psi_new=m_waveFunction->evaluate(X_old);
    psi_factor=psi_new*psi_new/(psi_old*psi_old);
     
    //Checks if the move is accepted:
    if (UniformNumberGenerator(gen)<=psi_factor){
        m_waveFunction->set_X(X_old);        

        return true;
     }
     else{
         X_old[random_index]=Position_old;
        
        return false;
     }
}

bool System::metropolisStepImportanceSampling() {
    // Performing the actual Metropolis step for the Metropolis- Hastings
    //algorithm: Choosing a particle at random and changing it's position 
    // by a random amount, and checks if the step is accepted by the 
    //Metropolis-Hastings test

    //Declaring vaiables to be used:
    double Position_old, wfnew, wfold, term_1, term_2, rand_norm ,green_factor, greenRate;
    int random_index;
    //Defining position and quantum force vectors
    //to be used in the importance sampling
    vec X_old=m_waveFunction->get_X();
    double QFOld;
    vec X_new=m_waveFunction->get_X();
    double QFNew;

    //Random integer generator
    uniform_int_distribution<int> distribution(0,m_numberOfVN-1);//-1?
    uniform_real_distribution<double> UniformNumberGenerator(0.0,1.0);
    normal_distribution<double> Normaldistribution(0.0,1.0);

    random_index  =distribution(gen);
    rand_norm  =Normaldistribution(gen);
    
    //Defining the values of the previous position
    wfold=m_waveFunction->evaluate(X_old);    
    QFOld=m_waveFunction->computeQuantumForce(X_old[random_index], random_index);
    Position_old= X_new[random_index];
    X_new[random_index]+=QFOld*m_timeStep*0.5 + sqrt(m_timeStep)*rand_norm;

    // Evaluate new quantities
    wfnew = m_waveFunction->evaluate(X_new);
    QFNew=m_waveFunction->computeQuantumForce(X_new[random_index], random_index);

    // Compute greens function
    term_1=X_old[random_index]-X_new[random_index]-0.5*m_timeStep*QFNew;
    term_2=X_new[random_index]-X_old[random_index]-0.5*m_timeStep*QFOld;
    greenRate=(term_2*term_2)-(term_1*term_1);

    greenRate = exp(greenRate/(2*m_timeStep));
    green_factor = greenRate*wfnew*wfnew/(wfold*wfold);

    // Check if the step is accepted
    if (UniformNumberGenerator(gen) <= green_factor) {
        m_waveFunction->set_X(X_new);
        return true;
    }
    else {
        X_old[random_index]=Position_old;
        return false;
    }

}

bool System::GibbsSampling() {
    // Performing Gibbs sampling
    double random_uniform, sigmoid_probabillity, x_avg, random_normal;
    vec X_new_2;
    vec weight_and_HN;

    X_new_2.zeros(m_numberOfVN);
    vec hh = m_waveFunction->get_h();
    vec aa = m_waveFunction->get_a();
    mat ww = m_waveFunction->get_w();

    //Defining the distribution
    uniform_real_distribution<double> UniformNumberGenerator(0.0,1.0);
    normal_distribution<double> Normaldistribution(0.0,1.0);
    
    for (int i=0; i < m_numberOfHN; i++){
        //random_uniform =UniformNumberGenerator(gen);
        random_uniform =Normaldistribution(gen);

        sigmoid_probabillity=m_waveFunction->sigmoid(m_waveFunction->sigmoid_input(i));

        hh[i]=sigmoid_probabillity;

        //cout<<"Sig prob"<<sigmoid_probabillity<<endl;
        /*
        if (sigmoid_probabillity >= UniformNumberGenerator(gen)){
            hh[i] = 1;}
        else{
            hh[i] = 0;}
        */
    }

    //New positions
    weight_and_HN = ww*hh;

    for (int j=0; j<m_numberOfVN; j++){
        x_avg = aa[j] + weight_and_HN[j];
        normal_distribution<double> Normaldistribution(x_avg, m_waveFunction->getSigma());
        random_normal=Normaldistribution(gen);
        //Filling up the new X with positions
        X_new_2[j] = random_normal;
    }

    //Set the new values
    m_waveFunction->set_X(X_new_2);

    return true;
}
    

void System::runBoltzmannMachine(int RBMCycles, int numberOfMetropolisSteps){
    m_RBMCycles                 = RBMCycles;
    m_SGD= (new SGD(this));
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
   
    //Looping over the boltzmann machine cycles
    for (int rbm_cycle=0; rbm_cycle<RBMCycles; rbm_cycle++){
        cout<<"RBM cycle: "<<rbm_cycle<<"\n------------------------------";
        m_sampler                   = new Sampler(this);
        m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);
        //Run the Monte Carlo algorithm
        runMetropolisSteps();
        //Update the parameters using SGD
        rbm_cycle=m_SGD->SGDOptimize(rbm_cycle,m_sampler->getGradient());

        //Benchmarking distribution values
        if (m_wtfDistribution==true){
            distribution_energy.push_back(m_sampler->getEnergy()); 
        }
        //Sampling the values after the parameters are optimized
        if(rbm_cycle==RBMCycles-1){
            if (m_general_wtf==true){
                m_sampler->writeToFile();
            }
            else if(m_wtfSteps==true){
                m_sampler->writeToFile_steps();
            }
            else if (m_wtfDistribution==true){
                m_sampler->writeToFiles_distribution();
            }
            else if (m_lr_and_nodes==true){
                m_sampler->writeToFile_lr_nodes();
            }
        }
    }

}
void System::runMetropolisSteps() {
    bool acceptedStep;
    
    //Looping over the amount of metropolis steps
    //for either of the sampling methods
    for (int i=0; i < m_numberOfMetropolisSteps; i++) {
      if (m_sampleMethod==0){
        acceptedStep = metropolisStep();}
      else if(m_sampleMethod==1){
        acceptedStep = metropolisStepImportanceSampling();
      }
      else if(m_sampleMethod==2){
        acceptedStep = GibbsSampling();
      }
      else{
        //Just ensuring that one of the samling methods is chosen
          cout<<"---No sampling method chosen---";
          exit (EXIT_FAILURE);
      }

      //If statement to send the accepted steps into the sampler
      //after the system is at rest
      if (i>=m_numberOfMetropolisSteps*m_equilibrationFraction){
        m_sampler->sample(acceptedStep);
      }
    }

    //Chooosing what to sample
    m_sampler->computeAverages();
    m_sampler->printOutputToTerminal();  
}

void System::setStepLength(double stepLength) {
    assert(stepLength >= 0);
    m_stepLength = stepLength;
}

void System::setTimeStep(double timeStep) {
    assert(timeStep >= 0);
    m_timeStep= timeStep;
}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}
