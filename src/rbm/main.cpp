#include <iostream>
#include "system.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/neuralstate.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "SGD.h"
#include <unistd.h>

#include <string>

using namespace std;

//Just a function used when investigating learning rates and the number of hiden nodes
void learning_rate_and_nodes(int lr_part, int lr_dim, double learning_rate, int h_node, double lr_sigma, double step_val, bool interacting_lrn, int solver){
          System* system = new System();
          system->setHamiltonian              (new HarmonicOscillator(system, 1.0));
          system->setWaveFunction             (new NeuralState(system, lr_part, lr_dim, lr_sigma));
          system->setInitialState             (new RandomUniform(system, h_node, lr_part*lr_dim,false, 0.001)); 
          system->setStepLength               (step_val);                                                   
          system->setLearningRate             (learning_rate);
          system->setTimeStep                 (step_val);
          system->setEquilibrationFraction    (0.2);
          system->setSampleMethod             (solver);
          system->setInteraction              (interacting_lrn);
          system->setWtfLrNodes               (true);
          system->setgeneralwtf               (false);
          system->runBoltzmannMachine         (40, (int) pow(2,18));
}
int main() {

    int numberOfSteps       = (int) pow(2,18); //Amount of metropolis steps
    int cycles_RBM          = 50;
    int numberOfDimensions  = 2;            // Set amount of dimensions
    int numberOfParticles   = 2;            // Set amount of particles
    int hidden_nodes        = 2;            // Set amount of hidden nodes
    int visible_nodes       = numberOfDimensions*numberOfParticles;
    int sampler_method      = 2;            //0=BF, 1=IS, 2=GS
    bool uniform_distr      = false;        //Normal=false, Uniform=true
    double omega            = 1.0;          // Oscillator frequency.
    double stepLength       = 0.5;          // Metropolis step length.
    double timeStep         = 0.4;          // Metropolis time step (Importance sampling)
    double equilibration    = 0.2;          // Amount of the total steps used for equilibration.
    bool interaction        = true;        // True-> interaction, False->Not interaction
    double sigma_val        = 0.7;            //Value of sigma, switch to 0.7 when using gibbs sampling for optimal results
    double initialization   = 0.001;        //Initialisation values of the distributions 
    double learningRate     = 0.051;        //Learning rate
    
    //Write to file, these values decides
    //which part of the investigation is going
    //to be written to file
    bool generalwtf        =false;          // Final results
    bool explore_distribution=false;        // Different distributions
    bool find_optimal_step =false;          //Investigating the step sizes and time steps
    bool nodes_lr          =false;          //Computing various values of hidden nodes vs learning rate

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
    system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes, uniform_distr, initialization));
    system->setStepLength               (stepLength);                                                               
    system->setLearningRate             (learningRate);
    system->setTimeStep                 (timeStep);
    system->setEquilibrationFraction    (equilibration);
    system->setSampleMethod             (sampler_method);
    system->setInteraction              (interaction);
    system->setgeneralwtf               (generalwtf);
    

    //From here on it is just various simulations ready to run in parallel
    if(explore_distribution==true){
      int pid, pid1, pid2, pid3, pid4, pid5, pid6;
      //Using more cores to achieve more results faster
      pid=fork();
      if(pid==0){
        //Uniform distribution, initialization value=0.25
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,true, 0.25)); 
          system->setWtfDistibution           (explore_distribution);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

        //Uniform distribution, initialization value=0.01
      else{pid1=fork(); if(pid1==0){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,true, 0.01)); 
          system->setWtfDistibution           (explore_distribution);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

        //Uniform distribution, initialization value=0.001
      else{pid2=fork(); if(pid2==0){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,true, 0.001)); 
          system->setWtfDistibution           (explore_distribution);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

        //Normal distribution, initialization value=0.001
      else{pid3=fork(); if(pid3==0){
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
        system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,true, 0.005)); 
        system->setWtfDistibution           (explore_distribution);
        system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

        //Normal distribution, initialization value=0.001
      else{pid4=fork(); if(pid4==0){
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
        system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,false, 0.25)); 
        system->setWtfDistibution           (explore_distribution);
        system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

      else{pid5=fork(); if(pid5==0){
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
        system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,false, 0.01)); 
        system->setWtfDistibution           (explore_distribution);
        system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

      else{pid6=fork(); if(pid6==0){
        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
        system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,false, 0.001)); 
        system->setWtfDistibution           (explore_distribution);
        system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);}

        //Normal distribution, initialization value=0.001
      else{
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,false, 0.005)); 
          system->setWtfDistibution           (explore_distribution);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}}}}}}}

    else if(find_optimal_step==true){
      int pid, pid1, pid2, pid3, pid4;
      //Using more cores to achieve more results faster
      pid=fork();
      if(pid==0){
        //bf, non interacting, different step sizes
        for (double i=1.5; i>0.1; i-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (i);                                                   
          system->setLearningRate             (0.1);
          system->setTimeStep                 (timeStep);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (0);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}

      
      //bf, interacting, different step sizes
      else{pid1=fork(); if(pid1==0){
        for (double k=1.5; k>0.1; k-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (k);                                                        
          system->setLearningRate             (0.25);
          system->setTimeStep                 (timeStep);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (0);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}

      //is, non interacting, different timesteps
      else{pid2=fork(); if(pid2==0){
        for (double j=1; j>0.1; j-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (stepLength);                                                  
          system->setLearningRate             (0.1);
          system->setTimeStep                 (j);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (1);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}
              //bf, interacting, different step sizes
      else{pid3=fork(); if(pid3==0){
        for (double f=1; f>0.1; f-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, sigma_val));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (stepLength);                                                        
          system->setLearningRate             (0.25);
          system->setTimeStep                 (f);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (1);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}

      //is, non interacting, different timesteps
      else{pid4=fork(); if(pid4==0){
        for (double g=1; g>0.1; g-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, g));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (stepLength);                                                  
          system->setLearningRate             (0.1);
          system->setTimeStep                 (timeStep);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (2);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}
      else{
        for (double gg=1; gg>0.1; gg-=0.1){
          system->setHamiltonian              (new HarmonicOscillator(system, omega));
          system->setWaveFunction             (new NeuralState(system, numberOfParticles, numberOfDimensions, gg));
          system->setInitialState             (new RandomUniform(system, hidden_nodes, visible_nodes,uniform_distr, initialization)); 
          system->setStepLength               (stepLength);                                                     
          system->setLearningRate             (0.25);
          system->setTimeStep                 (timeStep);
          system->setEquilibrationFraction    (equilibration);
          system->setSampleMethod             (2);
          system->setInteraction              (false);
          system->setgeneralwtf               (generalwtf);
          system->setwtfSteps                 (find_optimal_step);
          system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
        }}} }}}}
    else if(nodes_lr==true){
      //Testing for various learning rates and nodes
      int pid, pid1, pid2, pid3, pid4;
      //Using more cores to achieve more results faster
      pid=fork();
      if(pid==0){
        //bf, non interacting
        for (double lr_bf=0.001; lr_bf<0.41; lr_bf+=0.05){
          for (int node_bf=2; node_bf<20; node_bf+=2){
            learning_rate_and_nodes( 1, 1, lr_bf, node_bf,  1., 0.5, false, 0);
        }}}

      //is, non interacting
      else{pid1=fork(); if(pid1==0){
        for (double lr_is=0.001; lr_is<0.41; lr_is+=0.05){
          for (int node_is=2; node_is<20; node_is+=2){
            learning_rate_and_nodes(1, 1, lr_is, node_is, 1., 0.25, false, 1);
        }}}

      //gibbs, non interacting
      else{pid2=fork(); if(pid2==0){
        for (double lr_gibbs=0.001; lr_gibbs<0.41; lr_gibbs+=0.05){
          for (int node_gibbs=2; node_gibbs<20; node_gibbs+=2){
            learning_rate_and_nodes(1, 1, lr_gibbs, node_gibbs, 0.7, 0.5, false, 2);
        }}}
      //bf, interacting
      else{pid3=fork(); if(pid3==0){
                for (double lr_bf_int=0.001; lr_bf_int<0.41; lr_bf_int+=0.05){
          for (int node_bf_int=2; node_bf_int<20; node_bf_int+=2){
            learning_rate_and_nodes(2, 2, lr_bf_int, node_bf_int, 1., 0.5, true, 0);
        }}}

      //is, interacting
      else{pid4=fork(); if(pid4==0){
        for (double lr_is_int=0.001; lr_is_int<0.41; lr_is_int+=0.05){
          for (int node_is_int=2; node_is_int<20; node_is_int+=2){
            learning_rate_and_nodes(2, 2, lr_is_int, node_is_int, 1., 0.4, true, 1);
        }}}
      else{
        for (double lr_gibbs_int=0.001; lr_gibbs_int<0.41; lr_gibbs_int+=0.05){
          for (int node_gibbs_int=2; node_gibbs_int<20; node_gibbs_int+=2){
            learning_rate_and_nodes(2, 2, lr_gibbs_int, node_gibbs_int, 0.7, 0.5, true, 2);
        }}} }}}}}
    
    //This one is a bit overkill, since it runs on 12 cores, 
    //so be sure that the computer doing this specific part is ran on a computer with 12 cores
    else if(generalwtf==true){
      //Testing for various learning rates and nodes
      int pid, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10;
      //Using more cores to achieve more results faster
      pid=fork();
      if(pid==0){
        //bf, non interacting, 1D
        learning_rate_and_nodes(1, 1, 0.101, 14,  1., 0.5, false, 0);
        }
      //is, non interacting, 1D
      else{pid1=fork(); if(pid1==0){
        learning_rate_and_nodes(1, 1, 0.001, 14, 1., 0.25, false, 1);
        }
      //gibbs, non interacting, 1D
      else{pid2=fork(); if(pid2==0){
        learning_rate_and_nodes(1, 1, 0.251, 10, 0.7, 0.5, false, 2);
        }
      //bf, non interacting, 2D      
      else{pid3=fork(); if(pid3==0){
        learning_rate_and_nodes(1, 2, 0.101, 14,  1., 0.5, false, 0);
        }
      //is, non interacting, 2D
      else{pid4=fork(); if(pid4==0){
        learning_rate_and_nodes(1, 2, 0.001, 14, 1., 0.25, false, 1);
      }
      //gibbs, non interacting, 2D
      else{pid5=fork(); if(pid5==0){
        learning_rate_and_nodes(1, 2, 0.251, 10, 0.7, 0.5, false, 2);
      }
      //bf, non interacting, 3D      
      else{pid6=fork(); if(pid6==0){
        learning_rate_and_nodes(1, 3, 0.101, 14,  1., 0.5, false, 0);
      }
      //is, non interacting, 3D
      else{pid7=fork(); if(pid7==0){
        learning_rate_and_nodes(1, 3, 0.001, 14, 1., 0.25, false, 1);
      }
      //gibbs, non interacting, 3D
      else{pid8=fork(); if(pid8==0){
        learning_rate_and_nodes(1, 3, 0.251, 10, 0.7, 0.5, false, 2);
      }
      //bf, non interacting, 3D
      else{pid9=fork(); if(pid9==0){
        learning_rate_and_nodes(2, 2, 0.351, 8, 1., 0.5, true, 0);
      }
      //is, interacting, 3D
      else{pid10=fork(); if(pid10==0){
        learning_rate_and_nodes(2, 2, 0.251, 8, 1., 0.4, true, 1);
      }
      //gibbs, interacting, 3D
      else{
        learning_rate_and_nodes(2, 2, 0.101, 16, 0.7, 0.5, true, 2);
        }}}}}}}}}}}}
    else{
    //Setting the different values defined higher in the code
    system->runBoltzmannMachine         (cycles_RBM, numberOfSteps);
    }
    return 0;
}