#pragma once
#include <vector>
#include <string>

using namespace std;

class System {
public:
    System();

    //Central functions
    void runBoltzmannMachine        (int RBMCycles, int numberOfMetropolisSteps);
    void runMetropolisSteps         ();
    bool metropolisStep             ();
    bool metropolisStepImportanceSampling();
    bool GibbsSampling();
    
    //Classes
    class WaveFunction*             getWaveFunction()   { return m_waveFunction; }
    class Hamiltonian*              getHamiltonian()    { return m_hamiltonian; }
    class InitialState*             getInitialState()   { return m_initialState; }
    class Sampler*                  getSampler()        { return m_sampler; }
    class SGD*                      getSGD()            { return m_SGD; }

    //Set value-functions
    void setStepLength              (double stepLength);
    void setTimeStep                (double timeStep);
    void setEquilibrationFraction   (double equilibrationFraction);
    void setNumberOfParticles       (int numberOfParticles){m_numberOfParticles=numberOfParticles;}
    void setNumberOfDimensions      (int numberOfDimensions){m_numberOfDimensions=numberOfDimensions;}
    void setHamiltonian             (class Hamiltonian* hamiltonian){m_hamiltonian=hamiltonian;}
    void setWaveFunction            (class WaveFunction* waveFunction){m_waveFunction=waveFunction;}
    void setInitialState            (class InitialState* initialState){m_initialState=initialState;}
    void setLearningRate            (double learningRate){m_learningRate=learningRate;}
    void setSampleMethod            (int sampleMethod){m_sampleMethod=sampleMethod;}
    void setInteraction             (bool interaction){m_interaction=interaction;}
    void setgeneralwtf              (bool generalwtf){m_general_wtf=generalwtf;}
    void setNumberOfHN              (int n_hidden){m_numberOfHN=n_hidden;}
    void setNumberOfVN              (int n_visible){m_numberOfVN=n_visible;}
    void setDistribution            (bool uni){m_uniform_distr=uni;}
    void setwtfSteps                (bool find_optimal_step){m_wtfSteps=find_optimal_step;}
    void setWtfDistibution          (bool distribution_wtf){m_wtfDistribution=distribution_wtf;}
    void setWtfLrNodes              (bool lr_and_nodes){m_lr_and_nodes=lr_and_nodes;}
    void setInitialization          (double initialization2){m_initialization=initialization2;}

    //Get value-functions
    int getNumberOfVN()             { return m_numberOfVN;}
    int getNumberOfHN()             { return m_numberOfHN;}
    int getSampleMethod()           { return m_sampleMethod;}
    int getNumberOfParticles()      { return m_numberOfParticles; }
    int getNumberOfDimensions()     { return m_numberOfDimensions; }
    int getNumberOfMetropolisSteps(){ return m_numberOfMetropolisSteps; }
    int getRBMCycles()              { return m_RBMCycles; }
    double getEquilibrationFraction(){return m_equilibrationFraction; }
    double getLearningRate()        { return m_learningRate;}
    double getTimeStep()            { return m_timeStep;}
    double getStepLength()          { return m_stepLength;}
    double getInteraction()         { return m_interaction;}
    bool getDistribution()          { return m_uniform_distr;}
    double getInitialization()      {return m_initialization;}
    vector<double> Getdistribution_energy(){return distribution_energy;}

private:
    int                             m_numberOfHN = 0;
    int                             m_numberOfVN = 0;
    int                             m_numberOfParticles = 0;
    int                             m_numberOfDimensions = 0;
    int                             m_numberOfMetropolisSteps = 0;
    int                             m_RBMCycles = 0;
    double                          m_equilibrationFraction = 0.0;
    double m_initialization;
    int m_sampleMethod;
    double m_stepLength;
    double m_timeStep;
    bool m_interaction;
    bool m_general_wtf=false;
    bool m_wtfSteps=false;
    bool m_wtfDistribution=false;
    bool m_lr_and_nodes=false;
    vector<double> distribution_energy;
    double m_learningRate;
    bool m_uniform_distr;
    class WaveFunction*             m_waveFunction = nullptr;
    class Hamiltonian*              m_hamiltonian = nullptr;
    class Sampler*                  m_sampler = nullptr;
    class InitialState*             m_initialState = nullptr;
    class SGD*                      m_SGD = nullptr;
};
