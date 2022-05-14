#pragma once
#include <armadillo>
using namespace arma;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(bool acceptedStep);
    void printOutputToTerminal();
    void computeAverages();
    void writeToFile();
    void writeToFileSteps(std::vector<int> steps_list, std::vector<double> meanEL_list);
    double computeVariance(std::vector<double> x_sample, double x_mean);
    double getEnergy()          { return m_energy; }
    double getVariance()          { return m_variance; }
    double getAcceptRatio()          { return m_acceptRatio; }
    void writeToFile_steps();
    void writeToFiles_distribution();
    void writeToFile_lr_nodes();
    vec getGradient()             { return 2*(m_E_Lderiv_expect-(m_E_Lderiv*m_energy)); }

private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    double  m_variance = 0;
    double  m_acceptedSteps = 0;
    double  m_acceptRatio = 0;
    double time_sec;
    vec m_E_Lderiv;
    vec m_E_Lderiv_expect;
    vec m_cumulativeE_Lderiv;
    vec m_cumulativeE_Lderiv_expect;
    vec m_energy_derivate;

    std::vector<double> meanenergy_list = std::vector<double>();

    class System* m_system = nullptr;
};
