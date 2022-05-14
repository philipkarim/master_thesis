#pragma once
#include "hamiltonian.h"
#include <vector>
#include <armadillo>

class HarmonicOscillator : public Hamiltonian {
public:
    HarmonicOscillator(System* system, double omega);
    double computeLocalEnergy();
    double computePotentialEnergy(arma::vec X_visible);
    double computeKineticEnergy(arma:: vec X_visible);
    double computeInteractingEnergy(arma:: vec X_visible);
    arma::vec computeParameterDerivatives();

private:
    double m_omega = 0;


};
