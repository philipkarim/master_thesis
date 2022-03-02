#pragma once
#include <armadillo>

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy()=0;
    virtual double computePotentialEnergy(arma::vec X_visible)=0;
    virtual double computeKineticEnergy(arma:: vec X_visible)=0;
    virtual double computeInteractingEnergy(arma:: vec X_visible)=0;
    virtual arma::vec computeParameterDerivatives()=0;

protected:
    class System* m_system = nullptr;
};
