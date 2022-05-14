#pragma once
#include "system.h"
#include <armadillo>

using namespace arma;

class SGD {
public:
    SGD(System* system);
    int SGDOptimize(int cycle, arma::vec parameters_derivative);

private:
    double m_eta;

    class System* m_system = nullptr;
};