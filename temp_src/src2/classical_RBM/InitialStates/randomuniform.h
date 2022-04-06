#pragma once
#include "initialstate.h"
#include <armadillo>

class RandomUniform : public InitialState {
public:
    RandomUniform(System* system, int n_hidden, 
                    int n_visible, bool gaussian, 
                    double initialization);
    void setupInitialState();

private:    
    double numberOfVN;
    double numberOfHN;
    double m_sigma;
    bool m_uniform;


};

