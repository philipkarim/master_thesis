#pragma once
#include "wavefunction.h"
#include <iostream>
#include <armadillo>

using namespace arma;

class NeuralState : public WaveFunction {
public:
    NeuralState(class System* system, int part, int dim, double sigma);
    double evaluate(vec X_visible);
    double computeDoubleDerivative();
    double computeDerivative(vec X_visible);
    double computeQuantumForce(double X_visible_index, int index);

    double sigmoid(double x);
    double sigmoid_input(int x);

    void set_X(vec X){m_x = X;}
    void set_h(vec h){m_h = h;}
    void set_a(vec a){m_a = a;}
    void set_b(vec b){m_b = b;}
    void set_w(mat w) {m_w = w; }

    vec get_X(){return m_x;}
    vec get_h(){return m_h;}
    vec get_a(){return m_a;}
    vec get_b(){return m_b;}
    mat get_w(){return m_w;}
    
    double getSigma(){return m_sigma;}


private:
    double m_sigma;
    vec m_x;
    vec m_h;
    vec m_a;
    vec m_b;
    mat m_w;

};
