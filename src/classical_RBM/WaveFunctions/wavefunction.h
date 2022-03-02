#pragma once
#include <iostream>
#include <armadillo>

using namespace arma;

class WaveFunction {
public:
    WaveFunction(class System* system);

    virtual double evaluate(vec X_visible)=0;
    virtual double computeDoubleDerivative()=0;
    virtual double computeDerivative(vec X_visible)=0;
    virtual double sigmoid(double x)=0;
    virtual double sigmoid_input(int x)=0;
    virtual double computeQuantumForce(double X_visible_index, int index)=0;

    virtual void set_X(vec X)=0;
    virtual void set_h(vec h)=0;
    virtual void set_a(vec a)=0;
    virtual void set_b(vec b)=0;
    virtual void set_w(mat w)=0;
    virtual vec get_X()=0;
    virtual vec get_h()=0;
    virtual vec get_a()=0;
    virtual vec get_b()=0;
    //This is actually a matrix. Use armadillo or eigen maybe?
    virtual mat get_w()=0;
    virtual double getSigma()=0;

protected:
    vec m_x;
    vec m_h;
    vec m_a;
    vec m_b;
    mat m_w;

    class System* m_system = nullptr;
//private:
    //virtual double sigmoid(double x)=0;
    //virtual double sigmoid_input(int x)=0;

};
