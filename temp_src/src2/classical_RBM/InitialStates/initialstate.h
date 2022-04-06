#pragma once
#include <armadillo>

using namespace std;
class InitialState {
public:
    InitialState(class System* system);
    virtual void setupInitialState() = 0;

protected:
    class System* m_system = nullptr;

};

