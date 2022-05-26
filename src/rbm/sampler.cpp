#include <iostream>
#include <cmath>
#include <vector>
#include "sampler.h"
#include "system.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
 #include <iomanip>

//Write to file modules
#include <fstream>
//Modules to meassure the cpu time
#include <ctime>
#include <ratio>
#include <chrono>

#include <string>

using namespace std::chrono;
using namespace std;

using std::cout;
using std::endl;

using namespace arma;

Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::sample(bool acceptedStep) {
    //Sampling interesting results
    if (m_stepNumber == 0) {
        //m_energy=0;
        int VN=m_system->getNumberOfVN();
        int HN=m_system->getNumberOfHN();
        m_cumulativeE_Lderiv.zeros(HN + VN + HN*VN);
        m_cumulativeE_Lderiv_expect.zeros(HN + VN + HN*VN);
        m_cumulativeEnergy = 0;
        m_cumulativeEnergy2 = 0;
        time_sec =0;
        m_E_Lderiv.zeros(HN + VN + HN*VN);
        m_E_Lderiv_expect.zeros(HN + VN + HN*VN);
    }

    //Starting the clock
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    //Calculating the local energy
    //double localEnergy;
    double localEnergy= m_system->getHamiltonian()->computeLocalEnergy();

   //Stopping the clock, adding the time together for each energy cycle
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
	  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    time_sec += time_span.count();

    //Saving values to be used in blocking
    //Metropolis steps are 2^n, performs blocking on 2^(n-1) because of the equilibration factor
    if (int(meanenergy_list.size())<int(m_system->getNumberOfMetropolisSteps()/2)){
      meanenergy_list.push_back(localEnergy);
    }

    //Cumulating the energy
    m_cumulativeEnergy  += localEnergy;
    m_stepNumber++;

    //Used in variance
    m_cumulativeEnergy2+=(localEnergy*localEnergy);

    //Used in SGD
    vec energy_derivate = m_system->getHamiltonian()->computeParameterDerivatives();

    m_cumulativeE_Lderiv+=energy_derivate;
    m_cumulativeE_Lderiv_expect+=energy_derivate*localEnergy;

    if (acceptedStep){
        m_acceptedSteps++;
    }

}

void Sampler::printOutputToTerminal() {
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    double  ef = m_system->getEquilibrationFraction();
    int     h_nodes= m_system->getNumberOfHN();
    double  lr= m_system->getLearningRate();
    
    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << np << endl;
    cout << " Number of dimensions : " << nd << endl;
    cout << " Number of hidde nodes: " << h_nodes << endl;
    cout << " Learning rate        : " << lr << endl;
    cout << " Number of Metropolis steps run : 10^" << log10(ms) << endl;
    cout << " Number of equilibration steps  : 10^" << log10(round(ms*ef)) << endl;
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " CPU time: " << time_sec << " s" << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " Variance : " << m_variance << endl;
    cout << " Accepted step ratio : " << m_acceptRatio << endl;
    cout << endl;

}

void Sampler::computeAverages() {
    //Computing the averages of the sampled quantities.
    double steps_min_eq=m_system->getNumberOfMetropolisSteps()*(1-m_system->getEquilibrationFraction());
    m_energy = m_cumulativeEnergy / steps_min_eq;
    m_cumulativeEnergy2 =m_cumulativeEnergy2/ steps_min_eq;
    m_variance=m_cumulativeEnergy2-(m_energy*m_energy);
    m_acceptRatio = m_acceptedSteps / steps_min_eq;
    m_E_Lderiv=m_cumulativeE_Lderiv/steps_min_eq;
    m_E_Lderiv_expect=m_cumulativeE_Lderiv_expect/steps_min_eq;

}

//Just checking if the variance is correct
double Sampler::computeVariance(vector<double> x_sample, double x_mean){
    double var_sum=0;
    for (int i=0; i<m_acceptedSteps; i++){
        var_sum+=pow((x_sample[i]-x_mean),2);
    }
    cout<<x_sample.size();
    return var_sum/(x_sample.size()-1);

}

//Here on its just functions for writing different quantities to file
//Benchmarking results
void Sampler::writeToFile(){
  ofstream myfile, myfiletime;
  string folderpart1, distribution_part, method, interaction_part;

  int sample_method=m_system->getSampleMethod();

  if(m_system->getInteraction()){
    interaction_part="interaction";
  }
  else{
    interaction_part="no_interaction";
  }

  if(m_system->getDistribution()){
    distribution_part="uniform_distribution";
  }
  else{
    distribution_part="normal_distribution";
  }

  if (sample_method==0){
    method="bruteforce";
  }
  else if (sample_method==1){
    method="importance";
  }
  else{
    method="gibbs";
  }

  folderpart1 ="Results/"+interaction_part+"/"+method+"/"+distribution_part+"/";

  int parti= m_system->getNumberOfParticles();
  int dimen= m_system->getNumberOfDimensions();

  string filename=folderpart1+"N="+to_string(parti)+"D="+to_string(dimen);
  string filenametime=folderpart1+"time/"+"N="+to_string(parti)+"D="+to_string(dimen)+"new";

  myfile.open(filename);
  myfiletime.open(filenametime);

  cout << "Mean energies are being written to file.."<<endl;
  myfiletime<<time_sec<<endl;
  myfiletime.close();

  for(int i=0; i<int(meanenergy_list.size()); i++){
    myfile<< fixed << setprecision(8) <<meanenergy_list[i]<<endl;
  }
  cout << "Done!"<<endl;
  cout<<endl;

  myfile.close();


}

void Sampler::writeToFile_steps(){
  ofstream myfile, myfiletime;
  string folderpart1, distribution_part, method, interaction_part;
  string step_size="step_size";
  double step_value;
  int sample_method=m_system->getSampleMethod();

  if(m_system->getInteraction()){
    interaction_part="interaction";
  }
  else{
    interaction_part="no_interaction";
  }

  if(m_system->getDistribution()){
    distribution_part="uniform_distribution";
  }
  else{
    distribution_part="normal_distribution";
  }

  if (sample_method==0){
    method="bruteforce";
    step_value=m_system->getStepLength();
  }
  else if (sample_method==1){
    method="importance";
    step_value=m_system->getTimeStep();
  }
  else{
    method="gibbs";
  }

  folderpart1 ="Results/"+interaction_part+"/"+step_size+"/"+method+"/"+distribution_part+"/";

  int parti= m_system->getNumberOfParticles();
  int dimen= m_system->getNumberOfDimensions();
  int HN= m_system->getNumberOfHN();
  double lr=m_system->getLearningRate();

  string filename=folderpart1+"N="+to_string(parti)+"D="+to_string(dimen)+"HN="+to_string(HN)+"lr="+to_string((int)round(lr*1000));

  myfile.open(filename, fstream::app);

  cout << "Mean energies are being written to file.."<<endl;

  myfile<< fixed << setprecision(8) <<m_energy<<" "<<step_value<<" "<<m_acceptRatio<<endl;
  
  cout << "Done!"<<endl;
  cout<<endl;

  myfile.close();


}

void Sampler::writeToFiles_distribution(){
  ofstream myfile, myfiletime;
  string folderpart1, distribution_part, method, interaction_part;
  vector<double> dist_energy=m_system->Getdistribution_energy();

  int sample_method=m_system->getSampleMethod();

  if(m_system->getInteraction()){
    interaction_part="interaction";
  }
  else{
    interaction_part="no_interaction";
  }

  if(m_system->getDistribution()){
    distribution_part="uniform_distribution";
  }
  else{
    distribution_part="normal_distribution";
  }

  if (sample_method==0){
    method="bruteforce";
  }
  else if (sample_method==1){
    method="importance";
  }
  else{
    method="gibbs";
  }

  folderpart1 ="Results/distribution_investigation/"+distribution_part+"/";

  int HN= m_system->getNumberOfHN();
  double lr=m_system->getLearningRate();
  double init=m_system->getInitialization();

  string filename=folderpart1+"HN="+to_string(HN)+"lr="+to_string((int)round(lr*1000))+to_string((int)round(init*1000));
  myfile.open(filename);

  cout << "Mean energies are being written to file.."<<endl;

  for(int i=0; i<int(dist_energy.size()); i++){
    myfile<< fixed << setprecision(8) <<dist_energy[i]<<endl;
  }
  cout << "Done!"<<endl;
  cout<<endl;

  myfile.close();


}
void Sampler::writeToFile_lr_nodes(){
  ofstream myfile_energy, myfile_specs;
  string folderpart1, method, interaction_part;
  int sample_method=m_system->getSampleMethod();

  if(m_system->getInteraction()){
    interaction_part="interaction";
  }
  else{
    interaction_part="no_interaction";
  }

  if (sample_method==0){
    method="bruteforce";
  }
  else if (sample_method==1){
    method="importance";
  }
  else{
    method="gibbs";
  }

  folderpart1 ="Results/"+interaction_part+"/nodes_and_lr/"+method+"/";

  int parti= m_system->getNumberOfParticles();
  int dimen= m_system->getNumberOfDimensions();
  int HN= m_system->getNumberOfHN();
  double lr=m_system->getLearningRate();

  string filename_energy=folderpart1+"N="+to_string(parti)+"D="+to_string(dimen)+"energies";
  string filename_specs=folderpart1+"N="+to_string(parti)+"D="+to_string(dimen)+"specs";

  cout << "Mean energies are being written to file.."<<endl;

  myfile_energy.open(filename_energy, fstream::app);
    for(int i=0; i<int(meanenergy_list.size()); i++){
      myfile_energy<< fixed << setprecision(8) <<meanenergy_list[i]<<endl;
    }
    myfile_energy<<" "<<endl;
  myfile_energy.close();

  myfile_specs.open(filename_specs, fstream::app);
    myfile_specs<< fixed << setprecision(8) <<HN<<" "<<lr<<endl;
  myfile_specs.close();
  
  cout << "Done!"<<endl;
  cout<<endl;
}