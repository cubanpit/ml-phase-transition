// Wolff cluster algorithm for the 2-D Ising Model

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;

double J = +1;                  // ferromagnetic coupling
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
int **s;                        // the spins
double T;                       // temperature
int steps;                      // number of Monte Carlo steps

mt19937_64 rndGen;
uniform_real_distribution<double> rndDist(0,1);

void initialize ( ) {
  s = new int* [Lx];
  for (int i = 0; i < Lx; i++) {
    s[i] = new int [Ly];
  }
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      s[i][j] = rndDist(rndGen) < 0.5 ? +1 : -1;   // hot start
    }
  }
  steps = 0;
}

bool **cluster;                     // cluster[i][j] = true if i,j belongs
double addProbability;              // 1 - e^(-2J/kT)

void initializeClusterVariables() {

  // allocate 2-D array for spin cluster labels
  cluster = new bool* [Lx];
  for (int i = 0; i < Lx; i++) {
    cluster[i] = new bool [Ly];
  }

  // compute the probability to add a like spin to the cluster
  addProbability = 1 - exp(-2*J/T);
}

// declare functions to implement Wolff algorithm
void growCluster(int i, int j, int clusterSpin);
void tryAdd(int i, int j, int clusterSpin);

void oneMonteCarloStep() {

  // no cluster defined so clear the cluster array
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Lx; j++) {
      cluster[i][j] = false;
    }
  }

  // choose a random spin and grow a cluster
  int i = int(rndDist(rndGen) * Lx);
  int j = int(rndDist(rndGen) * Ly);
  growCluster(i, j, s[i][j]);

  ++steps;
}

void growCluster(int i, int j, int clusterSpin) {

  // mark the spin as belonging to the cluster and flip it
  cluster[i][j] = true;
  s[i][j] = -s[i][j];

  // find the indices of the 4 neighbors
  // assuming periodic boundary conditions
  int iPrev = i == 0    ? Lx-1 : i-1;
  int iNext = i == Lx-1 ? 0    : i+1;
  int jPrev = j == 0    ? Ly-1 : j-1;
  int jNext = j == Ly-1 ? 0    : j+1;

  // if the neighbor spin does not belong to the
  // cluster, then try to add it to the cluster
  if (!cluster[iPrev][j])
    tryAdd(iPrev, j, clusterSpin);
  if (!cluster[iNext][j])
    tryAdd(iNext, j, clusterSpin);
  if (!cluster[i][jPrev])
    tryAdd(i, jPrev, clusterSpin);
  if (!cluster[i][jNext])
    tryAdd(i, jNext, clusterSpin);
}

void tryAdd(int i, int j, int clusterSpin) {
  if (s[i][j] == clusterSpin) {
    if (rndDist(rndGen) < addProbability) {
      growCluster(i, j, clusterSpin);
    }
  }
}

// declare mean spin value
double magnetization = 0;

void measureObservables() {

  // compute mean spin value
  int M = 0;
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      M += s[i][j];
    }
  }
  magnetization = double(M) / double(N);
}

int main() {

  Ly = Lx = 32;
  N = Lx * Ly;
  int MCSteps = 5000;

  // start temperature
  T = 1;
  while (T <= 5) {
    initialize();
    initializeClusterVariables();

    for (int i = 0; i < MCSteps; i++) {
      oneMonteCarloStep();
    }
    measureObservables();
    cout << magnetization << " " << T << "\n";

    for (int i = 0; i < Lx; i++) {
      for (int j = 0; j < Ly; j++) {
        cout << s[i][j] << " ";
      }
    }
    cout << "\n";

    T += 0.1;
  }
}
