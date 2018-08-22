// Wolff cluster algorithm for the 2-D Ising Model

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random> // genearator and dist
#include <vector> // std::vector
#include <algorithm> // std::for_each
#include <sys/time.h>// time in microseconds

double J = +1;                  // ferromagnetic coupling
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
int **s;                        // the spins
double T;                       // temperature
int steps;                      // number of Monte Carlo steps

std::mt19937 rndGen;
std::uniform_real_distribution<double> rndDist(0,1);

void initialize ( ) {

  s =
    new int* [Lx];
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

  //
  //no cluster defined so clear the cluster array
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

  // add two diagonal nearest neighbor to obtain
  // a triangular lattice
  if (!cluster[iPrev][jPrev])
    tryAdd(iPrev, jPrev, clusterSpin);
  if (!cluster[iNext][jNext])
    tryAdd(iNext, iPrev, clusterSpin);
}

void tryAdd(int i, int j, int clusterSpin) {

  if
    (s[i][j] == clusterSpin) {
    if (rndDist(rndGen) < addProbability) {
      growCluster(i, j, clusterSpin);
    }
  }
}

// various block interesting values
double blockM = 0; // magnetization, spin average
double blockV = 0; // spin variance

// compute block averages, useful to understand model behaviour
void measureBlockObservables() {

  // compute mean spin value
  int M = 0;
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      M += s[i][j];
    }
  }
  blockM = fabs(double(M) / double(N));
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
  int MCSteps = 20000;
  int blockSize = 1000; // suggested by Wolff is 1000

  // if true block values will be computed and printed on stderr
  // more information, more time
  // useful to adjust parameters (steps, block size)
  bool computeBlockValues = false;
  std::vector<double> blockMvector; // magnetization averages computed on blocks

  if (computeBlockValues) {
    MCSteps += int(MCSteps/5);
  }

  // start temperature
  T = 2;
  while (T <= 4.5) {

    // get time in microseconds and use it as seed
    struct timeval tv;
    gettimeofday(&tv,NULL);
    rndGen.seed(tv.tv_usec);

    initialize();
    initializeClusterVariables();

    if (computeBlockValues) {
      // thermalize
      for (int i = 0; i < int(MCSteps/5); i++) {
        oneMonteCarloStep();
      }
    }

    for (int i = 0; i < MCSteps; i++) {
      oneMonteCarloStep();
      if (computeBlockValues and i != 0 and i % blockSize == 0) {
        measureBlockObservables();
        blockMvector.push_back(blockM);
      }
    }
    measureObservables();

    if (computeBlockValues) {
      // compute mean value of block avgs vector
      double sum = std::accumulate(
          std::begin(blockMvector),
          std::end(blockMvector),
          0.0);
      double meanM =  sum / blockMvector.size();

      // compute variance of block avgs vector
      double accum = 0.0;
      std::for_each (
          std::begin(blockMvector),
          std::end(blockMvector),
          [&](const double d) { accum += (d - meanM) * (d - meanM); });
      double meanV = accum / blockMvector.size();

      // reset vector for the next temperature run
      blockMvector.clear();

      std::cerr << T << " " << meanM << " " << meanV << std::endl;
	}
    std::cout << magnetization << " " << T << "\n";

    for (int i = 0; i < Lx; i++) {
      for (int j = 0; j < Ly; j++) {
        std::cout << s[i][j] << " ";
      }

    }
    std::cout << "\n";

    T += 0.05;
  }
}
