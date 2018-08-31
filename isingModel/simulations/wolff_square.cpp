// Wolff cluster algorithm for the 2-D Ising Model

#include <cmath>
#include <iostream>
#include <random> // genearator and dist
#include <vector> // std::vector
#include <array> // std::array
#include <algorithm> // std::for_each

#define Lx 30
#define Ly 30                     // number of spins in x and y
const unsigned int N = Lx * Ly;   // number of spins

std::array<std::array<short int, Ly>, Lx> s;     // the spins
std::array<std::array<bool, Ly>, Lx> cluster;    // cluster[i][j] = true if i,j belongs

double J = +1;                  // ferromagnetic coupling
double T;                       // temperature
double addProbability;          // 1 - exp(-2*J/T)

// random number generator and distribution
std::mt19937 rndGen(std::random_device{}());
std::uniform_real_distribution<double> rndDist(0,1);

void initialize ( ) {

  // initialize lattice spins
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      s[i][j] = rndDist(rndGen) < 0.5 ? +1 : -1;   // hot start
      //s[i][j] = 1;   // cold start
    }
  }
}

// declare functions to implement Wolff algorithm
void growCluster(int i, int j, int clusterSpin);
void tryAdd(int i, int j, int clusterSpin);

void oneMonteCarloStep() {

  //no cluster defined so clear the cluster array
  for (int i = 0; i < Lx; i++) {
    std::fill(cluster[i].begin(), cluster[i].end(), false);
  }

  // choose a random spin and grow a cluster
  unsigned int i = int(rndDist(rndGen) * Lx);
  unsigned int j = int(rndDist(rndGen) * Ly);
  growCluster(i, j, s[i][j]);
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

  int MCSteps = 10000;  // total number of steps
  int blockSize = 1000; // suggested by Wolff is 1000

  // if true block values will be computed and printed on stderr
  // more information, more time
  // useful to adjust parameters (steps, block size)
  bool computeBlockValues = true;
  std::vector<double> blockMvector; // magnetization averages computed on blocks

  if (computeBlockValues) {
    // add thermalization steps
    MCSteps += int(MCSteps/5);
  }

  double Tc = 2 / log1p(sqrt(2));   // critical temperature
  double Tstart = 1;                 // start temperature
  int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  T = Tstart;
  for (int t = 0; t < Tn; ++t) {

    initialize(); // initialize spins array
    addProbability = 1 - exp(-2*J/T);

    if (computeBlockValues) {
      // thermalization steps
      for (int i = 0; i < int(MCSteps/5); i++) {
        oneMonteCarloStep();
      }
    }

    // effective MC steps
    for (int i = 0; i < MCSteps; i++) {
      oneMonteCarloStep();
      if (computeBlockValues and i != 0 and i % blockSize == 0) {
        measureObservables();
        blockMvector.push_back(magnetization);
      }
    }

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
    measureObservables();
    std::cout << magnetization << " " << T << "\n";

    // print full spin lattice
    for (int i = 0; i < Lx; i++) {
      for (int j = 0; j < Ly; j++) {
        std::cout << s[i][j] << " ";
      }
    }
    std::cout << "\n";

    T += Tstep;
  }

  return 0;
}
