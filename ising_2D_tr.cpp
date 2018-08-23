//
// This code simulates an Ising square lattice with triangular interaction
// and periodic boundary condition, at different temperatures, with Monte
// Carlo (Metropolis) algorithm.
//
// This code is released under MIT license.
//
// Author: Pietro F. Fontana <pietrofrancesco.fontana@studenti.unimi.it>
//

#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

// random generator and distribution
std::mt19937 rndGen;
std::uniform_real_distribution<double> rndDist(0,1);

// array of spins
int L = 32;
std::vector<int> spins(L*L);

// nearest neighbour vector
std::vector<std::vector<int> > nn(L*L, std::vector<int>(6));

//
// FUNCTIONS
//

// Compute nearest neighbors table
void nn_array(int L) {

  // total spin number
  int N = L * L;

  for (int i = 0; i < N; ++i) {
    int xRef = i % L;
    int yRef = int(i / L);
    int xPrev = xRef == 0 ? L-1 : xRef-1;
    int yPrev = yRef == 0 ? L-1 : yRef-1;
    int xNext = xRef == L-1 ? 0 : xRef+1;
    int yNext = yRef == L-1 ? 0 : yRef+1;
    nn[i][1] = xPrev + L * yRef;
    nn[i][2] = xNext + L * yRef;
    nn[i][3] = xRef + L * yPrev;
    nn[i][4] = xRef + L * yNext;
    nn[i][5] = xPrev + L * yPrev;
    nn[i][6] = xNext + L * yNext;
  }
}

// Ising simulation with multiple blocks
std::vector<double> ising_2D(
                             int N,
                             double J,
                             double beta,
                             int Nblock,
                             int Nstep) {

  // measured values in blocks
  std::vector<double> block_spin_avgs (Nblock);

  // initialize all spins
  std::fill(spins.begin(), spins.end(), 1);

  // randomize all spins
  //for (int i = 0; i < N; ++i) {
  //  spins[i] = int(rndDist(rndGen) + 0.5) * 2 - 1;
  //}

  // thermalization steps
  for ( int i = 0; i < (5*Nstep); ++i) {
    // random spin in vector range
    int s = rndDist(rndGen) * N;
    int nn_sum = 0;
    for (int n= 0; n < nn[0].size(); ++n) {
      nn_sum += spins[nn[s][n]];
    }
    double cost = 2 * J * spins[s] * nn_sum;
    if (cost < 0) {
      spins[s] *= -1;
    } else {
      if (rndDist(rndGen) < exp(-beta*cost)) {
        spins[s] *= -1;
      }
    }
  }

  // effective blocks
  for (int b = 0; b < Nblock; ++b) {
    for ( int i = 0; i < (5*Nstep); ++i) {
      // random spin in vector range
      int s = rndDist(rndGen) * N;
      int nn_sum = 0;
      for (int n= 0; n < nn[0].size(); ++n) {
        nn_sum += spins[nn[s][n]];
      }
      double cost = 2 * J * spins[s] * nn_sum;
      if (cost < 0) {
        spins[s] *= -1;
      } else {
        if (rndDist(rndGen) < exp(-beta*cost)) {
          spins[s] *= -1;
        }
      }
    }
    block_spin_avgs[b] =
      fabs(std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size());
  }

  return block_spin_avgs;
}

//
// MAIN
//

// TODO remove from head of file lattice dimensions and random stuff

int main() {
  // lattice dimension (only even number please)
  // int L = 32;

  // every step try a flip of a random spin
  int Nstep = 100 * L * L;
  int Nblock = L;

  double J = 1.;
  // double beta_critical = log(1 + sqrt(2)) / (2 * J);
  double T = 2.5;
  double T_stop = 4.5;
  double T_step = 0.05;

  std::cout << "Ising simulation with a square lattice "
    << L << "x" << L << std::endl;

  std::cout << "Initializing nearest neighbors table..." << std::endl;
  nn_array(L);

  std::cout << "Looping simulation on different β values...\n" << std::endl;
  for (int i = 0; T <= T_stop; ++i) {
    std::vector<double> block_spin_avgs = ising_2D(L*L, J, 1/T, Nblock, Nstep);

    // compute mean value of spin for last spin config
    double magnetization =
      std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size();

    // compute mean value of block avgs vector
    double meanM =
      std::accumulate(std::begin(block_spin_avgs), std::end(block_spin_avgs), 0.0) /
      block_spin_avgs.size();

    // compute variance of block avgs vector
    double accum = 0.0;
    std::for_each (
        std::begin(block_spin_avgs),
        std::end(block_spin_avgs),
        [&](const double d) { accum += (d - meanM) * (d - meanM); });
    double meanV = accum / block_spin_avgs.size();

    std::cout << magnetization << " " << T << "\n";
    std::cerr << T << " " << meanM << " " << meanV << std::endl;

    for (int s = 0; s < spins.size(); ++s) {
      std::cout << spins[s] << " ";
    }
    std::cout << "\n";

    T += T_step;
  }

  return 0;
}
