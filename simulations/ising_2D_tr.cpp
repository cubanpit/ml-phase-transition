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
#include <sys/time.h>

//
// MAIN
//

// TODO put MC stuff in function, for more readability

int main() {
  // random generator and distribution
  std::mt19937 rndGen;
  std::uniform_real_distribution<double> rndDist(0,1);

  // array of spins
  unsigned int Lx = 60;
  unsigned int Ly = 60;
  // total spin number
  unsigned int N = Lx * Ly;
  std::vector<short int> spins(N);

  // every step try a flip of a random spin
  unsigned int Nstep = 100;
  unsigned int Nblock = 100;
  bool computeBlockValues = true;

  double J = 1.;                     // coupling costant
  double Tc = 4 / log(3);            // critical temperature
  double Tstart = 2;                 // start temperature
  unsigned int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  // nearest neighbour vector
  unsigned int nn[N][6];
  // initialize nn vector
  for (int i = 0; i < N; ++i) {
    unsigned int xRef = i % Lx;
    unsigned int yRef = int(i / Lx);
    unsigned int xPrev = xRef == 0    ? Lx-1 : xRef-1;
    unsigned int yPrev = yRef == 0    ? Ly-1 : yRef-1;
    unsigned int xNext = xRef == Lx-1 ? 0    : xRef+1;
    unsigned int yNext = yRef == Ly-1 ? 0    : yRef+1;
    nn[i][1] = xPrev + Lx * yRef;
    nn[i][2] = xNext + Lx * yRef;
    nn[i][3] = xRef + Lx * yPrev;
    nn[i][4] = xRef + Lx * yNext;
    nn[i][5] = xPrev + Lx * yPrev;
    nn[i][6] = xNext + Lx * yNext;
    // std::cout << i << " -> [ " << nn[i][1] << " " << nn[i][2] << " "
    //  << nn[i][3] << " " << nn[i][4] << " " << nn[i][5] << " " << nn[i][6]
    //  << " ]"  << std::endl;
  }


  double T = Tstart;
  for (int t = 0; t < Tn; ++t) {
    // get time in microseconds and use it as seed
    struct timeval tv;
    gettimeofday(&tv,NULL);
    rndGen.seed(tv.tv_usec);

    double beta = 1 / T;

    // initialize all spins - cold start
    std::fill(spins.begin(), spins.end(), 1);

    // randomize all spins - hot start
    //for (int i = 0; i < N; ++i) {
    //  spins[i] = int(rndDist(rndGen) + 0.5) * 2 - 1;
    //}

    // thermalization steps
    for (int i = 0; i < (5*Nstep) and computeBlockValues; ++i) {
      for (int s = 0; s < N; ++s) {
        // random spin in vector range
        //int s = rndDist(rndGen) * N;
        int nn_sum = 0;
        for (int n= 0; n < 6; ++n) {
          nn_sum += spins[nn[s][n]];
        }
        double cost = 2 * J * spins[s] * nn_sum;
        if (cost < 0) {
          spins[s] *= -1;
        } else {
          if (rndDist(rndGen) < exp(- beta * cost)) {
            spins[s] *= -1;
          }
        }
      }
    }

    // vector of block measured values
    std::vector<double> block_spin_avgs(Nblock);

    // effective blocks
    for (int b = 0; b < Nblock; ++b) {
      for ( int i = 0; i < (Nstep); ++i) {
        for (int s = 0; s < N; ++s) {
          // random spin in vector range
          //int s = rndDist(rndGen) * N;
          int nn_sum = 0;
          for (int n= 0; n < 6; ++n) {
            nn_sum += spins[nn[s][n]];
          }
          double cost = 2 * J * spins[s] * nn_sum;
          if (cost < 0) {
            spins[s] *= -1;
          } else {
            if (rndDist(rndGen) < exp(- beta * cost)) {
              spins[s] *= -1;
            }
          }
        }
      }
      block_spin_avgs[b] =
        fabs(std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size());
    }

    // compute mean value of spin for last spin config
    double magnetization =
      std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size();

    if (computeBlockValues) {
      // compute mean value of block avgs vector
      double meanM =
        std::accumulate(std::begin(block_spin_avgs), std::end(block_spin_avgs), 0.0) /
        block_spin_avgs.size();

      // compute variance of block avgs vector
      double accum = 0.;
      std::for_each (
          std::begin(block_spin_avgs),
          std::end(block_spin_avgs),
          [&](const double d) { accum += (d - meanM) * (d - meanM); });
      double meanV = accum / block_spin_avgs.size();

      std::cerr << T << " " << meanM << " " << meanV << std::endl;
    }
    std::cout << magnetization << " " << T << "\n";

    for (int s = 0; s < spins.size(); ++s) {
      std::cout << spins[s] << " ";
    }
    std::cout << "\n";

    T += Tstep;
  }

  return 0;
}
