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
  int L = 30;
  // total spin number
  int N = L * L;
  std::vector<int> spins(N);

  // every step try a flip of a random spin
  int Nstep = 5 * L * L;
  int Nblock = 10 * L;
  bool computeBlockValues = false;

  double J = 1.;                     // coupling costant
  double Tc = 4 / log(3);            // critical temperature
  double Tstart = 2;                 // start temperature
  int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  // nearest neighbour vector
  int nn[N][6];
  // initialize nn vector
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

  double T = Tstart;
  for (int t = 0; t < Tn; ++t) {
    // get time in microseconds and use it as seed
    struct timeval tv;
    gettimeofday(&tv,NULL);
    rndGen.seed(tv.tv_usec);

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
      for (int n= 0; n < 6; ++n) {
        nn_sum += spins[nn[s][n]];
      }
      double cost = 2 * J * spins[s] * nn_sum;
      if (cost < 0) {
        spins[s] *= -1;
      } else {
        if (rndDist(rndGen) < exp(-cost/T)) {
          spins[s] *= -1;
        }
      }
    }

    // vector of block measured values
    std::vector<double> block_spin_avgs(Nblock);

    // effective blocks
    for (int b = 0; b < Nblock; ++b) {
      for ( int i = 0; i < (Nstep); ++i) {
        // random spin in vector range
        int s = rndDist(rndGen) * N;
        int nn_sum = 0;
        for (int n= 0; n < 6; ++n) {
          nn_sum += spins[nn[s][n]];
        }
        double cost = 2 * J * spins[s] * nn_sum;
        if (cost < 0) {
          spins[s] *= -1;
        } else {
          if (rndDist(rndGen) < exp(-cost/T)) {
            spins[s] *= -1;
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
