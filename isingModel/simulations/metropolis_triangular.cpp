//
// This code simulates an Ising square lattice with triangular interaction
// and periodic boundary condition, at different temperatures, with Monte
// Carlo (Metropolis) algorithm.
//

#include <iostream>
#include <random>
#include <algorithm>
#include <array>

// lattice size
#define Lx 60
#define Ly 60
#define NN_SIZE 6         // number of nearest neighbour

double J = 1.;                     // coupling costant

// total spin number
const unsigned int N = Lx * Ly;
// array of spins
std::array<short int, N> spins;

// random generator and distribution
std::mt19937 rndGen(std::random_device{}());
std::uniform_real_distribution<double> rndDist(0,1);

double beta;
std::array<std::array<unsigned int, NN_SIZE>, N> nn;  // nearest neighbour array

void init_nn();
void metropolis(unsigned nsteps);
short int binGen();

int main() {

  // every step try a flip of a random spin N (=Lx*Ly) times
  unsigned int Nstep = 50;    // steps per block
  unsigned int Nblock = 100;  // block per temperature step
  unsigned int Nther = 10;    // thermalization blocks
  bool computeBlockValues = true;

  double Tc = 4 / log(3);            // critical temperature
  double Tstart = 2;                 // start temperature
  unsigned int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  // initialize nearest neighbours array
  init_nn();

  // initialize all spins - cold start
  std::fill(spins.begin(), spins.end(), 1);

  // randomize all spins - hot start
  //std::generate(spins.begin(), spins.end(), binGen);

  double T = Tstart;
  for (int t = 0; t < Tn; ++t) {

    beta = 1 / T;

    // thermalization steps
    if (computeBlockValues) {
      metropolis(Nther*Nstep);
    }

    // array of block measured values
    std::vector<double> block_spin_avgs(Nblock);

    // effective blocks
    for (int b = 0; b < Nblock; ++b) {
      metropolis(Nstep);
      if (computeBlockValues) {
        block_spin_avgs[b] =
          fabs(std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size());
      }
    }

    if (computeBlockValues) {
      // compute mean value of block avgs vector
      double meanM =
        std::accumulate(std::begin(block_spin_avgs), std::end(block_spin_avgs), 0.0) /
        block_spin_avgs.size();

      // compute variance of block avgs vector
      double accum = 0.;
      std::for_each (
          std::begin(block_spin_avgs), std::end(block_spin_avgs),
          [&](const double d) { accum += (d - meanM) * (d - meanM); });
      double meanV = accum / block_spin_avgs.size();

      std::cerr << T << " " << meanM << " " << meanV << "\n";
    }

    // compute mean value of spin for last spin config
    double magnetization =
      std::accumulate(std::begin(spins), std::end(spins), 0.0) / spins.size();
    std::cout << magnetization << " " << T << "\n";

    // print full spin configuration
    for (int s = 0; s < spins.size(); ++s) {
      std::cout << spins[s] << " ";
    }
    std::cout << "\n";

    T += Tstep;
  }

  return 0;
}


// initialize nearest neighbour array
// triangular lattice interaction have six nearest neighbour
// up, down, left, right, up-left, down-right
void init_nn() {
  for (unsigned int i = 0; i < N; ++i) {

    // central spin
    unsigned int xRef = i % Lx;
    unsigned int yRef = int(i / Lx);

    // relative positions
    unsigned int xPrev = xRef == 0    ? Lx-1 : xRef-1;
    unsigned int yPrev = yRef == 0    ? Ly-1 : yRef-1;
    unsigned int xNext = xRef == Lx-1 ? 0    : xRef+1;
    unsigned int yNext = yRef == Ly-1 ? 0    : yRef+1;

    nn[i][0] = xPrev + Lx * yRef;  // left
    nn[i][1] = xNext + Lx * yRef;  // right
    nn[i][2] = xRef + Lx * yPrev;  // up
    nn[i][3] = xRef + Lx * yNext;  // down
    nn[i][4] = xPrev + Lx * yPrev; // up - left
    nn[i][5] = xNext + Lx * yNext; // down - right
  }
}

// apply standard metropolis for a number of step
// every step is repeated N (=Lx*Ly) times
void metropolis(unsigned int nsteps) {
  for (unsigned int i = 0; i < nsteps; ++i) {
    for (unsigned int j = 0; j < N; ++j) {

      // random spin in array range
      unsigned int s = int(rndDist(rndGen) * N);

      // sum nearest neighbours
      int nn_sum = 0;
      std::for_each (
          std::begin(nn[s]), std::end(nn[s]),
          [&](const unsigned int n) { nn_sum += spins[n]; });

      // compute energy cost and apply metropolis
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
}

// generate +1 or -1 based on a random distribution
short int binGen() {
    return rndDist(rndGen) < 0.5 ? +1 : -1;
}

/* Copyright 2018 Pietro F. Fontana <pietrofrancesco.fontana@studenti.unimi.it>
 *                Martina Crippa    <martina.crippa2@studenti.unimi.it>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/
