// Wolff cluster algorithm for the 2-D Ising Model on honeycomb lattice

#include <cmath>
#include <iostream>
#include <random> // genearator and dist
#include <vector> // std::vector
#include <array> // std::array
#include <algorithm> // std::for_each

// On a square lattice there are two honeycomb lattice, we connect them with
//  boundary conditions in one direction (up-down).
// The only downside is that not all Ly values are allowed, some values connect
//  with boundary conditions only one honeycomb lattice and they can develop
//  different phases at low temperature.
//
// Working Ly are 30, 42, 50, 62, 70, 82
// It is necessary to check the cycle printing spins array at the end of the
//  main(), if you want to output a number of spin compatible with other square
//  lattice. (40x40, 60x60, etc.)
#define Lx 40
#define Ly 42
const unsigned int N = Lx * Ly;   // number of spins

// Here we approx at power of 10, for personal project
unsigned int Ly_d = (Ly / 10) * 10;

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

  // Find the indices of the 3 neighbors
  //  assuming periodic boundary conditions.
  // This method generates two indipendent honeycomb lattice.
  if (j % 2 == 0) {
    int iPrev = i == 0    ? Lx-1 : i-1;
    int iNext = i == Lx-1 ? 0    : i+1;
    int jPrev = j == 0    ? Ly-1 : j-1;
    int jNext = j == Ly-1 ? 0    : j+1;

    // if the neighbor spin does not belong to the
    // cluster, then try to add it to the cluster
    if (!cluster[i][jPrev])
      tryAdd(i, jPrev, clusterSpin);
    if (!cluster[iPrev][jNext])
      tryAdd(iPrev, jNext, clusterSpin);
    if (!cluster[iNext][jNext])
      tryAdd(iNext, jNext, clusterSpin);
  } else {
    int iPrev = i == 0    ? Lx-1 : i-1;
    int iNext = i == Lx-1 ? 0    : i+1;
    int jPrev = j == 0    ? Ly-1 : j-1;
    int jNext = j == Ly-1 ? 0    : j+1;

    // if the neighbor spin does not belong to the
    // cluster, then try to add it to the cluster
    if (!cluster[iPrev][jPrev])
      tryAdd(iPrev, jPrev, clusterSpin);
    if (!cluster[iNext][jPrev])
      tryAdd(iNext, jPrev, clusterSpin);
    if (!cluster[i][jNext])
      tryAdd(i, jNext, clusterSpin);
  }
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

  // We have two honeycomb lattice, we look at both of them.

  // compute mean spin value
  int M = 0;
  for (int j = 0; j < Ly_d; j++) {
    for (int i = 0; i < Lx; i++) {
      M += s[i][j];
    }
  }
  magnetization = double(M) / double(N);
}

int main() {

  int MCSteps = 10000;
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

  double Tc = 1 / 0.658478;   // critical temperature
  double Tstart = 0;                 // start temperature
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
      for (int i = 0; i < blockMvector.size(); ++i) {
        blockMvector[i] = fabs(blockMvector[i]);
      }
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

    // Print (quasi)full spin lattice, we want a square lattice to compare
    //  spins array to other models. (40x40, 60x60, etc.)
    // For further information see head of file.
    for (int j = 0; j < Ly_d; j++) {
      for (int i = 0; i < Lx; i++) {
        std::cout << s[i][j] << " ";
      }
    }
    std::cout << "\n";

    T += Tstep;
  }

  return 0;
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
