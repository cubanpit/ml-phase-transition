// Wolff cluster algorithm for the 2-D XY Model

#include <cmath>
#include <iostream>
#include <random> // genearator and dist
#include <vector> // std::vector
#include <array> // std::aray
#include <algorithm> // std::for_each

#define Lx 30
#define Ly 30                     // number of spins in x and y
const unsigned int N = Lx * Ly;   // number of spins

std::array<std::array<double, Ly>, Lx> s;     // the spins
std::array<std::array<bool, Ly>, Lx> cluster;    // cluster[i][j] = true if i,j belongs

double J = +1;                  // ferromagnetic coupling
double T;                       // temperature
double addProbability;

// random number generator and distribution
std::mt19937 rndGen(std::random_device{}());
std::uniform_real_distribution<double> rndDist(0,1);

double alpha; 					// random angle
double oldClusterSpin;
double newClusterSpin;

double mx, my, MAG;

void initialize ( ) {

  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      s[i][j] = rndDist(rndGen) * 2 * M_PI; // hot start
      //s[i][j] = 1;   // cold start
    }
  }

  mx = 0.0;
  my = 0.0;

  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      mx += cos(s[i][j]);
      my += sin(s[i][j]);
    }
  }
}

// declare functions to implement Wolff algorithm
void growCluster(int i, int j);
void tryAdd(int i, int j, double oldClusterSpin, double newClusterSpin);

void oneMonteCarloStep() {

  //no cluster defined so clear the cluster array
  for (int i = 0; i < Lx; i++) {
    std::fill(cluster[i].begin(), cluster[i].end(), false);
  }

  alpha = rndDist(rndGen)*M_PI;

  // choose a random spin and grow a cluster
  unsigned int i = int(rndDist(rndGen) * Lx);
  unsigned int j = int(rndDist(rndGen) * Ly);
  growCluster(i, j);
}

void growCluster(int i, int j) {

  // mark the spin as belonging to the cluster and flip it
  cluster[i][j] = true;
  oldClusterSpin = s[i][j];

  mx -= cos(oldClusterSpin);
  my -= sin(oldClusterSpin);

  double tmpSpin = 2 * alpha - s[i][j];
  if (tmpSpin < 0) {
    s[i][j] = 2 * M_PI + tmpSpin;
  } else {
    s[i][j] = fmod(tmpSpin, 2*M_PI);
  }

  newClusterSpin = s[i][j];

  mx += cos(newClusterSpin);
  my += sin(newClusterSpin);

  // find the indices of the 4 neighbors
  // assuming periodic boundary conditions
  int iPrev = i == 0    ? Lx-1 : i-1;
  int iNext = i == Lx-1 ? 0    : i+1;
  int jPrev = j == 0    ? Ly-1 : j-1;
  int jNext = j == Ly-1 ? 0    : j+1;

  // if the neighbor spin does not belong to the
  // cluster, then try to add it to the cluster
  if (!cluster[iPrev][j])
    tryAdd(iPrev, j, oldClusterSpin, newClusterSpin);
  if (!cluster[iNext][j])
    tryAdd(iNext, j, oldClusterSpin, newClusterSpin);
  if (!cluster[i][jPrev])
    tryAdd(i, jPrev, oldClusterSpin, newClusterSpin);
  if (!cluster[i][jNext])
     tryAdd(i, jNext, oldClusterSpin, newClusterSpin);
}

void tryAdd(int i, int j, double oldClusterSpin, double newClusterSpin) {

  addProbability = 1 - std::min(1., exp(-J * (cos(oldClusterSpin-s[i][j]) - cos(newClusterSpin-s[i][j])) / T));

  if (rndDist(rndGen) < addProbability) {
    growCluster(i, j);
  }
}

// various block interesting values
double blockM = 0; // magnetization, spin average
double blockV = 0; // spin variance
// declare mean spin value
double magnetization = 0;

// compute block averages, useful to understand model behaviour
void measureBlockObservables() {

  blockM = sqrt(mx * mx + my * my) / N;
}

int main() {

  int MCSteps = 100000;
  int blockSize = 1000; // suggested by Wolff is 1000

  // if true block values will be computed and printed on stderr
  // more information, more time
  // useful to adjust parameters (steps, block size)
  bool computeBlockValues = false;
  std::vector<double> blockMvector; // magnetization averages computed on blocks

  if (computeBlockValues) {
    MCSteps += int(MCSteps/5);
  }

  double Tc = 0.893;   // critical temperature
  double Tstart = 0.01;                 // start temperature
  int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  T = Tstart;
  for (int t = 0; t < Tn; ++t) {

    initialize(); // initialize spins array

    if (computeBlockValues) {
      // thermalize
      for (int i = 0; i < int(MCSteps/5); i++) {
        oneMonteCarloStep();
      }
    }

    // effective MC steps
    for (int i = 0; i < MCSteps; i++) {
      oneMonteCarloStep();
      if (computeBlockValues and i != 0 and i % blockSize == 0) {
        measureBlockObservables();
        blockMvector.push_back(blockM);
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

    std::cout << T << "\n";

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
