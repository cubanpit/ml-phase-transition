// Wolff cluster algorithm for the 3-D Ising Model

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random> // genearator and dist
#include <vector> // std::vector
#include <algorithm> // std::for_each
#include <sys/time.h>// time in microseconds

double J = +1;                  // ferromagnetic coupling
int Lx, Ly, Lz;                 // number of spins in x, y, z
int N;                          // number of spins
int ***s;                        // the spins
double T;                       // temperature
int steps;                      // number of Monte Carlo steps

std::mt19937 rndGen(std::random_device{}());
std::uniform_real_distribution<double> rndDist(0,1);

void initialize ( ) {

  s = new int** [Lx];
  for (int i = 0; i < Lx; i++) {
    s[i] = new int* [Ly];
    for (int j = 0; j < Ly; j++) {
      s[i][j] = new int [Lz];
    }
  }
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      for (int k = 0; k < Lz; k++) {
        //s[i][j][k] = rndDist(rndGen) < 0.5 ? +1 : -1;   // hot start
        s[i][j][k] = +1;   // cold start
      }
    }
  }
  steps = 0;
}

bool ***cluster;                     // cluster[i][j] = true if i,j belongs
double addProbability;              // 1 - e^(-2J/kT)

void initializeClusterVariables() {

  // allocate 2-D array for spin cluster labels
  cluster = new bool** [Lx];
  for (int i = 0; i < Lx; i++) {
    cluster[i] = new bool* [Ly];
    for (int j = 0; j < Ly; j++) {
      cluster[i][j] = new bool [Lz];
    }
  }

  // compute the probability to add a like spin to the cluster
  addProbability = 1 - exp(-2*J/T);
}

// declare functions to implement Wolff algorithm
void growCluster(int i, int j, int k, int clusterSpin);
void tryAdd(int i, int j, int k, int clusterSpin);

void oneMonteCarloStep() {

  //no cluster defined so clear the cluster array
  for (int i = 0; i < Lx; i++) {
    for (int j = 0; j < Ly; j++) {
      for (int k = 0; k < Lz; k++) {
        cluster[i][j][k] = false;
      }
    }
  }

  // choose a random spin and grow a cluster
  int i = int(rndDist(rndGen) * Lx);
  int j = int(rndDist(rndGen) * Ly);
  int k = int(rndDist(rndGen) * Lz);
  growCluster(i, j, k, s[i][j][k]);

  ++steps;
}

void growCluster(int i, int j, int k, int clusterSpin) {

  // mark the spin as belonging to the cluster and flip it
  cluster[i][j][k] = true;
  s[i][j][k] = -s[i][j][k];

  // find the indices of the 4 neighbors
  // assuming periodic boundary conditions
  int iPrev = i == 0    ? Lx-1 : i-1;
  int iNext = i == Lx-1 ? 0    : i+1;
  int jPrev = j == 0    ? Ly-1 : j-1;
  int jNext = j == Ly-1 ? 0    : j+1;
  int kPrev = k == 0    ? Lz-1 : k-1;
  int kNext = k == Lz-1 ? 0    : k+1;

  // if the neighbor spin does not belong to the
  // cluster, then try to add it to the cluster
  if (!cluster[iPrev][j][k])
    tryAdd(iPrev, j, k, clusterSpin);
  if (!cluster[iNext][j][k])
    tryAdd(iNext, j, k, clusterSpin);
  if (!cluster[i][jPrev][k])
    tryAdd(i, jPrev, k, clusterSpin);
  if (!cluster[i][jNext][k])
    tryAdd(i, jNext, k, clusterSpin);
  if (!cluster[i][j][kPrev])
    tryAdd(i, j, kPrev, clusterSpin);
  if (!cluster[i][j][kNext])
    tryAdd(i, j, kNext, clusterSpin);
}

void tryAdd(int i, int j, int k, int clusterSpin) {

  if (s[i][j][k] == clusterSpin) {
    if (rndDist(rndGen) < addProbability) {
      growCluster(i, j, k, clusterSpin);
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
      for (int k = 0; k < Lz; k++) {
        M += s[i][j][k];
      }
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
      for (int k = 0; k < Lz; k++) {
        M += s[i][j][k];
      }
    }
  }
  magnetization = double(M) / double(N);
}

int main() {

  Lx = 10;
  Ly = 10;
  Lz = 9;
  N = Lx * Ly * Lz;
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

  double Tc = 1/0.221654;   // critical temperature
  double Tstart = 1;                 // start temperature
  int Tn = 40;              // number of different temperatures (even number)
  double Tstep = 2 * (Tc - Tstart) / (Tn - 1); // step amplitude

  // get time in microseconds and use it as seed
  //struct timeval tv;
  //gettimeofday(&tv,NULL);
  //rndGen.seed(tv.tv_usec);

  T = Tstart;
  for (int t = 0; t < Tn; ++t) {

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
        for (int k = 0; k < Lz; k++) {
          std::cout << s[i][j][k] << " ";
        }
      }
    }
    std::cout << "\n";

    T += Tstep;
  }
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
