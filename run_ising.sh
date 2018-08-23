#!/bin/bash

for i in {1..313}; do
  for j in {1..6}; do
    ./a.out >> out$j.dat &
  done
  wait
done
