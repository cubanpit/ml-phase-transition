#!/bin/bash

for i in {1..300}; do
  for j in {1..6}; do
    ./a.out >> out$j.dat &
  done
  wait
  echo "Round $i"
done

echo "$(date) --- Done"

exit 0
