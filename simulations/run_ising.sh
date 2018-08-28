#!/bin/bash

for i in {1..125}; do
  for j in {1..2}; do
    ./a.out >> out$j.dat &
  done
  wait
  echo "Round $i"
done

echo "$(date) --- Done"

exit 0
