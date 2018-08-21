#!/bin/bash

for i in {1..313}; do
	./a.out >> out1.dat &
	./a.out >> out2.dat &
	./a.out >> out3.dat &
	./a.out >> out4.dat &
	wait
done
