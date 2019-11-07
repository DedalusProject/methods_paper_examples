#!/usr/bin/env bash

mpiexec -n 1 python3 density_profile.py
mpiexec -n 1 python3 density_slice.py

