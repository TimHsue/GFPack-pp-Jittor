#!/bin/bash
g++ -std=c++17 -O3 -march=native -finline-functions -funroll-loops -Wall -fPIC -fopenmp -shared `python3 -m pybind11 --includes` util.cpp clipper.engine.cpp clipper.offset.cpp clipper.rectclip.cpp -o calutil`python3-config --extension-suffix`
