# Kalman Filter
Variable dimension Kalman and its extensions for learning purposes

## Features
* Variable dimension
* Support system with and without control input
* Utilize automatic differentiation for extended Kalman filter with [autodiff](https://autodiff.github.io/)

## Build
```bash
cmake . -B build
make
make test
```

## Todo list
- [x] Basic funtionality
- [x] Extended Kalman filter
- [ ] Ensemble Kalman filter
- [ ] Bayesian filter