# RegisterWorkerAperturesMismatch.jl

This package is similar to [RegisterWorkerApertures](https://github.com/HolyLab/RegisterWorkerApertures.jl),
whose documentation you should consult for an overview.
This package differs in that it is targeted at "whole experiment" rather than
"stack-by-stack" registration.
It writes the mismatch data to disk, and then [RegisterOptimize](https://github.com/HolyLab/RegisterOptimize.jl) is used to optimize a time-series of deformations.

However, because the temporal regularization enforces smoothness, and many real-world
data sets have discontinuous movements, this approach is not currently recommended.
