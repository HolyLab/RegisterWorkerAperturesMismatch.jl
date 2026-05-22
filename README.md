# RegisterWorkerAperturesMismatch.jl

[![CI](https://github.com/HolyLab/RegisterWorkerAperturesMismatch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterWorkerAperturesMismatch.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/RegisterWorkerAperturesMismatch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterWorkerAperturesMismatch.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![version](https://img.shields.io/github/v/release/HolyLab/RegisterWorkerAperturesMismatch.jl)](https://github.com/HolyLab/RegisterWorkerAperturesMismatch.jl/releases)

Worker for aperture-based (blocked) image registration using mismatch polynomials.
The image domain is divided into a grid of apertures; a local shift is estimated
for each aperture by fitting a quadratic to the cross-correlation mismatch array.
Results are written to disk for downstream optimization with
[RegisterOptimize](https://github.com/HolyLab/RegisterOptimize.jl).

This package is similar to
[RegisterWorkerApertures](https://github.com/HolyLab/RegisterWorkerApertures.jl)
(see its documentation for a broader overview), but targets **whole-experiment**
rather than stack-by-stack registration.
Because the temporal regularization in `RegisterOptimize` enforces smoothness,
this approach is **not currently recommended** for data sets with discontinuous
movements.

CUDA-accelerated computation is supported via the `dev` keyword argument.

## Installation

This package is registered in the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install normally:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterWorkerAperturesMismatch")
```

## Usage

```julia
using RegisterWorkerAperturesMismatch

# Define a reference image and aperture grid
fixed  = Float32.(reshape(1:64*64, 64, 64)) ./ (64f0 * 64f0)
nodes  = (range(1, 64, length=4), range(1, 64, length=4))  # 4×4 aperture grid
maxshift = (5, 5)

# Build the worker and monitoring dict
alg = AperturesMismatch(fixed, nodes, maxshift)
mon = monitor(alg, (:Es, :cs, :Qs))

# Register a moving image
moving = fixed .+ 0.01f0
mon = worker(alg, moving, 1, mon)

# Inspect results
size(mon[:Es])   # (4, 4) — per-aperture mismatch energy
size(mon[:cs])   # (4, 4) — per-aperture shift estimates (SVector{2})
size(mon[:Qs])   # (4, 4) — per-aperture curvature matrices (SMatrix{2,2})
```

For a preprocessing function (applied to both `fixed` and each `moving` frame
before computing the mismatch):

```julia
pp    = img -> img ./ (maximum(img) + eps(eltype(img)))
fixed = pp(raw_fixed)
alg   = AperturesMismatch(fixed, nodes, maxshift, pp)
```

For CUDA-accelerated registration, pass the device index:

```julia
alg = AperturesMismatch(fixed, nodes, maxshift; dev=0)
```

See the docstrings (`?AperturesMismatch`, `?worker`) for the full keyword reference.
