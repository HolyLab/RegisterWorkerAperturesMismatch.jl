"""
    RegisterWorkerAperturesMismatch

Worker for aperture-based (blocked) image registration using mismatch polynomials.

Divides the image domain into a grid of apertures and estimates a local shift for
each aperture by fitting a quadratic to the mismatch array. Implements the
[`AbstractWorker`](@ref RegisterWorkerShell.AbstractWorker) interface from
`RegisterWorkerShell`; the primary entry points are [`AperturesMismatch`](@ref)
(constructor) and [`worker`](@ref) (single-frame registration).

CUDA-accelerated computation is supported when `dev` is set to a device index.
"""
module RegisterWorkerAperturesMismatch

using CoordinateTransformations: CoordinateTransformations
using ImageCore: ImageCore, coords_spatial, nimages
using Interpolations: Interpolations
using RegisterCore: RegisterCore, NumDenom, maxshift
using RegisterDeformation: RegisterDeformation
using RegisterFit: RegisterFit, qfit
using RegisterMismatch: RegisterMismatch, CMStorage, mismatch_apertures!
using RegisterMismatchCommon: RegisterMismatchCommon, allocate_mmarrays, aperture_grid,
                              correctbias, correctbias!, default_aperture_width, mismatch_apertures
using RegisterOptimize: RegisterOptimize
using RegisterPenalty: RegisterPenalty, interpolate_mm!
using RegisterWorkerShell: RegisterWorkerShell, AbstractWorker, ArrayDecl, getindex_t,
                           monitor, monitor!
using SharedArrays: SharedArrays, sdata
using StaticArrays: StaticArrays, SMatrix, SVector, Size, similar_type
# Note: RegisterMismatchCuda is loaded dynamically below when dev !== nothing

import RegisterWorkerShell: worker, init!, close!, load_mm_package, workertid

export AperturesMismatch, monitor, monitor!, worker

struct AperturesMismatch{A <: AbstractArray, T, N} <: AbstractWorker
    fixed::A
    nodes::NTuple{N}
    maxshift::NTuple{N, Int}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    Es
    cs
    Qs
    mmis
    tid::Int
    dev::Union{Nothing, Int}
    cuda_objects::Dict{Symbol, Any}
end

workertid(w::AperturesMismatch) = w.tid

function load_mm_package(dev)
    if dev !== nothing
        eval(:(using CUDA, RegisterMismatchCuda))
    end
    return nothing
end

function init!(algorithm::AperturesMismatch)
    if algorithm.dev !== nothing
        cuda_init!(algorithm)
    end
    return nothing
end

function cuda_init!(algorithm)
    dev = CuDevice(algorithm.dev)
    global old_active_context
    try
        old_active_context = current_context()
        if old_active_context == nothing || device(old_active_context) != dev
            device!(dev)
        end
    catch e
        old_active_context = nothing
        device!(dev)
    end
    fixed = algorithm.fixed
    T = cudatype(eltype(fixed))
    d_fixed = CuArray{T}(sdata(fixed))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.nodes)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    return algorithm.cuda_objects[:cms] = CMStorage{T}(undef, aperture_width, algorithm.maxshift)
end

function close!(algorithm::AperturesMismatch)
    if algorithm.dev !== nothing
        if !isnothing(old_active_context)
            activate(old_active_context)
        end
    end
    return nothing
end

"""
    AperturesMismatch(fixed, nodes, maxshift, preprocess=identity;
                      normalization=:pixels, thresh_fac=0.5^ndims(fixed),
                      thresh=nothing, correctbias=true, tid=1, dev=nothing)

Create a worker object for aperture-based (blocked) image registration.

`fixed` is the reference image. `nodes` is an `N`-tuple of ranges or
vectors specifying the aperture grid along each spatial dimension.
`maxshift` is an `N`-tuple of integers giving the maximum shift (in
pixels) to evaluate along each dimension. `preprocess` is an optional
function applied to each `moving` image before registration; `fixed`
should already have the same transformation applied.

# Keyword arguments

- `normalization`: `:pixels` (default) normalizes mismatch by the number
  of pixels in each aperture; `:intensity` normalizes by image intensity.
- `thresh_fac`: sets the default threshold as `thresh_fac / prod(gridsize)`
  times the image norm. Ignored when `thresh` is supplied explicitly.
- `thresh`: minimum mismatch energy required to fit a quadratic. Apertures
  below threshold are skipped. Defaults to a value derived from `thresh_fac`.
- `correctbias`: if `true` (default), applies bias correction to the
  mismatch arrays before fitting.
- `tid`: worker thread id (default `1`).
- `dev`: CUDA device index (`Int`). If `nothing` (default), runs on CPU.

The returned object is an `AbstractWorker` subtype. Use [`monitor`](@ref)
to create a monitoring dict, then [`worker`](@ref) (or `driver`) to run
registration. The key monitored quantities are:
- `:Es` — per-aperture mismatch energy
- `:cs` — per-aperture shift estimates
- `:Qs` — per-aperture quadratic curvature matrices
- `:mmis` — interpolated mismatch arrays (optional, expensive to store)

# Examples

Basic usage with a 4×4 aperture grid:

```julia
fixed = rand(Float32, 64, 64)
nodes = (range(1, 64, length=4), range(1, 64, length=4))
maxshift = (5, 5)
alg = AperturesMismatch(fixed, nodes, maxshift)
mon = monitor(alg, (:Es, :cs, :Qs))
moving = rand(Float32, 64, 64)
mon = worker(alg, moving, 1, mon)
size(mon[:Es])  # (4, 4)
```

With a preprocessing function applied to both `fixed` and each `moving` image:

```julia
fixed0 = rand(Float32, 64, 64)
pp = img -> img ./ (maximum(img) + eps(Float32))
fixed = pp(fixed0)
nodes = (range(1, 64, length=5), range(1, 64, length=7))
alg = AperturesMismatch(fixed, nodes, (10, 10), pp)
mon = monitor(alg, (:Es, :cs, :Qs))
moving0 = rand(Float32, 64, 64)
mon = worker(alg, moving0, 1, mon)
size(mon[:cs])  # (5, 7)
```
"""
function AperturesMismatch(fixed, nodes::NTuple{N}, maxshift::NTuple{N, <:Integer}, preprocess = identity; normalization = :pixels, thresh_fac = (0.5)^ndims(fixed), thresh = nothing, correctbias::Bool = true, tid = 1, dev = nothing) where {N}
    gridsize = map(length, nodes)
    nimages(fixed) == 1 || error("Register to a single image")
    if isnothing(thresh)
        thresh = (thresh_fac / prod(gridsize)) * (normalization == :pixels ? length(fixed) : sumabs2(fixed))
    end
    T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    # T = Float64   # Ipopt requires Float64
    Es = ArrayDecl(Array{T, N}, gridsize)
    cs = ArrayDecl(Array{SVector{N, T}, N}, gridsize)
    Qs = ArrayDecl(Array{similar_type(SMatrix, T, Size(N, N)), N}, gridsize)
    mmsize = map(x -> 2x + 1, maxshift)
    mmis = ArrayDecl(Array{NumDenom{T}, 2 * N}, (mmsize..., gridsize...))
    return AperturesMismatch{typeof(fixed), T, N}(fixed, nodes, maxshift, T(thresh), preprocess, normalization, correctbias, Es, cs, Qs, mmis, tid, dev, Dict{Symbol, Any}())
end

"""
    worker(algorithm::AperturesMismatch, img, tindex, mon) -> mon

Perform aperture-based mismatch registration for a single image frame.

`img` is the source of moving images. If `img` has a time axis, `tindex`
selects the frame; otherwise `img` itself is used directly. `mon` is the
monitoring dict returned by `monitor`; any keys present in `mon` (`:Es`,
`:cs`, `:Qs`, `:mmis`) are updated with the results of this call.

Returns `mon` with updated fields:
- `:Es` — per-aperture mismatch energy (scalar per aperture)
- `:cs` — per-aperture shift estimates (`SVector{N}` per aperture)
- `:Qs` — per-aperture quadratic curvature matrices (`SMatrix{N,N}` per aperture)
- `:mmis` — interpolated mismatch arrays (present only if `:mmis` key was in `mon`)
"""
function worker(algorithm::AperturesMismatch, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
    moving = algorithm.preprocess(moving0)
    gridsize = map(length, algorithm.nodes)
    use_cuda = algorithm.dev !== nothing
    if use_cuda
        device!(CuDevice(algorithm.dev))
        d_fixed = algorithm.cuda_objects[:d_fixed]
        d_moving = algorithm.cuda_objects[:d_moving]
        cms = algorithm.cuda_objects[:cms]
        copyto!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(map(d -> size(img, d), cs), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization = algorithm.normalization)
    else
        mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization = algorithm.normalization)
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    T = eltype(algorithm.Es)
    N = length(gridsize)
    Es = zeros(T, gridsize...)
    cs = Array{SVector{N, T}}(undef, gridsize...)
    Qs = Array{similar_type(SMatrix, T, Size(N, N))}(undef, gridsize...)
    thresh = algorithm.thresh
    for i in 1:length(mms)
        Es[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt = false)
    end
    monitor!(mon, :Es, Es)
    monitor!(mon, :cs, cs)
    monitor!(mon, :Qs, Qs)
    if haskey(mon, :mmis)
        mmis = interpolate_mm!(mms)
        N = ndims(mmis)
        gridsize = size(mmis)
        coefs1 = first(mmis).data.coefs
        result = Array{eltype(coefs1)}(undef, size(coefs1)..., gridsize...)
        _copy_mm!(result, mmis, ntuple(_ -> Colon(), N), CartesianIndices(gridsize))
        monitor!(mon, :mmis, result)
    end
    return mon
end

function _copy_mm!(dest, src, colons, R)
    for (I, mm) in zip(R, src)
        dest[colons..., I] = mm.data.coefs
    end
    return dest
end

cudatype(::Type{T}) where {T <: Union{Float32, Float64}} = T
cudatype(::Any) = Float32

end # module
