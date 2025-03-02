module RegisterWorkerAperturesMismatch

using ImageCore, CoordinateTransformations, Interpolations, StaticArrays, SharedArrays
using RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize
using RegisterMismatchCommon
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using RegisterWorkerShell # , RegisterDriver

import RegisterWorkerShell: worker, init!, close!, load_mm_package

export AperturesMismatch, monitor, monitor!, worker, workerpid

mutable struct AperturesMismatch{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    nodes::NTuple{N,K}
    maxshift::NTuple{N,Int}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    Es
    cs
    Qs
    mmis
    workerpid::Int
    dev::Int
    cuda_objects::Dict{Symbol,Any}
end

function load_mm_package(dev)
    if dev >= 0
        eval(:(using CUDA, RegisterMismatchCuda))
    else
        eval(:(using RegisterMismatch))
    end
    nothing
end

function init!(algorithm::AperturesMismatch)
    if algorithm.dev >= 0
        cuda_init!(algorithm)
    end
    nothing
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
    d_fixed  = CuArray{T}(sdata(fixed))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.nodes)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    algorithm.cuda_objects[:cms] = CMStorage{T}(undef, aperture_width, algorithm.maxshift)
end

function close!(algorithm::AperturesMismatch)
    if algorithm.dev >= 0
        if old_active_context != nothing
            activate(old_active_context)
        end
    end
    nothing
end

"""

`alg = AperturesMismatch(fixed, nodes, maxshift, [preprocess=identity];
kwargs...)` creates a worker-object for performing "apertured"
(blocked) registration.  `fixed` is the reference image, `nodes`
specifies the grid of apertures, `maxshift` represents the largest
shift (in pixels) that will be evaluated, and `preprocess` allows you
to apply a transformation (e.g., filtering) to the `moving` images
before registration; `fixed` should already have any such
transformations applied.

Registration is performed by calling `driver`.  You should monitor
`Es`, `cs`, `Qs`, and `mmis`.

## Example

Suppose your images are somewhat noisy, in which case a bit of
smoothing might help considerably.  Here we'll illustrate the use of a
pre-processing function, but see also `PreprocessSNF`.

```
   # Raw images are fixed0 and moving0, both two-dimensional
   pp = img -> imfilter_gaussian(img, [3, 3])
   fixed = pp(fixed0)
   # We'll use a 5x7 grid of apertures
   nodes = (linspace(1, size(fixed,1), 5), linspace(1, size(fixed,2), 7))
   # Allow shifts of up to 30 pixels in any direction
   maxshift = (30,30)

   # Create the algorithm-object
   alg = AperturesMismatch(fixed, nodes, maxshift, pp)

   mon = monitor(alg, (:Es, :cs, :Qs, :mmis))

   # Run the algorithm
   mon = driver(algorithm, moving0, mon)
```

"""
function AperturesMismatch(fixed, nodes::NTuple{N,K}, maxshift, preprocess=identity; normalization=:pixels, thresh_fac=(0.5)^ndims(fixed), thresh=nothing, correctbias::Bool=true, pid=1, dev=-1) where {K,N}
    gridsize = map(length, nodes)
    nimages(fixed) == 1 || error("Register to a single image")
    if thresh == nothing
        thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
    end
    T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    # T = Float64   # Ipopt requires Float64
    Es = ArrayDecl(Array{T,N}, gridsize)
    cs = ArrayDecl(Array{SVector{N,T},N}, gridsize)
    Qs = ArrayDecl(Array{similar_type(SMatrix, T, Size(N,N)),N}, gridsize)
    mmsize = map(x->2x+1, maxshift)
    mmis = ArrayDecl(Array{NumDenom{T},2*N}, (mmsize...,gridsize...))
    AperturesMismatch{typeof(fixed),T,K,N}(fixed, nodes, maxshift, T(thresh), preprocess, normalization, correctbias, Es, cs, Qs, mmis, pid, dev, Dict{Symbol,Any}())
end

function worker(algorithm::AperturesMismatch, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
    moving = algorithm.preprocess(moving0)
    gridsize = map(length, algorithm.nodes)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device!(CuDevice(algorithm.dev))
        d_fixed  = algorithm.cuda_objects[:d_fixed]
        d_moving = algorithm.cuda_objects[:d_moving]
        cms      = algorithm.cuda_objects[:cms]
        copyto!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(map(d->size(img,d),cs), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
        mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    Es = zeros(size(mms))
    cs = Array{Any}(undef, size(mms))
    Qs = Array{Any}(undef, size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        Es[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    monitor!(mon, :Es, Es)
    monitor!(mon, :cs, cs)
    monitor!(mon, :Qs, Qs)
    if haskey(mon, :mmis)
        mmis = interpolate_mm!(mms)
        R = CartesianIndices(size(mon[:mmis])[ndims(mmis)+1:end])
        colons = ntuple(d->Colon(), ndims(mmis))
        _copy_mm!(mon[:mmis], mmis, colons, R)
    end
    mon
end

function _copy_mm!(dest, src, colons, R)
    for (I, mm) in zip(R, src)
        dest[colons..., I] = mm.data.coefs
    end
    dest
end

cudatype(::Type{T}) where {T<:Union{Float32,Float64}} = T
cudatype(::Any) = Float32

myconvert(::Type{Array{T}}, A::Array{T}) where {T} = A
myconvert(::Type{Array{T}}, A::AbstractArray) where {T} = copyto!(Array{T}(undef, size(A)), A)

end # module
