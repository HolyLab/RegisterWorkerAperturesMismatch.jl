using ImageMagick
using Distributed, SharedArrays, JLD, Test
using Images, TestImages, StaticArrays, PaddedViews
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon
using RegisterDriver, RegisterWorkerAperturesMismatch

nt = 3  # number of time points
wpids = addprocs(min(nt, 3))
for p in wpids
    @spawnat p eval(quote
        using Pkg
        Pkg.activate(".")
        Pkg.instantiate()
        using StaticArrays
        using RegisterWorkerAperturesMismatch
    end)
end

include("overall.jl") # from BlockRegistration.jl
include("aperturedmm.jl")

rmprocs(wpids)
