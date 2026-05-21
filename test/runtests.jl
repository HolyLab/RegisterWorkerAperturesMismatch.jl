using SharedArrays, JLD, Test
using ImageCore, ImageAxes, TestImages, StaticArrays, PaddedViews
using AxisArrays: AxisArray
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon
using RegisterWorkerAperturesMismatch, RegisterDriver
using Aqua

@testset "Aqua" begin
    Aqua.test_all(RegisterWorkerAperturesMismatch;
        stale_deps=(; ignore=[:CUDA, :RegisterMismatchCuda]),
        deps_compat=(; check_extras=false),
        piracies=(; treat_as_own=[RegisterWorkerAperturesMismatch.load_mm_package]))
end

nt = 3  # number of time points
wtids = threadids()

include("overall.jl") # from BlockRegistration.jl
include("aperturedmm.jl")
include("internals.jl")
