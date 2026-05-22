using SharedArrays, JLD, Test, CUDA
using ImageCore, ImageAxes, TestImages, StaticArrays, PaddedViews
using AxisArrays: AxisArray
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon
using RegisterWorkerAperturesMismatch, RegisterDriver
using Aqua
using ExplicitImports

@testset "Aqua" begin
    Aqua.test_all(RegisterWorkerAperturesMismatch;
        stale_deps=(; ignore=[:CUDA, :RegisterMismatchCuda]),
        deps_compat=(; check_extras=false),
        piracies=(; treat_as_own=[RegisterWorkerAperturesMismatch.load_mm_package]),
        # AtomixCUDAExt declares __precompile__(false), which is disallowed during
        # precompilation on Julia 1.10, causing a spurious persistent_tasks failure.
        persistent_tasks=VERSION >= v"1.11")
end

@testset "ExplicitImports" begin
    test_explicit_imports(RegisterWorkerAperturesMismatch)
end

nt = 3  # number of time points
wtids = threadids()

include("overall.jl") # from BlockRegistration.jl
include("aperturedmm.jl")
include("internals.jl")
