using SharedArrays, JLD, Test
using ImageCore, ImageAxes, TestImages, StaticArrays, PaddedViews
using AxisArrays: AxisArray
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon
using RegisterWorkerAperturesMismatch, RegisterDriver

nt = 3  # number of time points
wtids = threadids()

include("overall.jl") # from BlockRegistration.jl
include("aperturedmm.jl")
include("internals.jl")
