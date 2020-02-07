using ImageMagick
using Distributed, SharedArrays, JLD, Test
using ImageCore, ImageAxes, TestImages, StaticArrays, PaddedViews
using AxisArrays: AxisArray
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon

nt = 3  # number of time points
wpids = addprocs(min(nt, 3))
@everywhere using RegisterWorkerAperturesMismatch, RegisterDriver

include("overall.jl") # from BlockRegistration.jl
include("aperturedmm.jl")

rmprocs(wpids)
