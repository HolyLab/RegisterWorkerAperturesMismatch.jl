using ImageMagick
using Distributed, SharedArrays, JLD, Test
using Images, TestImages, StaticArrays
using RegisterCore, RegisterOptimize, RegisterDeformation, RegisterPenalty
using RegisterMismatchCommon
using RegisterDriver, RegisterWorkerAperturesMismatch

include("aperturedmm.jl")
include("overall.jl") # from BlockRegistration.jl
