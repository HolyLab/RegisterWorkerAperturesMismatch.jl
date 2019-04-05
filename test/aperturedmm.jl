workdir = mktempdir()

### Apertured registration
# Create the data
fixed = testimage("cameraman")
gridsize = (5,5)
ntimes = 4
shift_amplitude = 5
u_dfm = shift_amplitude*randn(2, gridsize..., ntimes)
img = AxisArray(SharedArray{Float64}((size(fixed)..., ntimes), pids = union(myid(), wpids)), :y, :x, :time)
knots = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1,2))
tax = timeaxis(img)
for i = 1:ntimes
    ϕ_dfm = GridDeformation(u_dfm[:,:,:,i], knots)
    img[tax(i)] = warp(fixed, ϕ_dfm)
end
# Perform the registration
fn = joinpath(workdir, "apertured.jld")
maxshift = (3*shift_amplitude, 3*shift_amplitude)
algorithms = AperturesMismatch[AperturesMismatch(fixed, knots, maxshift; pid=p) for p in wpids]
mm_package_loader(algorithms)
mons = monitor(algorithms, (:Es, :cs, :Qs, :mmis))
driver(fn, algorithms, img, mons)

# With preprocessing
fn_pp = joinpath(workdir, "apertured_pp.jld")
pp = PreprocessSNF(0.1, [2,2], [10,10])
algorithms = AperturesMismatch[AperturesMismatch(pp(fixed), knots, maxshift, pp; pid=p) for p in wpids]
mm_package_loader(algorithms)
mons = monitor(algorithms, (:Es, :cs, :Qs, :mmis))
driver(fn_pp, algorithms, img, mons)

# using CUDA
if !(haskey(ENV,"CI")&&(ENV["CI"]=="true"))
    fn_cuda = joinpath(workdir, "apertured_cuda.jld")
    algorithm = AperturesMismatch(pp(fixed), knots, maxshift, pp; dev=0)
    mm_package_loader(algorithm)
    mons = monitor(algorithms, (:Es, :cs, :Qs, :mmis))
    driver(fn_cuda, algorithms, img, mons)
end

fns = [fn, fn_pp]
if (@isdefined fn_cuda)&&isfile(fn_cuda)
    push!(fns, fn_cuda)
end
for fname in fns
    jldopen(fname, "r") do file
        dEs, dcs, dQs, dmmis = file["Es"], file["cs"], file["Qs"], file["mmis"]
        for d in (dEs, dcs, dQs, dmmis)
            @test eltype(d) == Float32
        end
        @test size(dEs) == (gridsize..., ntimes)
        @test size(dcs) == (2, gridsize..., ntimes)
        @test size(dQs) == (2, 2, gridsize..., ntimes)
        innersize = map(x->2x+1, maxshift)
        @test size(dmmis) == (2, innersize..., gridsize..., ntimes)
    end
end

cs, Qs, mmis = jldopen(fn, mmaparrays=true) do file
    read(file, "cs"), read(file, "Qs"), read(file, "mmis")
end
ϕs, mismatch = fixed_λ(cs, Qs, knots, AffinePenalty(knots, 0.001), 1e-5, mmis)
for t = 1:nimages(img)
    moving = img[tax(t)]
    warped = warp(moving, ϕs[t])
    r_m = ratio(mismatch0(fixed, moving), 0)
    r_w = ratio(mismatch0(fixed, warped), 0)
    @test r_w < r_m
end

cs = Qs = mmis = 0   # since we're mmapping, we'd better free these before deleting files
GC.gc()

rm(workdir, recursive=true)
