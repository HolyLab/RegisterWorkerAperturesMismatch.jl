for (sz, mxshift, gridsize, order, steps) in (((150,), (6,), (4,), (:x, :time),(1,1)),
                                    ((150,140), (6,5), (4,3), (:x, :y, :time),(1,1,1)),
                                    ((150,140,15), (6,5,2), (4,3,3), (:x, :y, :z, :time), (1,1,1,1)))
    N = length(sz)
    knots = ntuple(d->range(1, stop=sz[d], length=gridsize[d]), N)
    # Create a fixed image with a uniform background except for some regions
    # around the center of each aperture.
    # [just one pixel per aperture is not enough to find the best mm]
    # We create as a larger image to test that having "padding"
    # with valid values gets exploited by the mismatch computation.
    padsz = [mxshift...]+ceil.(Int, [sz...]/2)  # excessive, but who cares?
    fullsz = (([sz...]+2padsz...,)...,)
    fixed_full = SharedArray{Float64}(fullsz)
    fill!(fixed_full, 1)
    window = ntuple(d->padsz[d]+1:padsz[d]+sz[d], N)
    fixed = view(fixed_full, ntuple(d->padsz[d]+1:padsz[d]+sz[d], N)...)
    for I in CartesianIndices(gridsize)
        c = [round(Int, knots[d][I[d]]) for d = 1:N]
        fixed[c...] = 2
    end

    # The moving image displaces the bright regions.
    # Make sure the shift won't move bright pixel into a different aperture
#    @assert all([sz...]./[gridsize...] .> 4*[mxshift...])
    movingsz = (fullsz..., nt)
    moving_full = SharedArray{Float64}(movingsz)
    fill!(moving_full, 1)
    window = (ntuple(d->padsz[d]+1:padsz[d]+sz[d], N)..., :)
    offset = ntuple(d->first(to_indices(moving_full, window)[d])-1,N)
    pdrng = shiftrange.(axes(moving_full),(.-offset...,0))
    moving = PaddedView(1, moving_full, pdrng, pdrng)
    displacements = Array{NTuple{N,Array{Int,N}}}(undef, nt)
    for t = 1:nt
        disp = ntuple(d->rand(-mxshift[d]+1:mxshift[d]-1, gridsize), N)
        displacements[t] = disp
        for I in CartesianIndices(gridsize)
            c = [round(Int, knots[d][I[d]])+disp[d][I]+offset[d] for d = 1:N]
            moving_full[c..., t] = 2
        end
    end
    img = AxisArray(moving, order, steps);

    ### Compute the mismatch
    baseout = tempname()
    fnmm = string(baseout, ".mm")
    algorithm = AperturesMismatch[AperturesMismatch(fixed, knots, mxshift; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
    mm_package_loader(algorithm)
    mon = monitor(algorithm, (:Es, :cs, :Qs, :mmis))

    driver(fnmm, algorithm, img, mon)

    # Append important extra information to the file
    jldopen(fnmm, "r+") do io
        write(io, "knots", knots)
    end

    ### Read the mismatch
    Es, cs, Qs, knots, mmis = jldopen(fnmm, mmaparrays=true) do file
        read(file, "Es"), read(file, "cs"), read(file, "Qs"), read(file, "knots"), read(file, "mmis")
    end;

    # Test that all pixels within an aperture were valid
    den = mmis[2,ntuple(d->Colon(), 2N+1)...]
    for i in CartesianIndices(size(den)[N+1:end])
        den1 = den[ntuple(d->Colon(), N)..., i]
        @test (1+1e-8)*minimum(den1) > maximum(den1)
    end

    # Test that a perfect match was found in each aperture
    @test all(Es .< 1e-12) # if this fails, make sure checkbias=false above
    # Test that the displacement is correct
    for t = 1:nt
        for d = 1:N
            css = cs[d,ntuple(d->Colon(), N)...,t]
            @test css == displacements[t][d]
        end
    end

    # Test initialization
    csr = reshape(reinterpret(SVector{N,Float64},vec(cs)), Base.tail(size(cs)))
    Qsr = reshape(reinterpret(SMatrix{N,N,Float64,N*N}, vec(Qs)), Base.tail(Base.tail(size(Qs))))
    ap = AffinePenalty(knots, 0.0)
    ur, _ = initial_deformation(ap, csr, Qsr)
    u = reshape(reinterpret(Float64, vec(ur)), (N, size(ur)...))
    @test maximum(abs.(u-cs)) <= 1e-3
end
