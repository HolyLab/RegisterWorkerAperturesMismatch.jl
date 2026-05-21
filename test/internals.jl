@testset "cudatype" begin
    @test RegisterWorkerAperturesMismatch.cudatype(Float32) == Float32
    @test RegisterWorkerAperturesMismatch.cudatype(Float64) == Float64
    @test RegisterWorkerAperturesMismatch.cudatype(Int32) == Float32
    @test RegisterWorkerAperturesMismatch.cudatype(UInt8) == Float32
end

@testset "AperturesMismatch constructor argument validation" begin
    fixed = rand(Float32, 20, 20)
    nodes = (range(1, stop=20, length=4), range(1, stop=20, length=4))
    # Vector maxshift should not match the NTuple{N,<:Integer} signature
    @test_throws MethodError AperturesMismatch(fixed, nodes, [3, 3])
end
