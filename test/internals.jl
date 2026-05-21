@testset "cudatype" begin
    @test RegisterWorkerAperturesMismatch.cudatype(Float32) == Float32
    @test RegisterWorkerAperturesMismatch.cudatype(Float64) == Float64
    @test RegisterWorkerAperturesMismatch.cudatype(Int32) == Float32
    @test RegisterWorkerAperturesMismatch.cudatype(UInt8) == Float32
end
