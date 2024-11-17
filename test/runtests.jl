using Test

include("Aqua.jl")

@testset "RicciTest" begin
    include("RicciTest.jl")
end

@testset "ForwardTest" begin
    include("ForwardTest.jl")
end
