using Test

@testset "Aqua" begin
    include("Aqua.jl")
end

@testset "RicciTest" begin
    include("RicciTest.jl")
end

@testset "ForwardTest" begin
    include("ForwardTest.jl")
end

@testset "StdTest" begin
    include("StdTest.jl")
end
