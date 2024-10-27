using MatrixCalculus
using Test

@testset "End-to-end simple" begin
    x = Sym("x", [Upper(3)])
    A = Sym("A", [Upper(1); Lower(2)])

    graph = record(A * x)
    p = assemble_pullback(graph)

    @test to_string.(simplify.(p(Sym("I", [])))) == "A"


    graph = record(x' * A)
    p = assemble_pullback(graph)

    @test to_string.(simplify.(p(Sym("I", [])))) == "A"
end

@testset "Invalid product" begin
    x = Sym("x", [Upper(3)])
    A = Sym("A", [Upper(1); Lower(2)])

    @test_throws DomainError x * A
end
