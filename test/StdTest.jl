using MatrixDiff
using Test

MD = MatrixDiff

@testset "create column vector" begin
    x = create_vector("x")
    y = create_vector("y")
    A = create_vector("A")

    @test x == Sym("x", Upper(1))
    @test y == Sym("y", Upper(2))
    @test A == Sym("A", Upper(3))
end

@testset "create matrix" begin
    A = create_matrix("A")
    B = create_matrix("B")
    X = create_matrix("X")

    @test A == Sym("A", Upper(4), Lower(5))
    @test B == Sym("B", Upper(6), Lower(7))
    @test X == Sym("X", Upper(8), Lower(9))
end