using MatrixDiff
using Test

MD = MatrixDiff

@testset "evaluate Sym" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(3))
    z = Sym("z")

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "evaluate KrD simple" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(1))
    z = Sym("z")

    d1 = KrD(Lower(1), Upper(3))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.UnaryOperation(d1, A)) == Sym("A", Upper(3), Lower(2))
    @test evaluate(MD.UnaryOperation(d1, x)) == Sym("x", Upper(3))
    @test evaluate(MD.UnaryOperation(d1, z)) == MD.UnaryOperation(d1, z)
    @test evaluate(MD.UnaryOperation(d2, A)) == Sym("A", Upper(1), Lower(3))
end

@testset "evaluate transpose simple" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(1))
    z = Sym("z")

    # @test evaluate(A') == Sym("A", Upper(2), Lower(1)) # Not implemented
    @test evaluate(x') == Sym("x", Lower(1))
    # @test evaluate(z') == Sym("z") # Not implemented
end

@testset "evaluate UnaryOperation vector-KrD" begin
    x = Sym("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.UnaryOperation(d1, x)) == Sym("x", Upper(3))
    @test evaluate(MD.UnaryOperation(d2, x)) == Sym("x", Lower(2))
end

@testset "evaluate UnaryOperation matrix-KrD" begin
    A = Sym("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.UnaryOperation(d1, A)) == Sym("A", Upper(3), Lower(4))
    @test evaluate(MD.UnaryOperation(d2, A)) == Sym("A", Lower(2), Lower(4))
end

@testset "evaluate UnaryOperation KrD-KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.UnaryOperation(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.UnaryOperation(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation vector-KrD" begin
    x = Sym("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.BinaryOperation(*, d1, x)) == Sym("x", Upper(3))
    @test evaluate(MD.BinaryOperation(*, x, d1)) == Sym("x", Upper(3))
    @test evaluate(MD.BinaryOperation(*, d2, x)) == Sym("x", Lower(2))
    @test evaluate(MD.BinaryOperation(*, x, d2)) == Sym("x", Lower(2))
end

@testset "evaluate BinaryOperation matrix-KrD" begin
    A = Sym("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.BinaryOperation(*, d1, A)) == Sym("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation(*, A, d1)) == Sym("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation(*, d2, A)) == Sym("A", Lower(2), Lower(4))
    @test evaluate(MD.BinaryOperation(*, A, d2)) == Sym("A", Lower(2), Lower(4))
end

@testset "evaluate BinaryOperation KrD-KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.BinaryOperation(*, d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.BinaryOperation(*, d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation*" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(2))
    y = Sym("y", Upper(3))

    @test evaluate(MD.BinaryOperation(*, A, x)) == MD.BinaryOperation(*, A, x)
    @test evaluate(MD.BinaryOperation(*, A, y)) == MD.BinaryOperation(*, A, y)
end
