using MatrixDiff
using Test

MD = MatrixDiff

@testset "evaluate Sym" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    z = Sym("z")

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "evaluate BinaryOperation vector-KrD" begin
    x = Sym("x", Upper())
    d1 = KrD(Lower(), Upper())
    d2 = KrD(Lower(), Lower())

    @test evaluate(MD.BinaryOperation{*}(d1, x, [(1, 1)])) == Sym("x", Upper())
    @test evaluate(MD.BinaryOperation{*}(x, d1, [(1, 1)])) == Sym("x", Upper())
    @test evaluate(MD.BinaryOperation{*}(d2, x, [(1, 1)])) == Sym("x", Lower())
    @test evaluate(MD.BinaryOperation{*}(x, d2, [(1, 1)])) == Sym("x", Lower())
    @test evaluate(MD.BinaryOperation{*}(d2, x, [(2, 1)])) == Sym("x", Lower())
    @test evaluate(MD.BinaryOperation{*}(x, d2, [(1, 2)])) == Sym("x", Lower())

end

@testset "evaluate BinaryOperation matrix-KrD" begin
    A = Sym("A", Upper(), Lower())
    d1 = KrD(Upper(), Upper())
    d2 = KrD(Lower(), Lower())

    @test evaluate(MD.BinaryOperation{*}(d1, A, [(2, 2)])) == Sym("A", Upper(), Upper())
    @test evaluate(MD.BinaryOperation{*}(d2, A, [(1, 1)])) == Sym("A", Lower(), Lower())
end

@testset "evaluate BinaryOperation KrD-KrD" begin
    d1 = KrD(Upper(), Lower())
    d2 = KrD(Upper(), Lower())

    @test evaluate(MD.BinaryOperation{*}(d1, d2, [(2, 1)])) == KrD(Upper(), Lower())
    @test evaluate(MD.BinaryOperation{*}(d2, d1, [(2, 1)])) == KrD(Upper(), Lower())
end

@testset "evaluate BinaryOperation matrix-KrD" begin
    A = Sym("A", Upper(), Lower())
    d1 = KrD(Lower(), Upper())
    d2 = KrD(Lower(), Lower())

    @test evaluate(MD.BinaryOperation{*}(d1, A, [(1, 1)])) == Sym("A", Upper(), Lower())
    @test evaluate(MD.BinaryOperation{*}(A, d1, [(1, 1)])) == Sym("A", Upper(), Lower())
    @test evaluate(MD.BinaryOperation{*}(d2, A, [(1, 1)])) == Sym("A", Lower(), Lower())
    @test evaluate(MD.BinaryOperation{*}(d2, A, [(2, 1)])) == Sym("A", Lower(), Lower())
    @test evaluate(MD.BinaryOperation{*}(A, d2, [(1, 1)])) == Sym("A", Lower(), Lower())
    @test evaluate(MD.BinaryOperation{*}(A, d2, [(1, 2)])) == Sym("A", Lower(), Lower())
end

@testset "evaluate BinaryOperation KrD-KrD" begin
    d1 = KrD(Upper(), Lower())
    d2 = KrD(Upper(), Lower())

    @test evaluate(MD.BinaryOperation{*}(d1, d2, [(1, 2)])) == KrD(Upper(), Lower())
    @test evaluate(MD.BinaryOperation{*}(d2, d1, [(2, 1)])) == KrD(Upper(), Lower())
end

@testset "evaluate BinaryOperation matrix-vector" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    y = Sym("y", Lower())

    @test evaluate(MD.BinaryOperation{*}(A, x, [(2, 1)])) == MD.BinaryOperation{*}(A, x, [(2, 1)])
    @test evaluate(MD.BinaryOperation{*}(y, A, [(1, 1)])) == MD.BinaryOperation{*}(y, A, [(1, 1)])
end

@testset "evaluate transpose" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())
    z = Sym("z")

    @test evaluate(A') == Sym("A", Lower(), Upper())
    @test evaluate(x') == Sym("x", Lower())
    # @test evaluate(z') == Sym("z") # Not implemented
end

@testset "diff Sym" begin
    x = Sym("x", Upper())
    y = Sym("y", Upper())
    A = Sym("A", Upper(), Lower())

    @test MD.diff(x, x) == KrD(Upper(), Lower())
    @test MD.diff(y, x) == Zero(Upper(), Lower())
    @test MD.diff(A, x) == Zero(Upper(), Lower(), Lower())
end

@testset "diff KrD" begin
    x = Sym("x", Upper())
    y = Sym("y", Upper())
    A = Sym("A", Upper(), Lower())

    @test MD.diff(x, x) == KrD(Upper(), Lower())
    @test MD.diff(y, x) == Zero(Upper(), Lower())
    @test MD.diff(A, x) == Zero(Upper(), Lower(), Lower())
end

@testset "diff UnaryOperation" begin
    x = Sym("x", Upper())

    op = MD.UnaryOperation(KrD(Lower(), Lower()), x)

    @test MD.diff(op, x) == MD.UnaryOperation(KrD(Lower(), Lower()), KrD(Upper(), Lower()))
end

@testset "diff BinaryOperation{*}" begin
    x = Sym("x", Upper())
    y = Sym("y", Lower())

    op = MD.BinaryOperation{*}(x, y, [(1, 1)])

    D = MD.diff(op, x)

    @test typeof(D) == MD.BinaryOperation{+}
    @test D.arg1 == MD.BinaryOperation{*}(x, Zero(Lower(), Lower()), [(1, 1)])
    @test D.arg2 == MD.BinaryOperation{*}(KrD(Upper(), Lower()), y, [(1, 1)])
end

@testset "diff BinaryOperation{+}" begin
    x = Sym("x", Upper())
    y = Sym("y", Upper())

    op = MD.BinaryOperation{+}(x, y, [])

    D = MD.diff(op, x)

    @test D == MD.BinaryOperation{+}(KrD(Upper(), Lower()), Zero(Upper(), Lower()), [])
end

@testset "Differentiate Ax" begin
    A = Sym("A", Upper(), Lower())
    x = Sym("x", Upper())

    e = A * x

    @test D(e, x) == A
end
