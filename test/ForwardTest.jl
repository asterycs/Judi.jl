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

    @test evaluate(A') == Sym("A", Upper(2), Lower(1))
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

    @test evaluate(MD.BinaryOperation{*}(d1, x)) == Sym("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(x, d1)) == Sym("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(d2, x)) == Sym("x", Lower(2))
    @test evaluate(MD.BinaryOperation{*}(x, d2)) == Sym("x", Lower(2))
end

@testset "evaluate BinaryOperation matrix-KrD" begin
    A = Sym("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.BinaryOperation{*}(d1, A)) == Sym("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d1)) == Sym("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(d2, A)) == Sym("A", Lower(2), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d2)) == Sym("A", Lower(2), Lower(4))
end

@testset "evaluate BinaryOperation KrD-KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.BinaryOperation{*}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.BinaryOperation{*}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation*" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(2))
    y = Sym("y", Upper(3))

    @test evaluate(MD.BinaryOperation{*}(A, x)) == MD.BinaryOperation{*}(A, x)
    @test evaluate(MD.BinaryOperation{*}(A, y)) == MD.BinaryOperation{*}(A, y)
end

@testset "diff Sym" begin
    x = Sym("x", Upper(2))
    y = Sym("y", Upper(3))
    A = Sym("A", Upper(4), Lower(5))

    @test MD.diff(x, x) == Zero(Upper(2))
    @test MD.diff(y, x) == Zero(Upper(3), Lower(2))
    @test MD.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test MD.diff(x, Sym("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test MD.diff(y, Sym("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test MD.diff(A, Sym("A", Upper(6), Lower(7))) == MD.BinaryOperation{*}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Sym("x", Upper(3))
    y = Sym("y", Upper(4))
    A = Sym("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test MD.diff(d, x) == Zero(Upper(1), Lower(2), Lower(3))
    @test MD.diff(d, y) == Zero(Upper(1), Lower(2), Lower(4))
    @test MD.diff(d, A) == Zero(Upper(1), Lower(2), Lower(5), Upper(6))
end

@testset "diff UnaryOperation" begin
    x = Sym("x", Upper(2))

    op = MD.UnaryOperation(KrD(Lower(2), Lower(2)), x)

    @test MD.diff(op, Sym("x", Upper(3))) == MD.UnaryOperation(KrD(Lower(2), Lower(2)), KrD(Upper(2), Lower(3)))
end

@testset "diff BinaryOperation{*}" begin
    x = Sym("x", Upper(2))
    y = Sym("y", Lower(2))

    op = MD.BinaryOperation{*}(x, y)

    D = MD.diff(op, Sym("x", Upper(3)))

    @test typeof(D) == MD.BinaryOperation{+}
    @test D.arg1 == MD.BinaryOperation{*}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == MD.BinaryOperation{*}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{+}" begin
    x = Sym("x", Upper(2))
    y = Sym("y", Upper(2))

    op = MD.BinaryOperation{+}(x, y)

    D = MD.diff(op, Sym("x", Upper(3)))

    @test D == MD.BinaryOperation{+}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
end

@testset "Differentiate Ax" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(3))

    e = A * x

    @test equivalent(D(e, Sym("x", Upper(4))), A)
end

@testset "Differentiate xᵀA " begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(3))

    e = x' * A

    @test equivalent(D(e, Sym("x", Upper(4))), Sym("A", Lower(1), Lower(2)))
end

@testset "Differentiate xᵀAx" begin
    A = Sym("A", Upper(1), Lower(2))
    x = Sym("x", Upper(3))

    e = x' * A * x

    differential_form = MD.D(e, Sym("x", Upper(4)))

    @test equivalent(differential_form.arg1, x' * A)
    @test equivalent(differential_form.arg2, MD.BinaryOperation{*}(x', Sym("A", Lower(1), Lower(2))))
end
