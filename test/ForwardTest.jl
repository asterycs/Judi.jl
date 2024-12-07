using MatrixDiff
using Test

MD = MatrixDiff

@testset "evaluate Tensor" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))
    z = Tensor("z")

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "evaluate KrD simple" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    z = Tensor("z")

    d1 = KrD(Lower(1), Upper(3))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.BinaryOperation{*}(A, d1)) == Tensor("A", Upper(3), Lower(2))
    @test evaluate(MD.BinaryOperation{*}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(z, d1)) == MD.BinaryOperation{*}(d1, z)
    @test evaluate(MD.BinaryOperation{*}(A, d2)) == Tensor("A", Upper(1), Lower(3))
end

@testset "evaluate transpose simple" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    z = Tensor("z")

    @test equivalent(evaluate(A'), Tensor("A", Lower(1), Upper(2)))
    @test equivalent(evaluate(x'), Tensor("x", Lower(1)))
    @test evaluate(z') == Tensor("z")
end

@testset "evaluate BinaryOperation{+-} Matrix and KrD" begin
    X = Tensor("X", Upper(2), Lower(3))
    d = KrD(Upper(4), Lower(5))

    for op ∈ (+, -)
        op1 = MD.BinaryOperation{op}(d, X)
        op2 = MD.BinaryOperation{op}(X, d)
        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate Matrix + Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = MD.BinaryOperation{+}(Z, X)
    op2 = MD.BinaryOperation{+}(X, Z)
    @test evaluate(op1) == X
    @test evaluate(op2) == X
end

@testset "evaluate Matrix - Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = MD.BinaryOperation{-}(Z, X)
    op2 = MD.BinaryOperation{-}(X, Z)
    @test evaluate(op1) == -X
    @test evaluate(op2) == X
end

@testset "evaluate BinaryOperation vector * KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Upper(2))

    @test evaluate(MD.BinaryOperation{*}(d1, x)) == Tensor("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(d2, x)) == Tensor("x", Upper(2))
    @test evaluate(MD.BinaryOperation{*}(x, d2)) == Tensor("x", Upper(2))
end

@testset "evaluate BinaryOperation matrix * KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(3))
    d3 = KrD(Upper(4), Lower(1))

    @test evaluate(MD.BinaryOperation{*}(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d1)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(d2, A)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d2)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(d3, A)) == Tensor("A", Upper(2), Lower(1))
    @test evaluate(MD.BinaryOperation{*}(A, d3)) == Tensor("A", Upper(2), Lower(1))
end

@testset "evaluate BinaryOperation matrix * Zero" begin
    A = Tensor("A", Upper(2), Lower(4))
    Z = Zero(Upper(4), Lower(3), Lower(5))

    @test evaluate(MD.BinaryOperation{*}(Z, A)) == Zero(Lower(3), Lower(5), Upper(2))
    @test evaluate(MD.BinaryOperation{*}(A, Z)) == Zero(Upper(2), Lower(3), Lower(5))
end

@testset "evaluate BinaryOperation KrD * KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.BinaryOperation{*}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.BinaryOperation{*}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation with outer product" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    @test evaluate(MD.BinaryOperation{*}(A, x)) == MD.BinaryOperation{*}(A, x)
    @test evaluate(MD.BinaryOperation{*}(A, y)) == MD.BinaryOperation{*}(A, y)
end

@testset "evaluate subtraction with * and +" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = A * x - (x + y)
    op2 = (x + y) - A * x

    @test evaluate(op1) == op1
    @test evaluate(op2) == - (A * x) + (x + y)
end

@testset "evaluate subtraction with * and + and simplify" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = (x + y) - (x + y)
    op2 = 2 * x - 2 * x

    @test equivalent(evaluate(op1), Zero(Upper(2)))
    @test equivalent(evaluate(op2), Zero(Upper(2)))
end

@testset "evaluate unary operations" begin
    A = Tensor("A", Upper(1), Lower(2))

    ops = (sin, cos)
    types = (MD.Sin, MD.Cos)

    for (op, type) ∈ zip(ops, types)
        @test typeof(op(A)) == type
        @test op(A).arg == A
    end
end

@testset "evaluate trace" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    # @test evaluate(tr(A)) == Tensor("A", Upper(1), Lower(1)) # TODO: Triggers assertion
    @test evaluate(tr(A * B)) == MD.BinaryOperation{*}(A, Tensor("B", Upper(2), Lower(1)))
end

@testset "evaluate outer product - contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(3), Lower(4))
    x = Tensor("x", Lower(3))

    @test evaluate(*, MD.BinaryOperation{*}(A, d), x) ==
          MD.BinaryOperation{*}(A, Tensor("x", Lower(4)))
    @test evaluate(*, MD.BinaryOperation{*}(d, A), x) ==
          MD.BinaryOperation{*}(A, Tensor("x", Lower(4)))
    @test evaluate(*, x, MD.BinaryOperation{*}(A, d)) ==
          MD.BinaryOperation{*}(Tensor("x", Lower(4)), A)
    @test evaluate(*, x, MD.BinaryOperation{*}(d, A)) ==
          MD.BinaryOperation{*}(Tensor("x", Lower(4)), A)

    @test evaluate(*, MD.BinaryOperation{*}(d, A), x) ==
          MD.BinaryOperation{*}(Tensor("x", Lower(4)), A)
    @test evaluate(*, MD.BinaryOperation{*}(A, d), x) ==
          MD.BinaryOperation{*}(Tensor("x", Lower(4)), A)
    @test evaluate(*, x, MD.BinaryOperation{*}(d, A)) ==
          MD.BinaryOperation{*}(A, Tensor("x", Lower(4)))
    @test evaluate(*, x, MD.BinaryOperation{*}(A, d)) ==
          MD.BinaryOperation{*}(A, Tensor("x", Lower(4)))
end

@testset "is_trace output is correct" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(2), Lower(1))
    d2 = KrD(Upper(2), Lower(3))

    @test MD.is_trace(A, d)
    @test !MD.is_trace(A, d2)
end

@testset "diff Tensor" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    A = Tensor("A", Upper(4), Lower(5))

    @test MD.diff(x, x) == KrD(Upper(2), Lower(2))
    @test MD.diff(y, x) == Zero(Upper(3), Lower(2))
    @test MD.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test MD.diff(x, Tensor("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test MD.diff(y, Tensor("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test MD.diff(A, Tensor("A", Upper(6), Lower(7))) ==
          MD.BinaryOperation{*}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Tensor("x", Upper(3))
    y = Tensor("y", Lower(4))
    A = Tensor("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test MD.diff(d, x) == MD.BinaryOperation{*}(d, Zero(Lower(3)))
    @test MD.diff(d, y) == MD.BinaryOperation{*}(d, Zero(Upper(4)))
    @test MD.diff(d, A) == MD.BinaryOperation{*}(d, Zero(Lower(5), Upper(6)))
end

@testset "diff BinaryOperation{*}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(2))

    op = MD.BinaryOperation{*}(x, y)

    D = MD.diff(op, Tensor("x", Upper(3)))

    @test typeof(D) == MD.BinaryOperation{+}
    @test D.arg1 == MD.BinaryOperation{*}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == MD.BinaryOperation{*}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{+-}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))

    for op ∈ (+, -)
        v = MD.BinaryOperation{op}(x, y)

        D = MD.diff(v, Tensor("x", Upper(3)))

        @test D == MD.BinaryOperation{op}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
    end
end

@testset "diff trace" begin
    A = Tensor("A", Upper(1), Lower(2))

    op = tr(A)

    D = MD.diff(op, Tensor("A", Upper(3), Lower(4)))

    @test equivalent(evaluate(D), KrD(Upper(1), Lower(2)))
end

@testset "diff sin" begin
    x = Tensor("x", Upper(2))

    op = sin(x)

    D = MD.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, MD.BinaryOperation{*}(MD.Cos(x), KrD(Upper(2), Lower(3))))
end

@testset "diff cos" begin
    x = Tensor("x", Upper(2))

    op = cos(x)

    D = MD.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, MD.BinaryOperation{*}(MD.Negate(MD.Sin(x)), KrD(Upper(2), Lower(3))))
end

@testset "diff negated vector" begin
    x = Tensor("x", Upper(2))

    op = -x

    D = MD.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, MD.Negate(KrD(Upper(2), Lower(3))))
end

@testset "free indices constant after evaluate" begin
    x = Tensor("x", Upper(2))
    c = Tensor("c", Upper(3))
    y = Tensor("y", Upper(4))

    op1 = (y .* c)' * x

    @test isempty(MD.get_free_indices(op1))
    @test isempty(MD.get_free_indices(evaluate(op1)))

    op2 = tr(x * x')

    @test isempty(MD.get_free_indices(op2))
    @test isempty(MD.get_free_indices(evaluate(op2)))
end

@testset "Differentiate Ax" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = A * x

    @test equivalent(derivative(e, "x"), A)
end

@testset "Differentiate xᵀA " begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A

    @test equivalent(derivative(e, "x"), Tensor("A", Lower(1), Lower(2)))
end

@testset "Differentiate xᵀAx" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    D = derivative(e, "x")

    @test equivalent(D.arg1, evaluate(x' * A))
    @test equivalent(
        D.arg2,
        evaluate(MD.BinaryOperation{*}(Tensor("A", Lower(1), Lower(3)), x)),
    )
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    D = derivative(A * (x + 2 * x), "x")

    @test equivalent(D, 3 * A)
end

@testset "Differentiate A(2x + x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    D = derivative(A * (x + 2 * x), "x")

    @test equivalent(D, 3 * A)
end
