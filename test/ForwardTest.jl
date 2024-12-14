using Yodi
using Test

yd = Yodi

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

    @test evaluate(yd.BinaryOperation{yd.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(2))
    @test evaluate(yd.BinaryOperation{yd.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(yd.BinaryOperation{yd.Mult}(z, d1)) == yd.BinaryOperation{yd.Mult}(d1, z)
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, d2)) == Tensor("A", Upper(1), Lower(3))
end

@testset "evaluate transpose simple" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    z = Tensor("z")

    @test equivalent(evaluate(A'), Tensor("A", Lower(1), Upper(2)))
    @test equivalent(evaluate(x'), Tensor("x", Lower(1)))
    @test evaluate(z') == Tensor("z")
end

@testset "evaluate BinaryOperation{AdditiveOperation} Matrix and KrD" begin
    X = Tensor("X", Upper(2), Lower(3))
    d = KrD(Upper(2), Lower(3))

    for op ∈ (yd.Add, yd.Sub)
        op1 = yd.BinaryOperation{op}(d, X)
        op2 = yd.BinaryOperation{op}(X, d)
        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate Matrix + Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = yd.BinaryOperation{yd.Add}(Z, X)
    op2 = yd.BinaryOperation{yd.Add}(X, Z)
    @test evaluate(op1) == X
    @test evaluate(op2) == X
end

@testset "evaluate Matrix - Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = yd.BinaryOperation{yd.Sub}(Z, X)
    op2 = yd.BinaryOperation{yd.Sub}(X, Z)
    @test evaluate(op1) == -X
    @test evaluate(op2) == X
end

@testset "evaluate BinaryOperation vector * KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Upper(3), Lower(2))

    @test evaluate(yd.BinaryOperation{yd.Mult}(d1, x)) == Tensor("x", Upper(3))
    @test evaluate(yd.BinaryOperation{yd.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(yd.BinaryOperation{yd.Mult}(d2, x)) == Tensor("x", Upper(3))
    @test evaluate(yd.BinaryOperation{yd.Mult}(x, d2)) == Tensor("x", Upper(3))
end

# TODO: Triggers an assertion for now
# @testset "evaluate BinaryOperation vector * KrD with ambiguous indices fails" begin
#     x = Tensor("x", Upper(2))
#     d = KrD(Upper(2), Lower(2))

#     @test_throws DomainError evaluate(yd.BinaryOperation{yd.Mult}(d, x))
#     @test_throws DomainError evaluate(yd.BinaryOperation{yd.Mult}(x, d))
# end

@testset "evaluate BinaryOperation matrix * KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(3))
    d3 = KrD(Upper(4), Lower(1))

    @test evaluate(yd.BinaryOperation{yd.Mult}(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(yd.BinaryOperation{yd.Mult}(d2, A)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, d2)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(yd.BinaryOperation{yd.Mult}(d3, A)) == Tensor("A", Upper(2), Lower(1))
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, d3)) == Tensor("A", Upper(2), Lower(1))
end

@testset "evaluate BinaryOperation matrix * Zero" begin
    A = Tensor("A", Upper(2), Lower(4))
    Z = Zero(Upper(4), Lower(3), Lower(5))

    @test evaluate(yd.BinaryOperation{yd.Mult}(Z, A)) == Zero(Upper(2), Lower(3), Lower(5))
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, Z)) == Zero(Upper(2), Lower(3), Lower(5))
end

@testset "evaluate BinaryOperation KrD * KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(yd.BinaryOperation{yd.Mult}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(yd.BinaryOperation{yd.Mult}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation with outer product" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    @test evaluate(yd.BinaryOperation{yd.Mult}(A, x)) == yd.BinaryOperation{yd.Mult}(A, x)
    @test evaluate(yd.BinaryOperation{yd.Mult}(A, y)) == yd.BinaryOperation{yd.Mult}(A, y)
end

@testset "evaluate yd.Subtraction with * and +" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = A * x - (x + y)
    op2 = (x + y) - A * x

    @test length(yd.get_free_indices(evaluate(op1))) == 1
    @test length(yd.get_free_indices(evaluate(op2))) == 1
end

@testset "evaluate yd.Subtraction with * and + and simplify" begin
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
    types = (yd.Sin, yd.Cos)

    for (op, type) ∈ zip(ops, types)
        @test typeof(op(A)) == type
        @test op(A).arg == A
    end
end

@testset "evaluate trace" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    # @test evaluate(tr(A)) == Tensor("A", Upper(1), Lower(1)) # TODO: Triggers assertion
    @test equivalent(evaluate(tr(A * B)), yd.BinaryOperation{yd.Mult}(A, Tensor("B", Upper(2), Lower(1))))
end

@testset "evaluate outer product - contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(3), Lower(4))
    x = Tensor("x", Lower(3))

    @test evaluate(yd.Mult(), yd.BinaryOperation{yd.Mult}(A, d), x) ==
          yd.BinaryOperation{yd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(yd.Mult(), yd.BinaryOperation{yd.Mult}(d, A), x) ==
          yd.BinaryOperation{yd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(yd.Mult(), x, yd.BinaryOperation{yd.Mult}(A, d)) ==
          yd.BinaryOperation{yd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(yd.Mult(), x, yd.BinaryOperation{yd.Mult}(d, A)) ==
          yd.BinaryOperation{yd.Mult}(Tensor("x", Lower(4)), A)

    @test evaluate(yd.Mult(), yd.BinaryOperation{yd.Mult}(d, A), x) ==
          yd.BinaryOperation{yd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(yd.Mult(), yd.BinaryOperation{yd.Mult}(A, d), x) ==
          yd.BinaryOperation{yd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(yd.Mult(), x, yd.BinaryOperation{yd.Mult}(d, A)) ==
          yd.BinaryOperation{yd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(yd.Mult(), x, yd.BinaryOperation{yd.Mult}(A, d)) ==
          yd.BinaryOperation{yd.Mult}(A, Tensor("x", Lower(4)))
end

@testset "is_trace output is correct" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(2), Lower(1))
    d2 = KrD(Upper(2), Lower(3))

    @test yd.is_trace(A, d)
    @test !yd.is_trace(A, d2)
end

@testset "diff Tensor" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    A = Tensor("A", Upper(4), Lower(5))

    @test yd.diff(x, x) == KrD(Upper(2), Lower(2))
    @test yd.diff(y, x) == Zero(Upper(3), Lower(2))
    @test yd.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test yd.diff(x, Tensor("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test yd.diff(y, Tensor("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test yd.diff(A, Tensor("A", Upper(6), Lower(7))) ==
          yd.BinaryOperation{yd.Mult}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Tensor("x", Upper(3))
    y = Tensor("y", Lower(4))
    A = Tensor("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test yd.diff(d, x) == yd.BinaryOperation{yd.Mult}(d, Zero(Lower(3)))
    @test yd.diff(d, y) == yd.BinaryOperation{yd.Mult}(d, Zero(Upper(4)))
    @test yd.diff(d, A) == yd.BinaryOperation{yd.Mult}(d, Zero(Lower(5), Upper(6)))
end

@testset "diff BinaryOperation{yd.Mult}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(2))

    op = yd.BinaryOperation{yd.Mult}(x, y)

    D = yd.diff(op, Tensor("x", Upper(3)))

    @test typeof(D) == yd.BinaryOperation{yd.Add}
    @test D.arg1 == yd.BinaryOperation{yd.Mult}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == yd.BinaryOperation{yd.Mult}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{AdditiveOperation}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))

    for op ∈ (yd.Add, yd.Sub)
        v = yd.BinaryOperation{op}(x, y)

        D = yd.diff(v, Tensor("x", Upper(3)))

        @test D == yd.BinaryOperation{op}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
    end
end

@testset "diff trace" begin
    A = Tensor("A", Upper(1), Lower(2))

    op = tr(A)

    D = yd.diff(op, Tensor("A", Upper(3), Lower(4)))

    @test equivalent(evaluate(D), KrD(Upper(1), Lower(2)))
end

@testset "diff sin" begin
    x = Tensor("x", Upper(2))

    op = sin(x)

    D = yd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, yd.BinaryOperation{yd.Mult}(yd.Cos(x), KrD(Upper(2), Lower(3))))
end

@testset "diff cos" begin
    x = Tensor("x", Upper(2))

    op = cos(x)

    D = yd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, yd.BinaryOperation{yd.Mult}(yd.Negate(yd.Sin(x)), KrD(Upper(2), Lower(3))))
end

@testset "diff negated vector" begin
    x = Tensor("x", Upper(2))

    op = -x

    D = yd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, yd.Negate(KrD(Upper(2), Lower(3))))
end

@testset "free indices constant after evaluate" begin
    x = Tensor("x", Upper(2))
    c = Tensor("c", Upper(3))
    y = Tensor("y", Upper(4))

    op1 = (y .* c)' * x

    @test isempty(yd.get_free_indices(op1))
    @test isempty(yd.get_free_indices(evaluate(op1)))

    op2 = tr(x * x')

    @test isempty(yd.get_free_indices(op2))
    @test isempty(yd.get_free_indices(evaluate(op2)))
end

@testset "KrD collapsed correctly on element wise yd.Multiplications" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    e = (y .* z)' * x

    expected = yd.BinaryOperation{yd.Mult}(Tensor("y", Lower(1)), Tensor("x", Lower(1)))

    @test equivalent(evaluate(yd.diff(e, Tensor("z", Upper(9)))), expected)

    expected = yd.BinaryOperation{yd.Mult}(Tensor("y", Upper(1)), Tensor("x", Upper(1)))
    @test equivalent(evaluate(yd.diff(e, Tensor("z", Upper(9)))'), expected)
    @test equivalent(evaluate(yd.diff(e', Tensor("z", Upper(9)))), expected)
    @test equivalent(evaluate(evaluate(yd.diff(e, Tensor("z", Upper(9))))'), expected)
end

@testset "KrD collapsed correctly on element wise yd.Multiplications (mirrored)" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    e = x' * (y .* z)

    expected = yd.BinaryOperation{yd.Mult}(Tensor("y", Lower(1)), Tensor("x", Lower(1)))

    @test equivalent(evaluate(yd.diff(e, Tensor("z", Upper(9)))), expected)

    expected = yd.BinaryOperation{yd.Mult}(Tensor("y", Upper(1)), Tensor("x", Upper(1)))
    @test equivalent(evaluate(yd.diff(e, Tensor("z", Upper(9)))'), expected)
    @test equivalent(evaluate(yd.diff(e', Tensor("z", Upper(9)))), expected)
    @test equivalent(evaluate(evaluate(yd.diff(e, Tensor("z", Upper(9))))'), expected)
end

@testset "Differentiate Ax" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(yd.diff(A * x, Tensor("x", Upper(4))), A)
end

@testset "Differentiate xᵀA " begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(yd.diff(x' * A, Tensor("x", Upper(4))), Tensor("A", Lower(1), Lower(2)))
end

@testset "Differentiate xᵀAx" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = yd.diff(x' * A * x, Tensor("x", Upper(4)))

    @test equivalent(evaluate(D.arg1), evaluate(x' * A))
    @test equivalent(
        evaluate(D.arg2),
        evaluate(yd.BinaryOperation{yd.Mult}(Tensor("A", Lower(1), Lower(3)), x)),
    )
end

@testset "Differentiate xx'x" begin
    x = Tensor("x", Upper(1))

    l = yd.BinaryOperation{yd.Mult}(Tensor("x", Upper(100)), Tensor("x", Lower(2)))
    rl = yd.BinaryOperation{yd.Mult}(Tensor("x", Upper(100)), Tensor("x", Lower(2)))
    rr = yd.BinaryOperation{yd.Mult}(KrD(Upper(100), Lower(2)), x'*x)
    expected = yd.BinaryOperation{yd.Add}(l, yd.BinaryOperation{yd.Add}(rl, rr))

    D = yd.diff(x * x' * x, Tensor("x", Upper(2)))

    @test equivalent(evaluate(D), expected)
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = yd.diff(A * (x + 2 * x), Tensor("x", Upper(3)))

    expected = yd.BinaryOperation{yd.Mult}(3, KrD(Upper(3), Lower(3)))
    expected = yd.BinaryOperation{yd.Mult}(expected, Tensor("A", Upper(1), Lower(3)))

    @test equivalent(evaluate(D), expected)
end

@testset "Differentiate A(2x + x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = yd.diff(A * (x + 2 * x), Tensor("x", Upper(4)))

    @test equivalent(evaluate(D), 3 * A)
end

@testset "evaluated derivative is equal to derivative of evaluated expression" begin
    prefix =
    """
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))
    y = Tensor("y", Upper(4))
    c = Tensor("c", Upper(5))
    """

    expr_strs = ( #
        "A * x", #
        "x' * A", #
        "x' * A * x", #
        "A * (x + 2 * x)", #
        "A * (2 * x + x)", #
        "x' * x", #
        "tr(x * x')", #
        "(y .* c)' * x", #
        "(x .* c)' * x", #
        "(x + y)' * x", #
        "(x - y)' * x", #
        "x' * (y .* c)", #
        "x' * (x .* c)", #
        "x' * (x + y)", #
        "x' * (x - y)", #
        "sin(tr(x * x'))", #
        "cos(tr(x * x'))", #
        "tr(sin(x * x'))", #
        "tr(A)", #
    )

    # TODO: There's probably a better way to do this.
    @eval $(Meta.parseall(prefix))

    for str ∈ expr_strs
        expr = @eval $(Meta.parse(str))

        @testset "$str" begin
            @test evaluate(yd.diff(evaluate(expr), Tensor("A", Upper(10), Lower(11)))) == evaluate(yd.diff(expr, Tensor("A", Upper(10), Lower(11))))
            @test evaluate(yd.diff(evaluate(expr), Tensor("x", Upper(10)))) == evaluate(yd.diff(expr, Tensor("x", Upper(10))))
            @test evaluate(yd.diff(evaluate(expr), Tensor("y", Upper(10)))) == evaluate(yd.diff(expr, Tensor("y", Upper(10))))
            @test evaluate(yd.diff(evaluate(expr), Tensor("c", Upper(10)))) == evaluate(yd.diff(expr, Tensor("c", Upper(10))))
        end
    end
end
