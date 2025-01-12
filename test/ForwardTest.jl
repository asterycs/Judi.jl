# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using Judi
using Test

jd = Judi

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

    @test evaluate(jd.BinaryOperation{jd.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(2))
    @test evaluate(jd.BinaryOperation{jd.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(jd.BinaryOperation{jd.Mult}(z, d1)) == jd.BinaryOperation{jd.Mult}(d1, z)
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, d2)) == Tensor("A", Upper(1), Lower(3))
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

    for op ∈ (jd.Add, jd.Sub)
        op1 = jd.BinaryOperation{op}(d, X)
        op2 = jd.BinaryOperation{op}(X, d)
        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate Matrix + Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = jd.BinaryOperation{jd.Add}(Z, X)
    op2 = jd.BinaryOperation{jd.Add}(X, Z)
    @test evaluate(op1) == X
    @test evaluate(op2) == X
end

@testset "evaluate Matrix - Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = jd.BinaryOperation{jd.Sub}(Z, X)
    op2 = jd.BinaryOperation{jd.Sub}(X, Z)
    @test evaluate(op1) == -X
    @test evaluate(op2) == X
end

@testset "evaluate adjoint is consistent" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))
    x = Tensor("x", Upper(5))
    y = Tensor("y", Upper(6))

    @test equivalent(evaluate(x' * A'), evaluate((A * x)'))
    @test equivalent(evaluate(x' * A), evaluate((A' * x)'))
    @test equivalent(evaluate(x' * A * x), evaluate((A' * x)' * x))
end

@testset "evaluate BinaryOperation vector * KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Upper(3), Lower(2))

    @test evaluate(jd.BinaryOperation{jd.Mult}(d1, x)) == Tensor("x", Upper(3))
    @test evaluate(jd.BinaryOperation{jd.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(jd.BinaryOperation{jd.Mult}(d2, x)) == Tensor("x", Upper(3))
    @test evaluate(jd.BinaryOperation{jd.Mult}(x, d2)) == Tensor("x", Upper(3))
end

# TODO: Triggers an assertion for now
# @testset "evaluate BinaryOperation vector * KrD with ambiguous indices fails" begin
#     x = Tensor("x", Upper(2))
#     d = KrD(Upper(2), Lower(2))

#     @test_throws DomainError evaluate(jd.BinaryOperation{jd.Mult}(d, x))
#     @test_throws DomainError evaluate(jd.BinaryOperation{jd.Mult}(x, d))
# end

@testset "evaluate BinaryOperation matrix * KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(3))
    d3 = KrD(Upper(4), Lower(1))

    @test evaluate(jd.BinaryOperation{jd.Mult}(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(jd.BinaryOperation{jd.Mult}(d2, A)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, d2)) == Tensor("A", Lower(3), Lower(4))
    @test evaluate(jd.BinaryOperation{jd.Mult}(d3, A)) == Tensor("A", Upper(2), Lower(1))
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, d3)) == Tensor("A", Upper(2), Lower(1))
end

@testset "evaluate BinaryOperation matrix * Zero" begin
    A = Tensor("A", Upper(2), Lower(4))
    Z = Zero(Upper(4), Lower(3), Lower(5))

    @test evaluate(jd.BinaryOperation{jd.Mult}(Z, A)) == Zero(Upper(2), Lower(3), Lower(5))
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, Z)) == Zero(Upper(2), Lower(3), Lower(5))
end

@testset "evaluate BinaryOperation KrD * KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(jd.BinaryOperation{jd.Mult}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(jd.BinaryOperation{jd.Mult}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation with outer product" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    @test evaluate(jd.BinaryOperation{jd.Mult}(A, x)) == jd.BinaryOperation{jd.Mult}(A, x)
    @test evaluate(jd.BinaryOperation{jd.Mult}(A, y)) == jd.BinaryOperation{jd.Mult}(A, y)
end

@testset "evaluate subtraction with * and +" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = A * x - (x + y)
    op2 = (x + y) - A * x

    @test length(jd.get_free_indices(evaluate(op1))) == 1
    @test length(jd.get_free_indices(evaluate(op2))) == 1
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
    types = (jd.Sin, jd.Cos)

    for (op, type) ∈ zip(ops, types)
        @test typeof(op(A)) == type
        @test op(A).arg == A
    end
end

@testset "evaluate trace" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test evaluate(tr(A)) == Tensor("A", Upper(2), Lower(2))
    @test equivalent(evaluate(tr(A * B)), jd.BinaryOperation{jd.Mult}(A, Tensor("B", Upper(2), Lower(1))))
end

@testset "evaluate outer product - contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(3), Lower(4))
    x = Tensor("x", Lower(3))

    @test evaluate(jd.Mult(), jd.BinaryOperation{jd.Mult}(A, d), x) ==
          jd.BinaryOperation{jd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(jd.Mult(), jd.BinaryOperation{jd.Mult}(d, A), x) ==
          jd.BinaryOperation{jd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(jd.Mult(), x, jd.BinaryOperation{jd.Mult}(A, d)) ==
          jd.BinaryOperation{jd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(jd.Mult(), x, jd.BinaryOperation{jd.Mult}(d, A)) ==
          jd.BinaryOperation{jd.Mult}(Tensor("x", Lower(4)), A)

    @test evaluate(jd.Mult(), jd.BinaryOperation{jd.Mult}(d, A), x) ==
          jd.BinaryOperation{jd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(jd.Mult(), jd.BinaryOperation{jd.Mult}(A, d), x) ==
          jd.BinaryOperation{jd.Mult}(Tensor("x", Lower(4)), A)
    @test evaluate(jd.Mult(), x, jd.BinaryOperation{jd.Mult}(d, A)) ==
          jd.BinaryOperation{jd.Mult}(A, Tensor("x", Lower(4)))
    @test evaluate(jd.Mult(), x, jd.BinaryOperation{jd.Mult}(A, d)) ==
          jd.BinaryOperation{jd.Mult}(A, Tensor("x", Lower(4)))
end

@testset "diff Tensor" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    A = Tensor("A", Upper(4), Lower(5))

    @test jd.diff(x, x) == KrD(Upper(2), Lower(2))
    @test jd.diff(y, x) == Zero(Upper(3), Lower(2))
    @test jd.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test jd.diff(x, Tensor("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test jd.diff(y, Tensor("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test jd.diff(A, Tensor("A", Upper(6), Lower(7))) ==
          jd.BinaryOperation{jd.Mult}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Tensor("x", Upper(3))
    y = Tensor("y", Lower(4))
    A = Tensor("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test jd.diff(d, x) == jd.BinaryOperation{jd.Mult}(d, Zero(Lower(3)))
    @test jd.diff(d, y) == jd.BinaryOperation{jd.Mult}(d, Zero(Upper(4)))
    @test jd.diff(d, A) == jd.BinaryOperation{jd.Mult}(d, Zero(Lower(5), Upper(6)))
end

@testset "diff BinaryOperation{jd.Mult}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(2))

    op = jd.BinaryOperation{jd.Mult}(x, y)

    D = jd.diff(op, Tensor("x", Upper(3)))

    @test typeof(D) == jd.BinaryOperation{jd.Add}
    @test D.arg1 == jd.BinaryOperation{jd.Mult}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == jd.BinaryOperation{jd.Mult}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{AdditiveOperation}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))

    for op ∈ (jd.Add, jd.Sub)
        v = jd.BinaryOperation{op}(x, y)

        D = jd.diff(v, Tensor("x", Upper(3)))

        @test D == jd.BinaryOperation{op}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
    end
end

@testset "diff trace" begin
    A = Tensor("A", Upper(1), Lower(2))

    op = tr(A)

    D = jd.diff(op, Tensor("A", Upper(3), Lower(4)))

    @test equivalent(evaluate(D), KrD(Upper(1), Lower(2)))
end

@testset "diff sin" begin
    x = Tensor("x", Upper(2))

    op = sin(x)

    D = jd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, jd.BinaryOperation{jd.Mult}(jd.Cos(x), KrD(Upper(2), Lower(3))))
end

@testset "diff cos" begin
    x = Tensor("x", Upper(2))

    op = cos(x)

    D = jd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, jd.BinaryOperation{jd.Mult}(jd.Negate(jd.Sin(x)), KrD(Upper(2), Lower(3))))
end

@testset "diff negated vector" begin
    x = Tensor("x", Upper(2))

    op = -x

    D = jd.diff(op, Tensor("x", Upper(3)))

    @test equivalent(D, jd.Negate(KrD(Upper(2), Lower(3))))
end

@testset "free indices constant after evaluate" begin
    x = Tensor("x", Upper(2))
    c = Tensor("c", Upper(3))
    y = Tensor("y", Upper(4))

    op1 = (y .* c)' * x

    @test isempty(jd.get_free_indices(op1))
    @test isempty(jd.get_free_indices(evaluate(op1)))

    op2 = tr(x * x')

    @test isempty(jd.get_free_indices(op2))
    @test isempty(jd.get_free_indices(evaluate(op2)))
end

@testset "KrD collapsed correctly on element wise jd.Multiplications" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    e = (y .* z)' * x

    expected = jd.BinaryOperation{jd.Mult}(Tensor("y", Lower(1)), Tensor("x", Lower(1)))

    @test equivalent(evaluate(jd.diff(e, Tensor("z", Upper(9)))), expected)

    expected = jd.BinaryOperation{jd.Mult}(Tensor("y", Upper(1)), Tensor("x", Upper(1)))
    @test equivalent(evaluate(jd.diff(e, Tensor("z", Upper(9)))'), expected)
    @test equivalent(evaluate(jd.diff(e', Tensor("z", Upper(9)))), expected)
    @test equivalent(evaluate(evaluate(jd.diff(e, Tensor("z", Upper(9))))'), expected)
end

@testset "KrD collapsed correctly on element wise jd.Multiplications (mirrored)" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    e = x' * (y .* z)

    expected = jd.BinaryOperation{jd.Mult}(Tensor("y", Lower(1)), Tensor("x", Lower(1)))

    @test equivalent(evaluate(jd.diff(e, Tensor("z", Upper(9)))), expected)

    expected = jd.BinaryOperation{jd.Mult}(Tensor("y", Upper(1)), Tensor("x", Upper(1)))
    @test equivalent(evaluate(jd.diff(e, Tensor("z", Upper(9)))'), expected)
    @test equivalent(evaluate(jd.diff(e', Tensor("z", Upper(9)))), expected)
    @test equivalent(evaluate(evaluate(jd.diff(e, Tensor("z", Upper(9))))'), expected)
end

@testset "Differentiate Ax" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(jd.diff(A * x, Tensor("x", Upper(4))), A)
end

@testset "Differentiate xᵀA " begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(jd.diff(x' * A, Tensor("x", Upper(4))), Tensor("A", Lower(1), Lower(2)))
end

@testset "Differentiate xᵀAx" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = jd.diff(x' * A * x, Tensor("x", Upper(4)))

    @test equivalent(evaluate(D.arg1), evaluate(x' * A))
    @test equivalent(
        evaluate(D.arg2),
        evaluate(jd.BinaryOperation{jd.Mult}(Tensor("A", Lower(1), Lower(3)), x)),
    )
end

@testset "Differentiate xx'x" begin
    x = Tensor("x", Upper(1))

    l = jd.BinaryOperation{jd.Mult}(Tensor("x", Upper(100)), Tensor("x", Lower(2)))
    rl = jd.BinaryOperation{jd.Mult}(Tensor("x", Upper(100)), Tensor("x", Lower(2)))
    rr = jd.BinaryOperation{jd.Mult}(KrD(Upper(100), Lower(2)), x'*x)
    expected = jd.BinaryOperation{jd.Add}(l, jd.BinaryOperation{jd.Add}(rl, rr))

    D = jd.diff(x * x' * x, Tensor("x", Upper(2)))

    @test equivalent(evaluate(D), expected)
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = jd.diff(A * (x + 2 * x), Tensor("x", Upper(4)))

    @test equivalent(evaluate(D), 3 * A)
end

# TODO: Fails because KrD(Upper(3), Lower(3)) * Tensor("A", Upper(1), Lower(3)) isn't evaluated properly
# @testset "Differentiate A(x + 2x)" begin
#     A = Tensor("A", Upper(1), Lower(2))
#     x = Tensor("x", Upper(3))

#     D = jd.diff(A * (x + 2 * x), Tensor("x", Upper(3)))

#     expected = jd.BinaryOperation{jd.Mult}(3, KrD(Upper(3), Lower(3)))
#     expected = jd.BinaryOperation{jd.Mult}(expected, Tensor("A", Upper(1), Lower(3)))

#     @test equivalent(evaluate(D), expected)
# end

@testset "Differentiate A(2x + x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = jd.diff(A * (x + 2 * x), Tensor("x", Upper(4)))

    @test equivalent(evaluate(D), 3 * A)
end

@testset "evaluated derivative is equal to derivative of evaluated expression" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))
    y = Tensor("y", Upper(4))
    c = Tensor("c", Upper(5))

    exprs = ( #
        A * x, #
        x' * A, #
        x' * A * x, #
        A * (x + 2 * x), #
        A * (2 * x + x), #
        x' * x, #
        tr(x * x'), #
        (y .* c)' * x, #
        (x .* c)' * x, #
        (x + y)' * x, #
        (x - y)' * x, #
        x' * (y .* c), #
        x' * (x .* c), #
        x' * (x + y), #
        x' * (x - y), #
        sin(tr(x * x')), #
        cos(tr(x * x')), #
        tr(sin(x * x')), #
        tr(A), #
    )

    for expr ∈ exprs
        @testset "$(to_string(expr))" begin
            @test evaluate(jd.diff(evaluate(expr), Tensor("A", Upper(10), Lower(11)))) == evaluate(jd.diff(expr, Tensor("A", Upper(10), Lower(11))))
            @test evaluate(jd.diff(evaluate(expr), Tensor("x", Upper(10)))) == evaluate(jd.diff(expr, Tensor("x", Upper(10))))
            @test evaluate(jd.diff(evaluate(expr), Tensor("y", Upper(10)))) == evaluate(jd.diff(expr, Tensor("y", Upper(10))))
            @test evaluate(jd.diff(evaluate(expr), Tensor("c", Upper(10)))) == evaluate(jd.diff(expr, Tensor("c", Upper(10))))
        end
    end
end
