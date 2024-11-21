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

    @test evaluate(A') == Tensor("A", Lower(1), Upper(2))
    @test evaluate(x') == Tensor("x", Lower(1))
    # @test evaluate(z') == Tensor("z") # Not implemented
end

@testset "evaluate UnaryOperation vector-KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.UnaryOperation(d1, x)) == Tensor("x", Upper(3))
    @test evaluate(MD.UnaryOperation(d2, x)) == Tensor("x", Lower(2))
end

@testset "evaluate UnaryOperation matrix-KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.UnaryOperation(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(MD.UnaryOperation(d2, A)) == Tensor("A", Lower(2), Lower(4))
end

@testset "evaluate UnaryOperation KrD-KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.UnaryOperation(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.UnaryOperation(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation vector-KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.BinaryOperation{*}(d1, x)) == Tensor("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(x, d1)) == Tensor("x", Upper(3))
    @test evaluate(MD.BinaryOperation{*}(d2, x)) == Tensor("x", Lower(2))
    @test evaluate(MD.BinaryOperation{*}(x, d2)) == Tensor("x", Lower(2))
end

@testset "evaluate BinaryOperation matrix-KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(2))

    @test evaluate(MD.BinaryOperation{*}(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d1)) == Tensor("A", Upper(3), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(d2, A)) == Tensor("A", Lower(2), Lower(4))
    @test evaluate(MD.BinaryOperation{*}(A, d2)) == Tensor("A", Lower(2), Lower(4))
end

@testset "evaluate BinaryOperation KrD-KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test evaluate(MD.BinaryOperation{*}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test evaluate(MD.BinaryOperation{*}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation*" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    @test evaluate(MD.BinaryOperation{*}(A, x)) == MD.BinaryOperation{*}(A, x)
    @test evaluate(MD.BinaryOperation{*}(A, y)) == MD.BinaryOperation{*}(A, y)
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
    @test MD.diff(A, Tensor("A", Upper(6), Lower(7))) == MD.BinaryOperation{*}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
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

@testset "diff UnaryOperation" begin
    x = Tensor("x", Upper(2))

    op = MD.UnaryOperation(KrD(Lower(2), Lower(2)), x)

    @test MD.diff(op, Tensor("x", Upper(3))) == MD.UnaryOperation(KrD(Lower(2), Lower(2)), KrD(Upper(2), Lower(3)))
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

@testset "diff BinaryOperation{+}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))

    op = MD.BinaryOperation{+}(x, y)

    D = MD.diff(op, Tensor("x", Upper(3)))

    @test D == MD.BinaryOperation{+}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
end

@testset "Differentiate Ax" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = A * x

    @test equivalent(D(e, Tensor("x", Upper(4))), A)
end

@testset "Differentiate xᵀA " begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A

    @test equivalent(D(e, Tensor("x", Upper(4))), Tensor("A", Lower(1), Lower(2)))
end

@testset "Differentiate xᵀAx" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    differential_form = D(e, Tensor("x", Upper(4)))

    @test equivalent(differential_form.arg1, evaluate(x' * A))
    @test equivalent(differential_form.arg2, evaluate(MD.BinaryOperation{*}(x', Tensor("A", Lower(1), Lower(2)))))
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    differential_form = D(A * (x + 2 * x), Tensor("x", Upper(7)))

    @test equivalent(differential_form, 3 * A)
end

@testset "Differentiate A(2x + x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    e = x' * A * x

    differential_form = D(A * (x + 2 * x), Tensor("x", Upper(7)))

    @test equivalent(differential_form, 3 * A)
end
