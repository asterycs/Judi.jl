using MatrixDiff
using Test

MD = MatrixDiff

@testset "create column vector" begin
    x = create_vector("x")
    y = create_vector("y")
    A = create_vector("A")

    @test x == Tensor("x", Upper(1))
    @test y == Tensor("y", Upper(2))
    @test A == Tensor("A", Upper(3))
end

@testset "create matrix" begin
    A = create_matrix("A")
    B = create_matrix("B")
    X = create_matrix("X")

    @test A == Tensor("A", Upper(4), Lower(5))
    @test B == Tensor("B", Upper(6), Lower(7))
    @test X == Tensor("X", Upper(8), Lower(9))
end

@testset "to_std_string output is correct with matrix-vector contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, y')) == "yᵀA"
    @test to_std_string(contract(y', A)) == "yᵀA"
    @test to_std_string(contract(A', x')) == "xᵀAᵀ"
    @test to_std_string(contract(x', A')) == "xᵀAᵀ"
    @test to_std_string(contract(A', y)) == "Aᵀy"
    @test to_std_string(contract(y, A')) == "Aᵀy"
end

@testset "to_std_string output is correct with all covariant bilinar form-vector contraction" begin
    A = Tensor("A", Lower(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, x)) == "xᵀAᵀ"
    @test to_std_string(contract(x, A)) == "xᵀAᵀ"
    @test to_std_string(contract(A, y)) == "yᵀA"
    @test to_std_string(contract(y, A)) == "yᵀA"
end

@testset "to_std_string output is correct with all contravariant bilinear form-vector contraction" begin
    A = Tensor("A", Upper(1), Upper(2))
    x = Tensor("x", Lower(2))
    y = Tensor("y", Lower(1))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, y)) == "Aᵀy"
    @test to_std_string(contract(y, A)) == "Aᵀy"
end

@testset "to_std_string output is correct with matrix-matrix contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    C = Tensor("C", Lower(1), Upper(3))
    D = Tensor("D", Lower(3), Upper(2))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, B)) == "AB"
    @test to_std_string(contract(B, A)) == "AB"
    @test to_std_string(contract(A, C)) == "CᵀA"
    @test to_std_string(contract(C, A)) == "CᵀA"
    @test to_std_string(contract(A, D)) == "ADᵀ"
    @test to_std_string(contract(D, A)) == "ADᵀ"
    @test to_std_string(contract(C, D)) == "DᵀCᵀ"
    @test to_std_string(contract(D, C)) == "DᵀCᵀ"
end

@testset "to_std_string output is correct with matrix-matrix sum" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Lower(2))
    C = Tensor("C", Lower(2), Upper(1))

    function sum(l, r)
        return evaluate(MD.BinaryOperation{+}(l, r))
    end

    @test to_std_string(sum(A, B)) == "A + B"
    @test to_std_string(sum(B, A)) == "B + A"
    @test to_std_string(sum(A, C)) == "A + Cᵀ"
    @test to_std_string(sum(C, A)) == "Cᵀ + A"
end
