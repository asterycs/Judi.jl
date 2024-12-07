using MatrixDiff
using Test

MD = MatrixDiff

@testset "create column vector" begin
    x = create_vector("x")
    y = create_vector("y")
    A = create_vector("A")

    # TODO: Find a better way to keep track of the indices and remove all "equivalent"
    @test equivalent(x, Tensor("x", Upper(1)))
    @test equivalent(y, Tensor("y", Upper(2)))
    @test equivalent(A, Tensor("A", Upper(3)))
end

@testset "create matrix" begin
    A = create_matrix("A")
    B = create_matrix("B")
    X = create_matrix("X")

    @test equivalent(A, Tensor("A", Upper(4), Lower(5)))
    @test equivalent(B, Tensor("B", Upper(6), Lower(7)))
    @test equivalent(X, Tensor("X", Upper(8), Lower(9)))
end

@testset "to_std_string output is correct with matrix-vector contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    At = Tensor("A", Lower(1), Upper(2))
    x = Tensor("x", Upper(2))
    xt = Tensor("x", Lower(2))
    y = Tensor("y", Upper(1))
    yt = Tensor("y", Lower(1))

    function contract(l, r)
        return evaluate(MD.BinaryOperation{*}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, yt)) == "yᵀA"
    @test to_std_string(contract(yt, A)) == "yᵀA"
    @test to_std_string(contract(At, xt)) == "xᵀAᵀ"
    @test to_std_string(contract(xt, At)) == "xᵀAᵀ"
    @test to_std_string(contract(At, y)) == "Aᵀy"
    @test to_std_string(contract(y, At)) == "Aᵀy"
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

@testset "derivative interface checks" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test equivalent(derivative(x' * A * x, "A"), evaluate(x * x')) # scalar input works
    @test equivalent(derivative(A * x, "x"), A) # vector input works
    @test_throws DomainError derivative(A * x, "ö") # ö is undefined
end

@testset "gradient interface checks" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test_throws DomainError gradient(A * x, "x") # input not a scalar
    @test_throws DomainError gradient(x' * A * x, "A") # A is a matrix
    @test_throws DomainError gradient(x' * A * x, "ö") # ö is undefined
end

@testset "jacobian interface checks" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test_throws DomainError jacobian(x' * A * x, "x") # input not a vector
    @test_throws DomainError jacobian(A * x, "A") # A is a matrix
    @test_throws DomainError jacobian(A * x, "ö") # ö is undefined
end

@testset "hessian interface checks" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test_throws DomainError hessian(A * x, "x") # input not a scalar
    @test_throws DomainError hessian(x' * A * x, "A") # A is a matrix
    @test_throws DomainError hessian(x' * A * x, "ö") # ö is undefined
end

@testset "to_std_string of gradient" begin
    x = create_vector("x")
    y = create_vector("y")
    c = create_vector("c")

    # TODO: Ensure evaluate(diff') == gradient
    @test to_std_string(gradient(x' * x, "x")) == "2x"
    @test to_std_string(gradient(tr(x * x'), "x")) == "2x"
    @test to_std_string(gradient((y .* c)' * x, "x")) == "y ⊙ c"
    @test to_std_string(gradient((x .* c)' * x, "x")) == "2y ⊙ c"
    @test to_std_string(gradient((x + y)' * x, "x")) == "x + y + x"
    @test to_std_string(gradient((x - y)' * x, "x")) == "x - y + x"
end

@testset "to_std_string of jacobian {A, A'} * x" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test to_std_string(jacobian(A * x, "x")) == "A"
    @test to_std_string(jacobian(A' * x, "x")) == "Aᵀ"
end

@testset "to_std_string of hessian" begin
    A = create_matrix("A")
    x = create_vector("x")

    @test to_std_string(hessian(x' * A * x, "x")) == "Aᵀ + A"
    @test to_std_string(hessian(2 * x' * A * x, "x")) == "2Aᵀ + 2A"
    @test to_std_string(hessian(2 * x' * x, "x")) == "4I"
end
