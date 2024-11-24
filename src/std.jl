export create_matrix
export create_vector

export differential

const global REGISTERED_SYMBOLS = Dict{String, Tensor}()

function create_matrix(name::String)
    T = Tensor(name, Upper(get_next_letter()), Lower(get_next_letter()))

    REGISTERED_SYMBOLS[name] = T

    return T
end

function create_vector(name::String)
    T = Tensor(name, Upper(get_next_letter()))

    REGISTERED_SYMBOLS[name] = T

    return T
end

function differential(expr, wrt::String)
    ∂ = Tensor(wrt)

    if wrt ∈ keys(REGISTERED_SYMBOLS)
        for index ∈ REGISTERED_SYMBOLS[wrt].indices
            push!(∂.indices, same(index, get_next_letter()))
        end
    end

    linear_form = diff(expr, ∂)
    linear_form = evaluate(linear_form)

    linear_form
end
