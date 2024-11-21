export create_matrix
export create_vector

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
