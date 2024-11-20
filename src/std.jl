export create_matrix
export create_vector


function create_matrix(name::String)
    return Tensor(name, Upper(get_next_letter()), Lower(get_next_letter()))
end

function create_vector(name::String)
    return Tensor(name, Upper(get_next_letter()))
end