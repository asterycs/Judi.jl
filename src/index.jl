import Base.hash

export Upper, Lower

Letter = Int

struct Upper
    letter::Letter
end

struct Lower
    letter::Letter
end

LowerOrUpperIndex = Union{Lower,Upper}

function flip(index::Lower)
    return Upper(index.letter)
end

function flip(index::Upper)
    return Lower(index.letter)
end

function flip_to(index::Lower, letter::Letter)
    return Upper(letter)
end

function flip_to(index::Upper, letter::Letter)
    return Lower(letter)
end

function same_to(old::Lower, letter::Letter)
    return Lower(letter)
end

function same_to(old::Upper, letter::Letter)
    return Upper(letter)
end

function hash(arg::LowerOrUpperIndex)
    h::UInt = 0

    if typeof(arg) == Lower
        h |= 1
    end

    h |= arg.letter << 1

    h
end

# TODO: Find a better solution for this. Perhaps a set with taken indices?
const global NEXT_LETTER = Ref{Letter}(100)

function get_next_letter()
    tmp = NEXT_LETTER[]
    NEXT_LETTER[] += 1

    return tmp
end
