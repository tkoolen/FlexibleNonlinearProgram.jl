function num_triangular_elements(dimension::Int64)
    div(dimension * (dimension + 1), 2)
end

function copy_lower_triangle_column_major!(dest::AbstractVector, source::AbstractMatrix)
    i = 1
    cols = 1 : size(source, 1)
    num_rows = size(source, 2)
    for col in cols
        for row = col : num_rows
            dest[i] = source[row, col]
            i += 1
        end
    end
end
