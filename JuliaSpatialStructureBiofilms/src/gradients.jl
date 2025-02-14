"""
    compute_laplacian_periodic_xy!(output_matrix, input_matrix, dx)

Computes the Laplacian using efficient neighbor operations with periodic boundaries in x,y
and no-flux in z.
"""
function compute_laplacian_periodic_xy!(output_matrix::Array{T,3}, input_matrix::Array{T,3}, dx::Float64) where {T<:Real}
    nx, ny, nz = size(input_matrix)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        # periodic in x
        ip = (i == nx ? 1 : i + 1)
        im = (i == 1 ? nx : i - 1)
        # periodic in y
        jp = (j == ny ? 1 : j + 1)
        jm = (j == 1 ? ny : j - 1)
        # no-flux in z
        kp = (k == nz ? k : k + 1)
        km = (k == 1 ? k : k - 1)

        output_matrix[i, j, k] = (
            input_matrix[im, j, k] + input_matrix[ip, j, k] +
            input_matrix[i, jm, k] + input_matrix[i, jp, k] +
            input_matrix[i, j, km] + input_matrix[i, j, kp] -
            6 * input_matrix[i, j, k]) / (dx^2)
    end
end

"""
    compute_laplacian_absorbing!(output_matrix, input_matrix, dx)

Computes the Laplacian using efficient neighbor operations with open boundaries in x,y
and top of z but no-flux in bottom of z.
"""
function compute_laplacian_absorbing!(output_matrix::Array{T,3}, input_matrix::Array{T,3}, dx::Float64, outside_val=0.0) where {T<:Real}
    nx, ny, nz = size(input_matrix)

    @inbounds for k in 1:nz
        for j in 1:ny, i in 1:nx
            output_matrix[i, j, k] = 0.0
            if (i != 1)
                output_matrix[i, j, k] += input_matrix[i-1, j, k] - input_matrix[i, j, k]
            else
                output_matrix[i, j, k] += outside_val - input_matrix[i, j, k]
            end
            if (i != nx)
                output_matrix[i, j, k] += input_matrix[i+1, j, k] - input_matrix[i, j, k]
            else
                output_matrix[i, j, k] += outside_val - input_matrix[i, j, k]
            end
            if (j != 1)
                output_matrix[i, j, k] += input_matrix[i, j-1, k] - input_matrix[i, j, k]
            else
                output_matrix[i, j, k] += outside_val - input_matrix[i, j, k]
            end
            if (j != ny)
                output_matrix[i, j, k] += input_matrix[i, j+1, k] - input_matrix[i, j, k]
            else
                output_matrix[i, j, k] += outside_val - input_matrix[i, j, k]
            end
            # for the z=1 layer don't diffuse through boundary
            if (k != 1)
                output_matrix[i, j, k] += input_matrix[i, j, k-1] - input_matrix[i, j, k]
            end
            if (k != nz)
                output_matrix[i, j, k] += input_matrix[i, j, k+1] - input_matrix[i, j, k]
            else
                output_matrix[i, j, k] += outside_val - input_matrix[i, j, k]
            end
            output_matrix[i, j, k] /= dx^2
        end
    end
end
