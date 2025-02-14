"""
Calculates nutrient uptake rate using Monod kinetics.
Returns uptake rate per hour.
"""
function calculate_uptake(nutrient_conc, strain_deps::StrainDeps, nutrient_idx)
    vmax = strain_deps.uptake_rates.vmax[nutrient_idx]
    K = strain_deps.uptake_rates.K[nutrient_idx]
    # Return rate per hour (no dt multiplication here)
    return vmax * nutrient_conc / (nutrient_conc + K)
end

"""
    compute_laplacian_periodic_xy!(output_matrix, input_matrix, neighbor_weight, center_weight, dx)

Computes the Laplacian using efficient neighbor operations with periodic boundaries in x,y
and no-flux in z. Uses LoopVectorization for performance.
"""
function compute_laplacian_periodic_xy!(output_matrix::Array{T,3}, input_matrix::Array{T,3}, neighbor_weight::Float64, center_weight::Float64, dx::Float64) where {T<:Real}
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

        output_matrix[i, j, k] = (neighbor_weight * (
            input_matrix[im, j, k] + input_matrix[ip, j, k] +
            input_matrix[i, jm, k] + input_matrix[i, jp, k] +
            input_matrix[i, j, km] + input_matrix[i, j, kp]
        ) + center_weight * input_matrix[i, j, k]) / (dx^2)
    end
end

"""
    compute_laplacian_absorbing!(output_matrix, input_matrix, neighbor_weight, center_weight, dx)

TODO: Edit this
Computes the Laplacian using efficient neighbor operations with periodic boundaries in x,y
and no-flux in z. Uses LoopVectorization for performance.
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

"""
Performs multiple steps of nutrient diffusion using Euler forward method.
dt = 1/(time_steps_per_hour * n_diffusion_steps) hours
"""
function diffuse_nutrients!(model, n_diffusion_steps)
    # dt in hours for each diffusion sub-step
    dt = 1.0 / (model.time_steps_per_hour * n_diffusion_steps)

    @inbounds for _ in 1:n_diffusion_steps
        for n_idx in eachindex(model.nutrients)
            # Compute Laplacian using pre-allocated arrays
            compute_laplacian_absorbing!(
                model.nutrients_changes[n_idx],
                model.nutrients[n_idx],
                model.cell_diameter,
                0.0
            )
            # compute_laplacian_periodic_xy!(
            #     model.nutrients_changes[n_idx],
            #     model.nutrients[n_idx],
            #     model.neighbor_weight,
            #     model.center_weight,
            #     model.cell_diameter
            # )
            # Forward Euler step: u_{n+1} = u_n + dt * D * ∇²u_n
            @. model.nutrients[n_idx] += dt * model.D * model.nutrients_changes[n_idx]
        end
    end
end
