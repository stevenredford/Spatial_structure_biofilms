struct DirectDiffusionNH <: NutrientHandler
    num_steps::Int
end

# FIX: This is not correct, needs a fix
function handle_nutrients!(dd::DirectDiffusionNH, model)
    diff_dt = model.dt / dd.num_steps

    # do diffusion
    @inbounds for _ in 1:dd.num_steps
        for (nut, nut_temp, np) in zip(model.nutrients, model.nutrients_temp, model.nutrient_props)
            compute_laplacian_absorbing!(nut_temp, nut, model.dx,)
            @. nut += diff_dt * np.D * nut_temp
        end
    end

    update_nutrients!(model)
end

function update_nutrients!(model)
    # Zero out the pre-allocated changes arrays
    @inbounds for changes in model.nutrients_temp
        changes .= 0.0
    end

    # Process all cells and accumulate changes
    @inbounds for cell in allagents(model)
        !cell.alive && continue

        strain = model.strain_props[cell.strain_id]
        pos = cell.pos

        # Handle uptake and release for each nutrient
        for n_idx in eachindex(model.nutrients)
            if strain.uptakes[n_idx]
                # Calculate uptake rate (per hour)
                uptake_rate = calculate_uptake(
                    model.nutrients[n_idx][pos...],
                    strain,
                    n_idx
                )
                # Apply timestep to get actual uptake
                uptake = uptake_rate * model.dt
                model.nutrients_temp[n_idx][pos...] -= uptake
                cell.internal_nutrients[n_idx] += uptake
            end

            if strain.releases[n_idx]
                release = strain.release_rates[n_idx] * model.dt
                model.nutrients_temp[n_idx][pos...] += release
            end
        end
    end

    # Apply changes using forward Euler
    @inbounds for n_idx in eachindex(model.nutrients)
        @. model.nutrients[n_idx] += model.nutrients_temp[n_idx]
    end
end

function calculate_uptake(nutrient, strain_props::StrainProps, nutrient_idx)
    vmax = strain_props.uptake_vmaxs[nutrient_idx]
    K = strain_props.uptake_Ks[nutrient_idx]
    # Return rate per hour (no dt multiplication here)
    return vmax * nutrient / (nutrient + K)
end
