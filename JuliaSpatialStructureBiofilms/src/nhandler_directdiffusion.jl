"""
This is the ideal model implementation which actually does a finite difference
sim of the diffusion, uptake and release by cells with a smaller diff_dt than
the life/death steps. The downside being that its slow!
NOTE: In particular, this has been implemented to definitelly be correct, and
has not been optimized yet.
"""
struct DirectDiffusionNH{LFT<:Function} <: NutrientHandler
    num_steps::Int
    compute_laplacian_func!::LFT
end
function make_directdiffusion_smart(N_or_dt=100, bcs=:PBC; dt)
    num_steps = isa(N_or_dt, Int) ? N_or_dt : ceil(dt / N_or_dt)
    laplacian_func = if bcs == :periodic
        compute_laplacian_periodic_xy!
    elseif bcs == :absorbing
        compute_laplacian_absorbing!
    else
        throw(ArgumentError(@sprintf "unrecognized bcs of %s" string(bcs)))
    end
    DirectDiffusionNH(num_steps, laplacian_func)
end
export DirectDiffusionNH, make_directdiffusion_smart

function handle_nutrients!(dd::DirectDiffusionNH, model)
    diff_dt = model.dt / dd.num_steps

    @inbounds for _ in 1:dd.num_steps
        # diffuse nutrients from last step, NOTE: much of this could be threaded
        # also, uses and resets all nutrients_temp to 0.
        for (nut, temp, np) in zip(model.nutrients, model.nutrients_temp, model.nutrient_props)
            dd.compute_laplacian_func!(temp, nut, model.dx)
            for i in eachindex(nut)
                nut[i] += diff_dt * np.D * temp[i]
                temp[i] = 0.0
            end
        end

        # do cell uptakes and releases in the same loop for performance,
        # however do not directly override nutrients for world age type issues
        # instead set the next value of nutrients to nutrients_temp which has
        # been reset in the diffusion process. This is equivalent to doing
        # all uptakes/consumptions first and then all releases/productions.
        @inbounds for cell in allagents(model)
            if cell.alive
                strain = model.strain_props[cell.strain_id]
                for nut_i in eachindex(model.nutrients)
                    if strain.uptakes[nut_i]
                        nut_cell = model.nutrients[nut_i][cell.pos...]
                        ratio = nut_cell / (nut_cell + strain.uptake_Ks[nut_i] * model.dx^3)
                        uptake = strain.uptake_vmaxs[nut_i] * ratio * diff_dt

                        cell.internal_nutrients[nut_i] += uptake
                        model.nutrients_temp[nut_i][cell.pos...] -= uptake
                    end

                    if strain.releases[nut_i]
                        model.nutrients_temp[nut_i][cell.pos...] += strain.release_rates[nut_i] * diff_dt
                    end
                end
            end
        end

        # Finally actually update the real nutrients
        @inbounds for n_idx in eachindex(model.nutrients)
            @. model.nutrients[n_idx] += model.nutrients_temp[n_idx]
        end
    end
end
