module JuliaSpatialStructureBiofilms

using Reexport
@reexport using Agents

using Random
using Base.Threads

################################################################################
# Setting up structs and making the model
################################################################################
"""
Enhanced Cell agent with strain type and internal nutrient tracking
"""
@agent struct CoSMOCell(GridAgent{3})
    alive::Bool = true
    quiescence_flag::Bool = false
    strain_id::Int = 1     # References the strain in strain_deps
    internal_nutrients::Vector{Float64} = Float64[]
end

"""
    StrainDeps

Holds the nutrient-uptake and -release properties for a single strain.
Includes both boolean flags for capabilities and specific rates for each process.
"""
struct StrainDeps
    label::String
    can_uptake::Vector{Bool}
    can_release::Vector{Bool}
    uptake_rates::NamedTuple{(:vmax, :K),Tuple{Vector{Float64},Vector{Float64}}}  # Monod parameters
    release_rates::Vector{Float64}
    alpha::Vector{Float64}  # nutrient required for reproduction
    growth_rate::Float64    # maximum growth rate
    death_rate::Float64     # death rate
end

"""
    build_model(config::NamedTuple)

Builds the initial model with spatial grid and nutrient fields.
"""
function build_model(;
    dims=(100, 100, 100),
    alive_probability=0.2,
    metric=:chebyshev,
    seed=42,
    time_steps_per_hour=10,
    n_diffusion_steps_per_model_step=360,
    D=360,
    cell_diameter=5.0
)
    space = GridSpaceSingle(dims; metric)

    # Create strain dependencies
    strain_deps = cosmo_straindeps()
    n_nutrients = length(first(strain_deps).can_uptake)
    @show n_nutrients

    # Initialize nutrient fields matching the spatial dimensions
    nutrients = [zeros(Float64, dims) for _ in 1:n_nutrients]

    # Pre-allocate nutrient_changes arrays with same dimensions
    nutrients_changes = [zeros(Float64, dims) for _ in 1:n_nutrients]

    # Remove padded matrices as they're no longer needed
    properties = (;
        nutrients=nutrients,
        nutrients_changes=nutrients_changes,
        center_weight=-6.0,  # Center coefficient
        neighbor_weight=1.0, # Neighbor coefficient
        strain_deps=strain_deps,
        time_steps_per_hour,
        n_diffusion_steps_per_model_step,
        D,
        cell_diameter
    )

    model = StandardABM(
        CoSMOCell,
        space;
        properties,
        (model_step!)=general_model_step!,
        rng=MersenneTwister(seed),
        container=Vector
    )

    # Initialize cells only on the bottom layer with random strain types
    for pos in positions(model)
        if pos[3] == 1 && rand(abmrng(model)) < alive_probability
            if (30 < pos[1] < 70) && (30 < pos[2] < 70)
                strain_id = rand(abmrng(model), 1:length(strain_deps))
                internal_nutrients = zeros(Float64, n_nutrients)
                add_agent!(
                    pos,
                    model;
                    alive=true,
                    strain_id=strain_id,
                    internal_nutrients=internal_nutrients
                )
            end
        end
    end

    return model
end

################################################################################
# Model evolution/steps -- the most complex bit by far
################################################################################
include("spreadresources_direct.jl")
"""
Updates nutrient concentrations based on cell uptake and release using Euler forward method.
dt = 1/time_steps_per_hour hours
"""
function update_nutrients!(model)
    dt = 1.0 / model.time_steps_per_hour  # dt in hours

    # Zero out the pre-allocated changes arrays
    @inbounds for changes in model.nutrients_changes
        changes .= 0.0
    end

    # Process all cells and accumulate changes
    @inbounds for cell in allagents(model)
        !cell.alive && continue

        strain = model.strain_deps[cell.strain_id]
        pos = cell.pos

        # Handle uptake and release for each nutrient
        for n_idx in eachindex(model.nutrients)
            if strain.can_uptake[n_idx]
                # Calculate uptake rate (per hour)
                uptake_rate = calculate_uptake(
                    model.nutrients[n_idx][pos...],
                    strain,
                    n_idx
                )
                # Apply timestep to get actual uptake
                uptake = uptake_rate * dt
                model.nutrients_changes[n_idx][pos...] -= uptake
                cell.internal_nutrients[n_idx] += uptake
            end

            if strain.can_release[n_idx]
                # Release_rates are per hour, multiply by dt
                release = strain.release_rates[n_idx] * dt
                model.nutrients_changes[n_idx][pos...] += release
            end
        end
    end

    # Apply changes using forward Euler
    @inbounds for n_idx in eachindex(model.nutrients)
        @. model.nutrients[n_idx] += model.nutrients_changes[n_idx]
    end
end

"""
Checks if a cell can reproduce based on nutrient requirements.
Uses dt = 1/time_steps_per_hour hours for probability calculation.
"""
function can_reproduce(cell, model)
    strain = model.strain_deps[cell.strain_id]
    dt = 1.0 / model.time_steps_per_hour  # dt in hours

    # Check each required nutrient
    for n_idx in eachindex(model.nutrients)
        if strain.can_uptake[n_idx]
            if cell.internal_nutrients[n_idx] < strain.alpha[n_idx]
                return false
            end
        end
    end

    # Convert hourly rate to probability per timestep
    return rand(abmrng(model)) <= strain.growth_rate * dt
end

"""
Determines if a cell should die based on death rate.
Uses dt = 1/time_steps_per_hour hours for probability calculation.
"""
function should_die(cell, model)
    strain = model.strain_deps[cell.strain_id]
    dt = 1.0 / model.time_steps_per_hour  # dt in hours
    # Convert hourly rate to probability per timestep
    return rand(abmrng(model)) <= strain.death_rate * dt
end

"""
Creates a daughter cell with inherited properties.
"""
function create_daughter!(new_pos, parent, model)
    n_nutrients = length(model.nutrients)
    add_agent!(
        new_pos,
        model;
        alive=true,
        strain_id=parent.strain_id,
        internal_nutrients=zeros(Float64, n_nutrients)
    )
    # Reset parent's internal nutrients after division
    parent.internal_nutrients .= 0.0
end

"""
Main stepping function that handles both nutrient dynamics and cell life cycle.
"""
function general_model_step!(model)
    # First run multiple diffusion steps
    diffuse_nutrients!(model, model.n_diffusion_steps_per_model_step)  # You can adjust the number of steps

    # Then update nutrients based on cell uptake/release
    update_nutrients!(model)

    # Finally handle cell lifecycle
    @inbounds for cell in allagents(model)
        !cell.alive && continue

        # Handle death
        if should_die(cell, model)
            cell.alive = false
            continue
        end

        # Handle reproduction
        if !cell.quiescence_flag && can_reproduce(cell, model)
            new_pos = find_bud_pos(cell, model)
            if new_pos !== nothing
                create_daughter!(new_pos, cell, model)
            else
                cell.quiescence_flag = true
            end
        end
    end
end

"""
Finds new budding position. First in the same layer, then in the layer above.
Returns the position if found, otherwise returns nothing.
"""
function find_bud_pos(cell, model)
    ## Try the same layer
    for new_pos in nearby_positions(cell, model)
        if new_pos[3] == cell.pos[3] && isempty(new_pos, model)
            return new_pos
        end
    end
    ## Try layer above
    for new_pos in nearby_positions(cell, model)
        if new_pos[3] == cell.pos[3] + 1 && isempty(new_pos, model)
            return new_pos
        end
    end
    return nothing
end


################################################################################
# Cosmo stuff and particular tests/runs
################################################################################
"""
    cosmo_straindeps()

Creates the strain dependencies for the classic CoSMO setup with Lys/Ade cross-feeding.
"""
function cosmo_straindeps()
    # Strain 1: Lys auxotroph (needs Lys, produces Ade)
    strain1 = StrainDeps(
        "Lys+ Ade-",               # label
        [true, false],             # can_uptake: [Lys, Ade]
        [false, true],             # can_release
        (
            vmax=[5.4 * 0.51, 0.0],  # vmL for Lys, 0 for Ade
            K=[2.1e6, 0.0]           # KL for Lys, 0 for Ade
        ),
        [0.0, 0.4],               # release_rates: no Lys, gammaA for Ade
        [5.4, 0.0],               # alpha: alphaL for Lys, 0 for Ade
        0.51,                     # rL: growth rate
        0.021                     # dL: death rate
    )

    # Strain 2: Ade auxotroph (needs Ade, produces Lys)
    strain2 = StrainDeps(
        "Lys- Ade+",              # label
        [false, true],            # can_uptake: [Lys, Ade]
        [true, false],            # can_release
        (
            vmax=[0.0, 3.1 * 0.44],  # 0 for Lys, vmA for Ade
            K=[0.0, 1.3e6]           # 0 for Lys, KA for Ade
        ),
        [0.26, 0.0],             # release_rates: gammaL for Lys, 0 for Ade
        [0.0, 3.1],              # alpha: 0 for Lys, alphaA for Ade
        0.44,                     # rA: growth rate
        0.015                     # dA: death rate
    )

    return [strain1, strain2]
end

# make model
# m = build_model()
# run it for n steps:
# @time step!(m, n)
# plot nutrients with colorbar:
# f = Figure(); ax = Axis(f[1,1]); hm = heatmap!(ax, m.nutrients[1][50,:,:]); Colorbar(f[1,2],hm); f
# plot the abm
# xx = abmplot(m; agent_size=1, agent_color=c->c.strain_id); xx[1]


function main()
    # # Run several steps
    model = build_model()
    adata = [:pos, :alive, :quiescence_flag, :strain_id, :internal_nutrients]
    @time adf, _ = run!(model, 240; adata)
    adf[end-10:end, :] # display only the last few rows

    ## 0.005390 seconds (4.29 k allocations: 751.711 KiB)

    model

    # @profview adf, _ = run!(model, 1; adata)


    # # Run 100 model steps
    # @assert false "Stop here not to run the next part"
    #
    # model = build_model(; n_diffusion_steps_per_model_step=360)
    # adata = [:pos, :alive, :quiescence_flag, :strain_id, :internal_nutrients]
    # @time adf, _ = run!(model, 240; adata)
    # adf[end-10:end, :] # display only the last few rows
    # 232.001167 seconds (505.80 k allocations: 108.956 MiB, 0.03% gc time)
end

end # module JuliaSpatialStructureBiofilms
