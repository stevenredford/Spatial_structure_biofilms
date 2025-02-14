module JuliaSpatialStructureBiofilms

using Reexport
@reexport using Agents

using Printf
using Random
using Base.Threads

################################################################################
# Setting up structs and making the model
################################################################################
"""
Enhanced Cell agent with strain type and internal nutrient tracking
"""
@agent struct Cell(GridAgent{3})
    alive::Bool = true
    quiescence_flag::Bool = false
    strain_id::Int = 1
    internal_nutrients::Vector{Float64} = Float64[]
end
export Cell

"""
    StrainProps

Holds the nutrient-uptake and -release properties for a single strain.
Includes both boolean flags for capabilities and specific rates for each process.
"""
struct StrainProps
    label::String
    uptakes::Vector{Bool}
    releases::Vector{Bool}
    uptake_vmaxs::Vector{Float64}
    uptake_Ks::Vector{Float64}
    release_rates::Vector{Float64}
    alpha::Vector{Float64}  # amounts of nutrients required for reproduction
    growth_rate::Float64    # maximum growth rate
    death_rate::Float64     # death rate
end
struct NutrientProps
    D::Float64
end
export StrainProps, NutrientProps

abstract type NutrientHandler end
function handle_nutrients!(nh::NutrientHandler, model)
    throw(ErrorException(@sprintf "no method handle_nutrients! was defined for %s" string(typeof(nh))))
end
handle_nutrients!(m::AgentBasedModel) = handle_nutrients!(m.nutrient_handler, m)
export NutrientHandler, handle_nutrients!

struct ModelProperties{NH<:NutrientHandler}
    dt::Float64
    dx::Float64
    strain_props::Vector{StrainProps}
    nutrient_props::Vector{NutrientProps}
    nutrients::Vector{Array{Float64,3}}
    nutrients_temp::Vector{Array{Float64,3}}
    nutrient_handler::NH
end
export ModelProperties

################################################################################
# Model evolution/steps -- the most complex bit by far
################################################################################
include("gradients.jl")
include("handle_resources_direct.jl")

"""
Returns true if all internal resources are beyond their alphas.
"""
function can_reproduce(cell, model)
    strain = model.strain_props[cell.strain_id]
    for n_idx in eachindex(model.nutrients)
        if strain.uptakes[n_idx]
            if cell.internal_nutrients[n_idx] < strain.alpha[n_idx]
                return false
            end
        end
    end
    return true
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

"""
Main stepping function that handles both nutrient dynamics and cell life cycle.
"""
function general_model_step!(model)
    # This handles nutrient diffusion, production and consumption between the life/death steps, note that this happens much faster than life/death!
    handle_nutrients!(model)

    # Finally handle cell lifecycle
    @inbounds for cell in allagents(model)
        if cell.alive
            strain = model.strain_props[cell.strain_id]
            # Handle death
            chance_of_death = strain.death_rate * model.dt
            if rand(abmrng(model)) <= chance_of_death
                cell.alive = false
                continue
            end

            # Handle reproduction
            if !cell.quiescence_flag && can_reproduce(cell, model)
                if rand(abmrng(model)) <= strain.growth_rate * model.dt
                    new_pos = find_bud_pos(cell, model)
                    if !isnothing(new_pos)
                        cell.internal_nutrients .= 0.0
                        replicate!(cell, model; pos=new_pos)
                    else
                        cell.quiescence_flag = true
                    end
                end
            end
        end
    end
end

################################################################################
# Cosmo stuff and particular tests/runs
################################################################################
function build_model(;
    # space
    width,
    height=width,
    dx,
    # time, one of these two has to be given
    dt=nothing,
    time_steps_per_hour=nothing,
    # strains and nutrients
    strain_props,
    D,
    # initial setup
    initial_cell_density, # this is for generating the initial agents
    seed=42,
)
    dims = (width, width, height)

    if isnothing(dt) && isnothing(time_steps_per_hour)
        dt = 0.1
    elseif isnothing(dt)
        dt = 1.0 / time_steps_per_hour
    elseif !isnothing(time_steps_per_hour)
        throw(ArgumentError("only one of `dt` and `time_steps_per_hour` is allowed"))
    end

    n_nutrients = length(first(strain_props).uptakes)
    nutrients = [zeros(Float64, dims) for _ in 1:n_nutrients]

    properties = ModelProperties(
        dt, dx,
        strain_props, [NutrientProps(D) for _ in 1:n_nutrients],
        nutrients, map(similar, nutrients),
        DirectDiffusionNH(100)
    )

    model = StandardABM(
        Cell,
        GridSpaceSingle(dims; metric=:chebyshev);
        properties,
        (model_step!)=general_model_step!,
        rng=MersenneTwister(seed),
        container=Vector
    )

    # Initialize cells only on the bottom layer with random strain types
    for pos in positions(model)
        if pos[3] == 1 && rand(abmrng(model)) < initial_cell_density
            if (30 < pos[1] < 70) && (30 < pos[2] < 70)
                strain_id = rand(abmrng(model), 1:length(strain_props))
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
export build_model

"""
Creates the strain dependencies for the classic CoSMO setup with Lys/Ade cross-feeding.
"""
function cosmo_strains()
    # Strain 1: Lys auxotroph (needs Lys, produces Ade)
    strain1 = StrainProps(
        "Lys+ Ade-",               # label
        [true, false],             # uptakes: [Lys, Ade]
        [false, true],             # releases
        [5.4 * 0.51, 0.0],         # vmL for Lys, 0 for Ade
        [2.1e6, 0.0],              # KL for Lys, 0 for Ade
        [0.0, 0.4],                # release_rates: no Lys, gammaA for Ade
        [5.4, 0.0],                # alpha: alphaL for Lys, 0 for Ade
        0.51,                      # rL: growth rate
        0.021                      # dL: death rate
    )

    # Strain 2: Ade auxotroph (needs Ade, produces Lys)
    strain2 = StrainProps(
        "Lys- Ade+",              # label
        [false, true],            # uptakes: [Lys, Ade]
        [true, false],            # releases
        [0.0, 3.1 * 0.44],        # 0 for Lys, vmA for Ade
        [0.0, 1.3e6],             # 0 for Lys, KA for Ade
        [0.26, 0.0],              # release_rates: gammaL for Lys, 0 for Ade
        [0.0, 3.1],               # alpha: 0 for Lys, alphaA for Ade
        0.44,                     # rA: growth rate
        0.015                     # dA: death rate
    )

    return [strain1, strain2]
end
function build_cosmo1()
    build_model(
        width=100,
        dx=5.0,
        time_steps_per_hour=10,
        D=360,
        initial_cell_density=0.2,
        strain_props=cosmo_strains()
    )
end
export build_cosmo1, cosmo_strains

end
