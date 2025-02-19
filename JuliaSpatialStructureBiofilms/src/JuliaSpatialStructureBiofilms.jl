module JuliaSpatialStructureBiofilms

using Reexport
@reexport using Agents

@reexport using Printf
@reexport using Random
using Base.Threads

################################################################################
# Setting up types and the core of the model
################################################################################
"""Agent type"""
@agent struct Cell(GridAgent{3})
    alive::Bool = true
    quiescence_flag::Bool = false
    strain_id::Int = 1
    internal_nutrients::Vector{Float64} = Float64[] # U[#nut]
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
    uptake_vmaxs::Vector{Float64}   # U[#nut/T]
    uptake_Ks::Vector{Float64}      # U[#nut/L^3], this one is complex! This will in the numerics be multiplied by dx^3, this is so that we don't have to update this value when changing dx and be more physical, though this is a choice!
    release_rates::Vector{Float64}  # U[#nut/T], simple release rate
    alphas::Vector{Float64}          # U[#nut], amount of nuts. needed for rep.
    growth_rate::Float64            # U[1/T], rep. rate once alphas are met
    death_rate::Float64             # U[1/T]
end
"""Properties of each nutrient, for now very basic..."""
struct NutrientProps
    D::Float64 # U[L^2/T]
end
export StrainProps, NutrientProps

"""
This is setup for flexiblity in how to handle the most numerically intense bit,
the resource dynamics accounting for diffusion, cell release and uptake and
perhaps others like decay or dilutions through boundaries.

The NutrientHandler abstract type forms the interface with the only method
required for it is a handle_nutrients! function which will do exactly that
between timesteps, in particular it ought to update the nutrients property
and each cells internal nutrients according to some logic.
"""
abstract type NutrientHandler end
function handle_nutrients!(nh::NutrientHandler, model)
    throw(ErrorException(@sprintf "no method handle_nutrients! was defined for %s" string(typeof(nh))))
end
handle_nutrients!(m::AgentBasedModel) = handle_nutrients!(m.nutrient_handler, m)
export NutrientHandler, handle_nutrients!

"""
This holds all the interesting properties of the model which will be accesible
through model.? . The Warn type parameter is a Bool flag about doing some checks
which sacrifice some performance, by default it is true (in type for speed).
"""
struct ModelProperties{Warn,NH<:NutrientHandler}
    dt::Float64 # U[T]
    dx::Float64 # U[L]
    strain_props::Vector{StrainProps}
    nutrient_props::Vector{NutrientProps}
    nutrients::Vector{Array{Float64,3}} # U[#nut]
    nutrients_temp::Vector{Array{Float64,3}}
    nutrient_handler::NH
    function ModelProperties{Warn}(dt, dx, strain_props, nutrient_props,
        nutrients, nutrients_temp, nutrient_handler
    ) where {Warn}
        if !isa(Warn, Bool)
            throw(ArgumentError("the passed Warn type parameter is not a Bool"))
        end
        new{Warn,typeof(nutrient_handler)}(dt, dx, strain_props, nutrient_props,
            nutrients, nutrients_temp, nutrient_handler
        )
    end
    ModelProperties(args...) = ModelProperties{true}(args...)
end
export ModelProperties

"""
This should be fairly general so that any runs can use this to make the model.
"""
function build_model(;
    # space
    gridwidth,
    gridheight=gridwidth,
    dx=nothing, # grid spacing, same in all dims
    width=nothing, # physical width in lenght units, used for dx
    # time, one of these two has to be given
    dt=nothing,
    steps_per_unit_time=nothing,
    # strains, this needs to be passed
    strain_props,
    # nutrients
    nutrients=nothing, # initial state
    nutrient_props=nothing, # one of these two needs to be given
    D=nothing,
    # nutrient handler stuff
    nhandler=:dd,
    # initial cell setup
    initial_cell_density=nothing, # this is for generating the initial agents
    seed=42,
)
    dims = (gridwidth, gridwidth, gridheight)

    # set dx
    if isnothing(dx) == isnothing(width)
        throw(ArgumentError("precisely one of dx and width has to be given"))
    elseif isnothing(dx)
        dx = width / gridwidth
    end

    # set dt
    if isnothing(dt) && isnothing(steps_per_unit_time)
        dt = 1.0
    elseif isnothing(dt)
        dt = 1.0 / steps_per_unit_time
    elseif !isnothing(steps_per_unit_time)
        throw(ArgumentError("only one of `dt` and `steps_per_unit_time` is allowed"))
    end

    n_strains = length(strain_props)
    n_nutrients = length(first(strain_props).uptakes)
    # set nutrients
    if isnothing(nutrients)
        nutrients = [zeros(Float64, dims) for _ in 1:n_nutrients]
    end

    # set nutrient_props
    if isnothing(nutrient_props) == isnothing(D)
        throw(ArgumentError("precisely one of nutrient_props and D has to be given"))
    elseif isnothing(nutrient_props)
        nutrient_props = [NutrientProps(D) for _ in 1:n_nutrients]
    end

    # setup up the NutrientHandler
    if !isa(nhandler, NutrientHandler)
        if isa(nhandler, Tuple)
            nhcode, nhargs... = nhandler
        else
            nhcode = nhandler
            nhargs = ()
        end
        if nhcode == :dd
            if length(nhargs) == 0
                nhargs = (100, :periodic)
            elseif length(nhargs) == 1
                nhargs = (nhargs[1], :periodic)
            else
                throw(ArgumentError("could not understand the passed nhandler"))
            end
            nhandler = make_directdiffusion_smart(nhargs...; dt)
        else
            throw(ArgumentError(@sprintf "unrecognized nhandler code %s" string(nhcode)))
        end
    end

    properties = ModelProperties(
        dt, dx,
        strain_props, nutrient_props,
        nutrients, map(similar, nutrients),
        nhandler
    )

    model = StandardABM(
        Cell,
        GridSpaceSingle(dims; metric=:chebyshev);
        properties,
        (model_step!)=model_step!,
        rng=MersenneTwister(seed),
        container=Vector
    )

    # Initialize cells only on the bottom layer with random strain types
    if !isnothing(initial_cell_density)
        midpoint = (div(gridwidth, 2), div(gridwidth, 2))
        initial_pop_radius = div(gridwidth, 4)
        for x in 1:gridwidth
            for y in 1:gridwidth
                # only initialize in a smallish circle
                if sqrt(sum(x -> x^2, (x, y) .- midpoint)) < initial_pop_radius
                    if rand(abmrng(model)) < initial_cell_density
                        add_agent!(
                            (x, y, 1),
                            model;
                            strain_id=rand(abmrng(model), 1:length(strain_props)),
                            internal_nutrients=zeros(Float64, n_nutrients)
                        )
                    end
                end
            end
        end
    end

    return model
end
export build_model

function model_step!(model)
    # This handles nutrient diffusion, production and consumption between the life/death steps, note that this happens much faster than life/death!
    handle_nutrients!(model)

    # Finally handle cell lifecycle
    lifedeath_step!(model)
end

################################################################################
# Model evolution/steps -- the most complex bit by far
################################################################################
# NutrientHandler concrete types
include("fdiff_gradients.jl") # these are used by
include("nhandler_directdiffusion.jl")

# Life/death step logic
function lifedeath_step!(model)
    @inbounds for cell in allagents(model)
        if cell.alive
            strain = model.strain_props[cell.strain_id]

            # Handle death
            if rand(abmrng(model)) <= strain.death_rate * model.dt
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

"""
Returns true if all internal resources are beyond their alphas.
"""
function can_reproduce(cell, model)
    strain = model.strain_props[cell.strain_id]
    for n_idx in eachindex(model.nutrients)
        if strain.uptakes[n_idx]
            if cell.internal_nutrients[n_idx] < strain.alphas[n_idx]
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

################################################################################
# Cosmo stuff and particular tests/runs
################################################################################
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
        [5.4, 0.0],                # alphas: alphaL for Lys, 0 for Ade
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
        [0.0, 3.1],               # alphas: 0 for Lys, alphaA for Ade
        0.44,                     # rA: growth rate
        0.015                     # dA: death rate
    )

    return [strain1, strain2]
end
function build_cosmo1()
    build_model(
        gridwidth=100,
        dx=5.0,
        steps_per_unit_time=10,
        strain_props=cosmo_strains(),
        D=360,
        initial_cell_density=0.2
    )
end
export build_cosmo1, cosmo_strains

end
