using Agents
using StaticArrays
using PyFormattedStrings
using GLMakie

using Dates
using Random
using Base.Threads

################################################################################
# Define the main and somewhat general building blocks
################################################################################
@agent struct Cell{M}(GridAgent{3}) # TODO: Would be nice to add <:Integer somehow
    type::Int
    sizes::SVector{M,Float64}
    willbud::Bool
    dead::Bool
end

mutable struct ModelProperties{N,M} # N -> num cell types, M -> num resources
    # the finite difference spacings
    dt::Float64
    dx::Float64

    # physics/sim parameters
    dependencies::SArray{Tuple{N,M,2},Bool}
    type_deathrates::SVector{N,Float64}
    type_reprates::SVector{N,Float64}

    resource_vmaxs::SVector{M,Float64}
    resource_kms::SVector{M,Float64}
    resource_alphas::SVector{M,Float64}
    resource_gammas::SVector{M,Float64}
    resource_Ds::SVector{M,Float64} # the diffusion parameters for each resource

    # Utilities for the coarse-grained resource grid
    resource_grid_factor::Int # specifies how many space grid cells make one resource one
    resource_spacesize::Tuple{Int,Int,Int} # the resource grid dimensions
    resources::SVector{M,Array{Float64,3}} # matrices of available resources

    lifedeathtimer::Int
    lifedeathoccurence::Int

    # internals/cache for performance etc.
    res_grid_dx::Float64
    resources_temp::SVector{M,Array{Float64,3}} # just a preallocated temp var
    resource_betas::SVector{M,Float64} # master equation diffusion coeffs
end
getNM(_::ModelProperties{N,M}) where {N,M} = (N, M)
getNM(m::AgentBasedModel) = getNM(abmproperties(m))
function make_model(space_size, dt, dx, step!,
    dependencies, type_deathrates, type_reprates,
    resource_vmaxs, resource_kms, resource_alphas, resource_gammas, resource_Ds,
    resource_grid_factor, lifedeathoccurence, resources=nothing; # by default resources start at 0.
)
    dims = (space_size, space_size, space_size)
    space = GridSpace(dims; periodic=true, metric=:chebyshev)

    numtypes = length(type_deathrates) # just used the first lists here, it is
    numresources = length(resource_vmaxs) # (type) checked later that they all agree

    if !all(x -> x == 0, mod.(spacesize(space), resource_grid_factor))
        throw(ArgumentError("space dimensions are not all divisible by resource_grid_factor"))
    end
    res_spacesize = div.(spacesize(space), resource_grid_factor)
    res_grid_dx = resource_grid_factor * dx
    betas = resource_Ds ./ ((res_grid_dx^2) / dt) # converted diff param

    resources_ = if isnothing(resources) # by default all 0.
        [fill(0.0, res_spacesize) for _ in 1:length(resource_vmaxs)]
    else # otherwise be smart, for each element passed
        map(resources) do res
            if isa(res, Array) # if its already an array, check dimensions and use it
                if size(res) == res_spacesize
                    res
                else
                    throw(ArgumentError(f"passed resource matrix {res} does not have the correct shape"))
                end
            elseif isa(res, Number) # if its a number make it that number everywhere
                fill(res, res_spacesize)
            else # otherwise fail
                throw(ArgumentError(f"cannot interpret {res} as a resource"))
            end
        end
    end

    properties = ModelProperties{numtypes,numresources}(dt, dx,
        # physical/sim params
        dependencies, type_deathrates, type_reprates,
        resource_vmaxs, resource_kms, resource_alphas, resource_gammas, resource_Ds,
        # system stuff
        resource_grid_factor, res_spacesize, resources_,
        1, lifedeathoccurence,
        res_grid_dx, map(similar, resources_), betas
    )

    StandardABM(Cell{numresources}, space;
        (model_step!)=step!,
        properties
    )
end

"""
For a given agent space grid position returns the coarser resource grid position
"""
real_to_res_pos(rgf::Integer, pos) = div.(pos, rgf, RoundUp)
real_to_res_pos(model, pos) = real_to_res_pos(model.resource_grid_factor, pos)

################################################################################
# Diffusion algorithms (may reflect BCs)
################################################################################
"""
Will diffuse `resources` (and inplace modify) given the passed params, `resources_temp` is used as a temp buffer, the values after should be considered
random. Uses fixed BCs everywhere. TODO: This is easily parallelizeable!
"""
function diffuse_allboundaries!(resources, resources_temp, betas, gridsize)
    for (res, temp, beta) in zip(resources, resources_temp, betas)
        @threads for x in 1:gridsize[1]
            for y in 1:gridsize[2]
                for z in 1:gridsize[3]
                    temp[x, y, z] = res[x, y, z]
                    if x != 1
                        temp[x, y, z] += beta * (res[x-1, y, z] - res[x, y, z])
                    end
                    if x != gridsize[1]
                        temp[x, y, z] += beta * (res[x+1, y, z] - res[x, y, z])
                    end
                    if y != 1
                        temp[x, y, z] += beta * (res[x, y-1, z] - res[x, y, z])
                    end
                    if y != gridsize[2]
                        temp[x, y, z] += beta * (res[x, y+1, z] - res[x, y, z])
                    end
                    if z != 1
                        temp[x, y, z] += beta * (res[x, y, z-1] - res[x, y, z])
                    end
                    if z != gridsize[3]
                        temp[x, y, z] += beta * (res[x, y, z+1] - res[x, y, z])
                    end
                    # TODO: Warning that may be worth removing for performance
                    if temp[x, y, z] < 0.0
                        @error f"Getting a negative resource value of {temp[x,y,z]}! This is very likely a finite difference issue from dt being too big or dx too small"
                    end
                end
            end
        end
        res .= temp
    end
end
"""
As above but x and y directions are considered periodic.
"""
function diffuse_xyperiodic!(resources, resources_temp, betas, gridsize)
    for (res, temp, beta) in zip(resources, resources_temp, betas)
        @threads for x in 1:gridsize[1]
            for y in 1:gridsize[2]
                for z in 1:gridsize[3]
                    temp[x, y, z] = res[x, y, z]
                    if x != 1
                        temp[x, y, z] += beta * (res[x-1, y, z] - res[x, y, z])
                    else
                        temp[x, y, z] += beta * (res[gridsize[1], y, z] - res[x, y, z])
                    end
                    if x != gridsize[1]
                        temp[x, y, z] += beta * (res[x+1, y, z] - res[x, y, z])
                    else
                        temp[x, y, z] += beta * (res[1, y, z] - res[x, y, z])
                    end
                    if y != 1
                        temp[x, y, z] += beta * (res[x, y-1, z] - res[x, y, z])
                    else
                        temp[x, y, z] += beta * (res[x, gridsize[2], z] - res[x, y, z])
                    end
                    if y != gridsize[2]
                        temp[x, y, z] += beta * (res[x, y+1, z] - res[x, y, z])
                    else
                        temp[x, y, z] += beta * (res[x, 1, z] - res[x, y, z])
                    end
                    if z != 1
                        temp[x, y, z] += beta * (res[x, y, z-1] - res[x, y, z])
                    end
                    if z != gridsize[3]
                        temp[x, y, z] += beta * (res[x, y, z+1] - res[x, y, z])
                    end
                    # TODO: Warning that may be worth removing for performance
                    if temp[x, y, z] < 0.0
                        @error f"Getting a negative resource value of {temp[x,y,z]}! This is very likely a finite difference issue from dt being too big or dx too small"
                    end
                end
            end
        end
        res .= temp
    end
end

################################################################################
# The evolution steps
################################################################################
const inplane_offsets = [(0, -1, 0), (-1, -1, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, -1, 0)]
const offplane_offsets = [(0, 0, 1), (0, -1, 1), (-1, -1, 1), (-1, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, -1, 1)]

function step1!(model)
    # this modifies resources
    diffuse_xyperiodic!(model.resources, model.resources_temp, model.resource_betas, model.resource_spacesize)

    deps = model.dependencies

    numtypes, numresources = getNM(model)

    # do uptake, release
    for nn in 1:numresources
        for cell in allagents(model)
            if !cell.dead
                cell_res_pos = real_to_res_pos(model, cell.pos)
                if deps[cell.type, nn, 1]
                    uptake = (model.resource_vmaxs[nn] * model.resources[nn][cell_res_pos...] * model.dt) / (model.resources[nn][cell_res_pos...] + model.resource_kms[nn] * model.dt)
                    cell.sizes = setindex(cell.sizes, cell.sizes[nn] + uptake, nn)
                    model.resources[nn][cell_res_pos...] -= uptake
                end
                if deps[cell.type, nn, 2]
                    release = model.resource_gammas[nn] * model.dt
                    model.resources[nn][cell_res_pos...] += release
                end
            end
        end
    end

    if model.lifedeathtimer == model.lifedeathoccurence
        model.lifedeathtimer = 1
        # flag life/death
        for cell in allagents(model)
            if !cell.dead
                if rand(abmrng(model)) < (model.type_deathrates[cell.type] * model.dt)
                    cell.dead = true
                else
                    cell.willbud = true
                    for nn in 1:numresources
                        if deps[cell.type, nn, 1] && cell.sizes[nn] < model.resource_alphas[nn]
                            cell.willbud = false
                            break
                        end
                    end
                end
            end
        end

        # do budding
        for cell in allagents(model)
            if cell.willbud
                new_pos = nothing
                # look within plane
                for offset in shuffle(inplane_offsets)
                    pos = normalize_position(cell.pos .+ offset, model)
                    if isempty(pos, model)
                        new_pos = pos
                        break
                    end
                end
                # look in the place above if not the last one
                if cell.pos[3] != spacesize(model)[3]
                    for offset in shuffle(offplane_offsets)
                        pos = normalize_position(cell.pos .+ offset, model)
                        if isempty(pos, model)
                            new_pos = pos
                            break
                        end
                    end
                end
                if !isnothing(new_pos)
                    cell.sizes = @SVector fill(0.0, length(cell.sizes))
                    replicate!(cell, model; pos=new_pos)
                else
                    cell.dead = true
                end
            end
        end
    else
        model.lifedeathtimer += 1
    end

    # TODO: should at some point remove the dead cells too
end

################################################################################
# Some running examples
################################################################################
#  For reference
# function make_model(space_size, dt, dx, step!,
#     dependencies, type_deathrates, type_reprates,
#     resource_vmaxs, resource_kms, resource_alphas, resource_gammas, resource_Ds,
#     resource_grid_factor, resources=nothing; # by default resources start at 0.
# )

function step_time!(model, deltat)
    step!(model, floor(deltat / model.dt))
end

function mmempty()
    dependencies = SArray{Tuple{0,0,2},Float64}()
    type_deathrates = SVector{0,Float64}()
    type_reprates = SVector{0,Float64}()
    resource_vmaxs = SVector{0,Float64}()
    resource_kms = SVector{0,Float64}()
    resource_alphas = SVector{0,Float64}()
    resource_gammas = SVector{0,Float64}()
    resource_Ds = SVector{0,Float64}()

    make_model(100, 0.01, 1.0, step1!,
        dependencies, type_deathrates, type_reprates,
        resource_vmaxs, resource_kms, resource_alphas, resource_gammas, resource_Ds,
        5, 1
    )
end

function mmtest2(ncells=10)
    uptakes = [true false; false true]
    releases = [false true; true false]
    dependencies = SArray{Tuple{2,2,2}}(cat(uptakes, releases; dims=3))

    type_deathrates = SA[0.021, 0.015] # /hour
    type_reprates = SA[0.51, 0.44] # /hour

    resource_alphas = SA[5.4, 3.1] # fmol
    resource_kms = SA[2.6e4, 1.3e6] # /hour
    resource_gammas = SA[1440.0, 936.0] # /hour
    resource_vmaxs = type_reprates .* resource_alphas # fmol/hour

    resource_Ds = [20.0, 20.0] .* 3600.0 # um^2/hour

    model = make_model(150, 1e-4, 5.0, step1!,
        dependencies, type_deathrates, type_reprates,
        resource_vmaxs, resource_kms, resource_alphas, resource_gammas, resource_Ds,
        10, 10^3
    )

    ncellpertype = floor(ncells / 2)
    for _ in 1:ncellpertype
        add_agent!(model, 1, fill(0.0, 2), false, false)
    end
    for _ in 1:ncellpertype
        add_agent!(model, 2, fill(0.0, 2), false, false)
    end

    model
end

function test1()
    ncellss = LinRange(0, 1000, 15)
    times = []
    for ncells in ncellss
        m = mmtest2(ncells)
        startt = Dates.now()
        step!(m, 10^4)
        deltat = Dates.now() - startt
        push!(times, deltat)
    end
    ncellss, times
end

function runmmtest()
    m = mmtest(0.001, 1.0; space_size=200, pp=50, res1_D=100.0, res2_D=100.0)
    # xx = abmexploration(m; add_controls=true, agent_color=c -> c.consumes)
    xx = abmexploration(m; add_controls=true, agent_color=c -> c.consumes,
        adata=[(c -> c.consumes == 1, count), (c -> c.consumes == 2, count)],
        mdata=[m -> sum(m.resources[1]), m -> sum(m.resources[2])],
        alabels=["#consumers of 1", "#consumers of 2"],
        mlabels=["total res 1", "total res 2"],
    )
    display(xx[1])
end

# function custom_abmexplore(m)
# end
