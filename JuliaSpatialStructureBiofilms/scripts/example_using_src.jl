using Revise
using JuliaSpatialStructureBiofilms

using Profile, ProfileView, PProf

function example()
    m = build_model(;
        gridwidth=30, dx=1.0, strain_props=cosmo_strains(), D=360.0,
        initial_cell_density=0.5, dt=0.01, nhandler=(:dd, 100)
    )
    # alternatively m - build_cosmo1()

    # then you can step as you like
    @time step!(m)

    # etc., same as before

    # profile using ProfileView (Jan S used)
    # Profile.clear()
    # @profview step!(m, 5)

    # profile using PProf (Jan K prefers)
    # Profile.clear()
    # @profile step!(m, 5)
    # pprof()
end
