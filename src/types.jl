# Types defined in types.h
# https://github.com/oxfordcontrol/osqp/blob/master/include/types.h

# Integer type from C
#if Sys.WORD_SIZE == 64   # 64bit system
    # const Cc_int = Clonglong
#else  # 32bit system
    const Cc_int = Cint
#end

struct Ccsc
    m::Cc_int
    n::Cc_int
    p::Ptr{Cc_int}
    i::Ptr{Cc_int}
    x::Ptr{Cdouble}
    nzmax::Cc_int
    nz::Cc_int
end


struct ManagedCcsc
    m::Cc_int
    n::Cc_int
    p::Vector{Cc_int}
    i::Vector{Cc_int}
    x::Vector{Cdouble}
    nzmax::Cc_int
    nz::Cc_int

end

# Construct ManagedCcsc matrix from SparseMatrixCSC
function ManagedCcsc(M::SparseMatrixCSC)

    # Get dimensions
    m = M.m
    n = M.n

    # Get vectors of data, rows indices and column pointers
    x = convert(Array{Float64,1}, M.nzval)
    # C is 0 indexed
    i = convert(Array{Cc_int,1}, M.rowval .- 1)
    # C is 0 indexed
    p = convert(Array{Cc_int,1}, M.colptr .- 1)

    # Create new ManagedCcsc matrix
    ManagedCcsc(m, n, p, i, x, length(M.nzval), -1)
end

function Base.convert(::Type{SparseMatrixCSC}, c::OSQP.Ccsc)
  m = c.m
  n = c.n
  nzmax = c.nzmax
  nzval = [unsafe_load(c.x, i) for i=1:nzmax]
  rowval = [unsafe_load(c.i, i) for i=1:nzmax] .+ 1
  colptr = [unsafe_load(c.p, i) for i=1:(n+1)] .+ 1
  SparseMatrixCSC{Float64, Cc_int}(m, n, colptr, rowval, nzval)
end

# Returns an Ccsc matrix. The vectors are *not* GC tracked in the struct.
# Use this only when you know that the managed matrix will outlive the Ccsc
# matrix.
Ccsc(m::ManagedCcsc) =
    Ccsc(m.m, m.n, pointer(m.p), pointer(m.i), pointer(m.x), m.nzmax, m.nz)


struct Solution
    x::Ptr{Cdouble}
    y::Ptr{Cdouble}
    prim_inf_cert::Ptr{Cdouble}
    dual_inf_cert::Ptr{Cdouble}
end

# Internal C type for info
# N.B. This is not the one returned to the user!
struct CInfo
    iter::Cc_int
    # We need to allocate 32 bytes for a character string, so we allocate 256 bits
    # of integer instead
    # TODO: Find a better way to do this
    status::NTuple{32,Cchar}
    status_val::Cc_int
    rho_updates::Cc_int
    rho_estimate::Cdouble
    status_polish::Cc_int
    obj_val::Cdouble
    pri_res::Cdouble
    dua_res::Cdouble
    setup_time::Cdouble
    solve_time::Cdouble
    update_time::Cdouble
    polish_time::Cdouble
    run_time::Cdouble
end

struct Settings
    rho::Cdouble
    rho_is_vec::Cc_int
    sigma::Cdouble
    scaling::Cc_int
    max_iter::Cc_int
    eps_abs::Cdouble
    eps_rel::Cdouble
    eps_prim_inf::Cdouble
    eps_dual_inf::Cdouble
    alpha::Cdouble
    scaled_termination::Cc_int
    check_termination::Cc_int
    warm_start::Cc_int
    linsys_solver::Cint  # Enum type
    adaptive_rho::Cc_int
    adaptive_rho_interval::Cc_int
    adaptive_rho_tolerance::Cdouble
    adaptive_rho_fraction::Cdouble
    delta::Cdouble
    polish::Cc_int
    polish_refine_iter::Cc_int
    verbose::Cc_int
    time_limit::Cdouble
end

function Settings()
    s = Ref{OSQP.Settings}()
    ccall((:osqp_set_default_settings, OSQP.osqp), Nothing,
          (Ref{OSQP.Settings},), s)
    return s[]
end

function Settings(settings_dict::Dict{Symbol,Any})
#  function Settings(settings::Base.Iterators.IndexValue)
#  function Settings(settings::Array{Any, 1})
    default_settings = OSQP.Settings()


       # Convert linsys_solver string to number
    # linsys_solver_str_to_int!(settings_dict)

    # Get list with elements of default and user settings
    # If setting is in the passed settings (settings_dict),
    # then convert type to the right type. Otherwise just take
    # the default one

    # These default settings are form the non-cuda version of OSQP
    # settings_dict[:check_termination] = 25
    # settings_dict[:adaptive_rho_interval] = 0
    # settings_dict[:adaptive_rho_tolerance] = 5.0
    settings_list = [setting in keys(settings_dict) ?
             convert(fieldtype(typeof(default_settings), setting), settings_dict[setting]) :
             getfield(default_settings, setting)
             for setting in fieldnames(typeof(default_settings))]

    # Create new settings with new dictionary
    s = OSQP.Settings(settings_list...)
    return s

end

mutable struct Info
    iter::Int64
    status::Symbol
    status_val::Int64
    rho_updates::Int64
    rho_estimate::Float64
    status_polish::Int64
    obj_val::Float64
    pri_res::Float64
    dua_res::Float64
    setup_time::Float64
    solve_time::Float64
    update_time::Float64
    polish_time::Float64
    run_time::Float64

    Info() = new()
end

mutable struct Solver
  settings::Ptr{Settings}  
  solution::Ptr{Solution}
  info::Ptr{CInfo}    
  work::Ptr{Nothing}
end

function copyto!(info::Info, cinfo::CInfo)
    info.iter = cinfo.iter
    info.status = OSQP.status_map[cinfo.status_val]
    info.status_val = cinfo.status_val
    info.status_polish = cinfo.status_polish
    info.obj_val = cinfo.obj_val
    info.pri_res = cinfo.pri_res
    info.dua_res = cinfo.dua_res
    info.setup_time = cinfo.setup_time
    info.solve_time = cinfo.solve_time
    info.update_time = cinfo.update_time
    info.polish_time = cinfo.polish_time
    info.run_time = cinfo.run_time
    info.rho_updates = cinfo.rho_updates
    info.rho_estimate = cinfo.rho_estimate
    info
end

mutable struct Results
    x::Vector{Float64}
    y::Vector{Float64}
    info::OSQP.Info
    prim_inf_cert::Vector{Float64}
    dual_inf_cert::Vector{Float64}
end

Results() = Results(Float64[], Float64[], Info(), Float64[], Float64[])

function Base.resize!(results::Results, n::Int, m::Int)
    resize!(results.x, n)
    resize!(results.y, m)
    resize!(results.prim_inf_cert, m)
    resize!(results.dual_inf_cert, n)
    results
end
