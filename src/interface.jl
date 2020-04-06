# Wrapper for the low level functions defined in https://github.com/oxfordcontrol/osqp/blob/master/include/osqp.h

# Ensure compatibility between Julia versions with @gc_preserve
@static if isdefined(Base, :GC)
    import Base.GC: @preserve
else
    macro preserve(args...)
        body = args[end]
        esc(body)
    end
end

"""
    Model()

Initialize OSQP model
"""
mutable struct Model
    solver::Ptr{Solver}
    lcache::Vector{Float64} # to facilitate converting l to use OSQP_INFTY
    ucache::Vector{Float64} # to facilitate converting u to use OSQP_INFTY
    m::Int
    n::Int

    function Model()
        model = new(C_NULL, Float64[], Float64[])
        finalizer(OSQP.clean!, model)
        return model

    end


end

"""
    setup!(model, P, q, A, l, u, settings)

Perform OSQP solver setup of model `model`, using the inputs `P`, `q`, `A`, `l`, `u`.
"""
function setup!(model::OSQP.Model;
        P::Union{SparseMatrixCSC,Nothing} = nothing,
        q::Union{Vector{Float64},Nothing} = nothing,
        A::Union{SparseMatrixCSC,Nothing} = nothing,
        l::Union{Vector{Float64},Nothing} = nothing,
        u::Union{Vector{Float64},Nothing} = nothing,
        settings...)

    # Check problem dimensions
    if P == nothing
        if q != nothing
            n = length(q)
        elseif A != nothing
            n = size(A, 2)
        else
            error("The problem does not have any variables!")
        end

    else
        n = size(P, 1)
    end

    if A == nothing
        m = 0
    else
        m = size(A, 1)
    end
    model.m = m
    model.n = n


    # Check if parameters are nothing
    if ((A == nothing) & ( (l != nothing) | (u != nothing))) |
        ((A != nothing) & ((l == nothing) & (u == nothing)))
        error("A must be supplied together with l and u")
    end

    if (A != nothing) & (l == nothing)
        l = -Inf * ones(m)
    end
    if (A != nothing) & (u == nothing)
        u = Inf * ones(m)
    end

    if P == nothing
        P = sparse([], [], [], n, n)
    end
    if q == nothing
        q = zeros(n)
    end
    if A == nothing
        A = sparse([], [], [], m, n)
        l = zeros(m)
        u = zeros(m)
    end


    # Check if dimensions are correct
    if length(q) != n
        error("Incorrect dimension of q")
    end
    if length(l) != m
        error("Incorrect dimensions of l")
    end
    if length(u) != m
        error("Incorrect dimensions of u")
    end


    # Check or sparsify matrices
    if !issparse(P)
        @warn("P is not sparse. Sparsifying it now (it might take a while)")
        P = sparse(P)
    end
    if !issparse(A)
        @warn("A is not sparse. Sparsifying it now (it might take a while)")
        A = sparse(A)
    end

    # Constructing upper triangular from P
    if !istriu(P)
        P = triu(P)
    end

    # Convert lower and upper bounds from Julia infinity to OSQP infinity
    u = min.(u, OSQP_INFTY)
    l = max.(l, -OSQP_INFTY)

    # Resize caches
    resize!(model.lcache, m)
    resize!(model.ucache, m)

    # Create managed matrices to avoid segfaults (See SCS.jl)
    managedP = OSQP.ManagedCcsc(P)
    managedA = OSQP.ManagedCcsc(A)

    # Get managed pointers (Ref) Pdata and Adata
    Pdata = Ref(OSQP.Ccsc(managedP))
    Adata = Ref(OSQP.Ccsc(managedA))

    # Create OSQP settings
    settings_dict = Dict{Symbol,Any}()
    if !isempty(settings)
        for (key, value) in settings
            settings_dict[key] = value
        end
    end

    stgs = OSQP.Settings(settings_dict)

    @preserve managedP Pdata managedA Adata q l u begin
        # Perform setup
    solver = Ref{Ptr{Solver}}()
        exitflag = ccall((:osqp_setup, OSQP.osqp), Cc_int,
                     (Ptr{Ptr{Solver}}, Ptr{OSQP.Ccsc}, Ptr{Cdouble}, Ptr{OSQP.Ccsc}, Ptr{Cdouble}, Ptr{Cdouble}, 
                     Cc_int, Cc_int, Ptr{OSQP.Settings}),
                     solver, Base.unsafe_convert(Ptr{OSQP.Ccsc}, Pdata), pointer(q), Base.unsafe_convert(Ptr{OSQP.Ccsc}, Adata), pointer(l), pointer(u), 
                     m, n, Ref(stgs))
    model.solver = solver[]
    end

    if exitflag != 0
        error("Error in OSQP setup")
    end

end


function solve!(model::OSQP.Model, results::Results = Results())
    ccall((:osqp_solve, OSQP.osqp), Cc_int,
             (Ptr{Solver}, ), model.solver)
    solver = unsafe_load(model.solver)
    info = results.info
    copyto!(info, unsafe_load(solver.info))
    solution = unsafe_load(solver.solution)
    n = model.n
    m = model.m
    resize!(results, n, m)
    has_solution = false
    for status in SOLUTION_PRESENT
        info.status == status && (has_solution = true; break)
    end
    if has_solution
        # If solution exists, copy it
        unsafe_copyto!(pointer(results.x), solution.x, n)
        unsafe_copyto!(pointer(results.y), solution.y, m)
        fill!(results.prim_inf_cert, NaN)
        fill!(results.dual_inf_cert, NaN)
    else
        # else fill with NaN and return certificates of infeasibility
        fill!(results.x, NaN)
        fill!(results.y, NaN)
        if info.status == :Primal_infeasible || info.status == :Primal_infeasible_inaccurate
            unsafe_copyto!(pointer(results.prim_inf_cert), solution.prim_inf_cert, m)
            fill!(results.dual_inf_cert, NaN)
        elseif info.status == :Dual_infeasible || info.status == :Dual_infeasible_inaccurate
            fill!(results.prim_inf_cert, NaN)
            unsafe_copyto!(pointer(results.dual_inf_cert), solution.dual_inf_cert, n)
        else
            fill!(results.prim_inf_cert, NaN)
            fill!(results.dual_inf_cert, NaN)
        end
    end

    if info.status == :Non_convex
        info.obj_val = NaN
    end

    results
end


function version()
    return unsafe_string(ccall((:osqp_version, OSQP.osqp), Cstring, ()))
end

function clean!(model::OSQP.Model)
    exitflag = ccall((:osqp_cleanup, OSQP.osqp), Cc_int,
             (Ptr{Solver},), model.solver)
    if exitflag != 0
        error("Error in OSQP cleanup")
    end
end

function update_q!(model::OSQP.Model, q::Vector{Float64})
    (n, m) = OSQP.dimensions(model)
    if length(q) != n
        error("q must have length n = $(n)")
    end
    exitflag = ccall((:osqp_update_lin_cost, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}), model.solver, q)
    if exitflag != 0 error("Error updating q") end
end

function update_bounds!(model::OSQP.Model, l::Vector{Float64}, u::Vector{Float64})
    (n, m) = OSQP.dimensions(model)
    if length(l) != m
        error("l must have length m = $(m)")
    end
    if length(u) != m
        error("u must have length m = $(m)")
    end
    model.lcache .= max.(l, -OSQP_INFTY)
    model.ucache .= min.(u, OSQP_INFTY)
    exitflag = ccall((:osqp_update_bounds, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}, Ptr{Cdouble}),
        model.solver, model.lcache, model.ucache)
    if exitflag != 0 error("Error updating bounds l and u") end
end

prep_idx_vector_for_ccall(idx::Nothing, n::Int, namesym::Symbol) = C_NULL
function prep_idx_vector_for_ccall(idx::Vector{Int}, n::Int, namesym::Symbol)
    if length(idx) != n
        error("$(namesym) and $(namesym)_idx must have the same length")
    end
    idx .-= 1 # Shift indexing to match C
    idx
end

restore_idx_vector_after_ccall!(idx::Nothing) = nothing
function restore_idx_vector_after_ccall!(idx::Vector{Int})
    idx .+= 1 # Unshift indexing
    nothing
end

function update_P!(model::OSQP.Model, Px::Vector{Float64}, Px_idx::Union{Vector{Int}, Nothing})
    Px_idx_prepped = prep_idx_vector_for_ccall(Px_idx, length(Px), :P)
    exitflag = ccall((:osqp_update_P, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}, Ptr{Cc_int}, Cc_int),
        model.solver, Px, Px_idx_prepped, length(Px))
    restore_idx_vector_after_ccall!(Px_idx)
    if exitflag != 0 error("Error updating P") end
end

function update_A!(model::OSQP.Model, Ax::Vector{Float64}, Ax_idx::Union{Vector{Int}, Nothing})
    Ax_idx_prepped = prep_idx_vector_for_ccall(Ax_idx, length(Ax), :A)
    exitflag = ccall((:osqp_update_A, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}, Ptr{Cc_int}, Cc_int),
        model.solver, Ax, Ax_idx_prepped, length(Ax))
    restore_idx_vector_after_ccall!(Ax_idx)
    if exitflag != 0 error("Error updating A") end
end

function update_P_A!(model::OSQP.Model, Px::Vector{Float64}, Px_idx::Union{Vector{Int}, Nothing}, Ax::Vector{Float64}, Ax_idx::Union{Vector{Int}, Nothing})
    Px_idx_prepped = prep_idx_vector_for_ccall(Px_idx, length(Px), :P)
    Ax_idx_prepped = prep_idx_vector_for_ccall(Ax_idx, length(Ax), :A)
    exitflag = ccall((:osqp_update_P_A, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble},
        Ptr{Cc_int}, Cc_int, Ptr{Cdouble}, Ptr{Cc_int}, Cc_int),
        model.solver, Px, Px_idx_prepped, length(Px), Ax, Ax_idx_prepped, length(Ax))
    restore_idx_vector_after_ccall!(Ax_idx)
    restore_idx_vector_after_ccall!(Px_idx)
    if exitflag != 0 error("Error updating P and A") end
end

function update!(model::OSQP.Model; q = nothing, l = nothing, u = nothing, Px = nothing, Px_idx = nothing, Ax = nothing, Ax_idx = nothing)
    # q
    if q != nothing
        update_q!(model, q)
    end

    # l and u
    if l != nothing && u != nothing
        update_bounds!(model, l, u)
    end

    # P and A
    if Px != nothing && Ax != nothing
        update_P_A!(model, Px, Px_idx, Ax, Ax_idx)
    elseif Px != nothing
        update_P!(model, Px, Px_idx)
    elseif Ax != nothing
        update_A!(model, Ax, Ax_idx)
    end
end



function update_settings!(model::OSQP.Model; kwargs...)

    if isempty(kwargs)
        return
    else
        data = Dict{Symbol,Any}()
        for (key, value) in kwargs
            if !(key in UPDATABLE_SETTINGS)
                error("$(key) cannot be updated or is not recognized")
            else
                data[key] = value
            end
        end
    end

    # Get arguments
    max_iter = get(data, :max_iter, nothing)
    eps_abs = get(data, :eps_abs, nothing)
    eps_rel = get(data, :eps_rel, nothing)
    eps_prim_inf = get(data, :eps_prim_inf, nothing)
    eps_dual_inf = get(data, :eps_dual_inf, nothing)
    rho = get(data, :rho, nothing)
    alpha = get(data, :alpha, nothing)
    delta = get(data, :delta, nothing)
    polish = get(data, :polish, nothing)
    polish_refine_iter = get(data, :polish_refine_iter, nothing)
    verbose = get(data, :verbose, nothing)
    scaled_termination = get(data, :early_terminate, nothing)
    check_termination = get(data, :check_termination, nothing)
    warm_start = get(data, :warm_start, nothing)
    time_limit = get(data, :time_limit, nothing)

    # Update individual settings
    if max_iter != nothing
        exitflag = ccall((:osqp_update_max_iter, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, max_iter)
        if exitflag != 0 error("Error updating max_iter") end
    end

    if eps_abs != nothing
        exitflag = ccall((:osqp_update_eps_abs, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, eps_abs)
        if exitflag != 0 error("Error updating eps_abs") end
    end

    if eps_rel != nothing
        exitflag = ccall((:osqp_update_eps_rel, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, eps_rel)
        if exitflag != 0 error("Error updating eps_rel") end
    end


    if eps_prim_inf != nothing
        exitflag = ccall((:osqp_update_eps_prim_inf, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, eps_prim_inf)
        if exitflag != 0 error("Error updating eps_prim_inf") end
    end

    if eps_dual_inf != nothing
        exitflag = ccall((:osqp_update_eps_dual_inf, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, eps_dual_inf)
        if exitflag != 0 error("Error updating eps_dual_inf") end
    end

    if rho != nothing
        exitflag = ccall((:osqp_update_rho, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, rho)
        if exitflag != 0 error("Error updating rho") end
    end

    if alpha != nothing
        exitflag = ccall((:osqp_update_alpha, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, alpha)
        if exitflag != 0 error("Error updating alpha") end
    end

    if delta != nothing
        exitflag = ccall((:osqp_update_delta, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, delta)
        if exitflag != 0 error("Error updating delta") end
    end

    if polish != nothing
        exitflag = ccall((:osqp_update_polish, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, polish)
        if exitflag != 0 error("Error updating polish") end
    end

    if polish_refine_iter != nothing
        exitflag = ccall((:osqp_update_polish_refine_iter, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, polish_refine_iter)
        if exitflag != 0 error("Error updating polish_refine_iter") end
    end

    if verbose != nothing
        exitflag = ccall((:osqp_update_verbose, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, verbose)
        if exitflag != 0 error("Error updating verbose") end
    end

    if scaled_termination != nothing
        exitflag = ccall((:osqp_update_scaled_termination, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, scaled_termination)
        if exitflag != 0 error("Error updating scaled_termination") end
    end

    if check_termination != nothing
        exitflag = ccall((:osqp_update_check_termination, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, check_termination)
        if exitflag != 0 error("Error updating check_termination") end
    end

    if warm_start != nothing
        exitflag = ccall((:osqp_update_warm_start, OSQP.osqp), Cc_int, (Ptr{Solver}, Cc_int), model.solver, warm_start)
        if exitflag != 0 error("Error updating warm_start") end
    end

   if time_limit != nothing
        exitflag = ccall((:osqp_update_time_limit, OSQP.osqp), Cc_int, (Ptr{Solver}, Cdouble), model.solver, time_limit)
        if exitflag != 0 error("Error updating time_limit") end
    end

    return nothing
end

function warm_start_x!(model::OSQP.Model, x::Vector{Float64})
    (n, m) = OSQP.dimensions(model)
    length(x) == n || error("Wrong dimension for variable x")
    exitflag = ccall((:osqp_warm_start_x, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}), model.solver, x)
    exitflag == 0  || error("Error in warm starting x")
    nothing
end

function warm_start_y!(model::OSQP.Model, y::Vector{Float64})
    (n, m) = OSQP.dimensions(model)
    length(y) == m || error("Wrong dimension for variable y")
    exitflag = ccall((:osqp_warm_start_y, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}), model.solver, y)
    exitflag == 0 || error("Error in warm starting y")
    nothing
end

function warm_start_x_y!(model::OSQP.Model, x::Vector{Float64}, y::Vector{Float64})
    (n, m) = OSQP.dimensions(model)
    length(x) == n || error("Wrong dimension for variable x")
    length(y) == m || error("Wrong dimension for variable y")
    exitflag = ccall((:osqp_warm_start, OSQP.osqp), Cc_int, (Ptr{Solver}, Ptr{Cdouble}, Ptr{Cdouble}), model.solver, x, y)
    exitflag == 0 || error("Error in warm starting x and y")
    nothing
end


function warm_start!(model::OSQP.Model; x::Union{Vector{Float64}, Nothing} = nothing, y::Union{Vector{Float64}, Nothing} = nothing)
    if x isa Vector{Float64} && y isa Vector{Float64}
        warm_start_x_y!(model, x, y)
    elseif x isa Vector{Float64}
        warm_start_x!(model, x)
    elseif y isa Vector{Float64}
        warm_start_y!(model, y)
    end
end



# Auxiliary low-level functions
"""
    dimensions(model::OSQP.Model)

Obtain problem dimensions from OSQP model
"""
function dimensions(model::OSQP.Model)
    return model.n, model.m
end
