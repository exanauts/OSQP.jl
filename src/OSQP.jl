module OSQP

export OSQPMathProgBaseInterface
using SparseArrays
using LinearAlgebra
using Libdl


if isfile(joinpath(dirname(@__FILE__), "..", "deps", "deps.jl"))
    include("../deps/deps.jl")
else
    error("OSQP not properly installed. Please run Pkg.build(\"OSQP\")")
end


function __init__()
    # Get version
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libcublas.so.10",RTLD_GLOBAL) 
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libcublasLt.so.10", RTLD_GLOBAL) 
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so", RTLD_GLOBAL) 
    Libdl.dlopen("/usr/local/cuda/lib64/libcusparse.so", RTLD_GLOBAL) 
    # Libdl.dlopen("libcublas.so.10", RTLD_GLOBAL) 
    # Libdl.dlopen("libcublasLt.so.10", RTLD_GLOBAL) 
    # Libdl.dlopen("libcuda.so", RTLD_GLOBAL) 
    # Libdl.dlopen("libcusparse.so", RTLD_GLOBAL) 
    ver_array = split(version(), ".")
    ver_string = string(ver_array[1], ".", ver_array[2], ".", ver_array[3])  # Get string without dev vers
    vnum = VersionNumber(ver_string)


    depsdir = realpath(joinpath(dirname(@__FILE__), "..", "deps"))
    if (vnum.major != 0 && vnum.minor != 6)
        error("Current OSQP version installed is $(osqp_version()), but we require version 0.6.*. Delete the contents of the `$depsdir` directory except for the files `build.jl` and `.gitignore`, then rerun Pkg.build(\"OSQP\").")
    end
end

include("constants.jl")
include("types.jl")
include("interface.jl")
include("MPB_wrapper.jl")
include("MOI_wrapper.jl")
const Optimizer = MathOptInterfaceOSQP.Optimizer

end # module
