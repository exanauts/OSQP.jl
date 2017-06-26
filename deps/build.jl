using BinDeps

@BinDeps.setup

# Add library dependency for direct method
osqp = library_dependency("osqp", aliases=["libosqpdir"])

# Current version
version = "0.1.1"

# Using latest master to debug
version = "master"
provides(Sources, URI("https://github.com/oxfordcontrol/osqp/archive/master.tar.gz"),
    [osqp], unpacked_dir="osqp-$version")

# provides(Sources, URI("https://github.com/oxfordcontrol/osqp/archive/v$version.tar.gz"),
#     [osqp], unpacked_dir="osqp-$version")

# Define directories locations deps/usr and deps/src
prefix = joinpath(BinDeps.depsdir(osqp),"usr")
srcdir = joinpath(BinDeps.depsdir(osqp),"src","osqp-$version")
blddir = joinpath(srcdir, "build")

# Define library name
libname = "libosqpdir.$(Libdl.dlext)"


provides(SimpleBuild,
    (@build_steps begin
        GetSources(osqp)
        CreateDirectory(joinpath(prefix, "lib"))
        @build_steps begin
            # ChangeDirectory(srcdir)
            CreateDirectory(blddir)
            @build_steps begin
                ChangeDirectory(blddir)
                FileRule(joinpath(prefix, "lib", libname), @build_steps begin
                    `cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..`
                    `make osqpdir`
                    `mv out/$libname $prefix/lib`
                end)
            end
        end
    end),
[osqp])




# TODO: Add proper dependencies download
# if is_apple()
#     using Homebrew
#     provides(Homebrew.HB, "osqp", osqp, os = :Darwin)
# end


# TODO: Provide sources only for Unix
# provides(Sources, URI("https://github.com/oxfordcontrol/osqp/archive/v$version.tar.gz"),
#     [osqp], os = :Unix, unpacked_dir="osqp-$version")




@BinDeps.install Dict(:osqp => :osqp)