using SpaceCharge
using CUDA: CuArray

"""
    SpaceChargeParams{M}

A struct containing space charge parameters including a mesh object.
The mesh can be either CPU or GPU based, with types automatically inferred.

# Fields
- `mesh::Union{M, Nothing}`: The 3D mesh object (can be CPU or GPU based), or nothing if not yet created
- `backend::Symbol`: The backend being used (:cpu or :gpu)
- `grid_size::NTuple{3, Int}`: The grid dimensions for the mesh
- `total_charge::Float64`: The total charge in the bunch (in Coulombs)

# Type Parameters
- `M`: The concrete mesh type
"""
@kwdef mutable struct SpaceChargeParams{M} <: AbstractParams
  mesh::Union{M, Nothing}
  backend::Symbol
  grid_size::NTuple{3, Int}
  total_charge::Float64
end

"""
    SpaceChargeParams(backend::Symbol, grid_size::NTuple{3, Int}; T::Type{<:AbstractFloat}=Float64)

Construct a SpaceChargeParams object with the specified backend and grid size.
Creates a mesh with placeholder bounds that will be determined during deposition.

# Arguments
- `backend::Symbol`: Either `:cpu` or `:gpu` to specify the computation backend
- `grid_size::NTuple{3, Int}`: Number of grid points in each dimension (nx, ny, nz)

# Keyword Arguments
- `T`: Floating-point type for calculations

# Returns
- A `SpaceChargeParams` object with an initialized mesh using placeholder bounds
- The mesh bounds will be automatically updated during particle deposition
"""
function SpaceChargeParams(
  backend::Symbol,
  grid_size::NTuple{3, Int};
  T::Type{<:AbstractFloat}=Float64,
  total_charge::Float64=0.0
)
  if backend == :gpu
    array_type = CuArray
  elseif backend == :cpu
    array_type = Array
  else
    error("Unknown backend: $backend")
  end

  mesh = Mesh3D(
    grid_size;
    T=T,
    array_type=array_type
  )

  return SpaceChargeParams(mesh=mesh, backend=backend, grid_size=grid_size, total_charge=total_charge)
end

# Convenience constructors
"""
  SpaceChargeParams(; kwargs...)

Default constructor that creates CPU-based space charge parameters with default grid size.
"""
#SpaceChargeParams(; kwargs...) = SpaceChargeParams(:cpu, (32, 32, 32); kwargs...)

"""
  SpaceChargeParams(grid_size; kwargs...)

Constructor with specified grid size, defaulting to CPU backend.
"""
SpaceChargeParams(grid_size::NTuple{3, Int}; kwargs...) = SpaceChargeParams(:cpu, grid_size; kwargs...)
