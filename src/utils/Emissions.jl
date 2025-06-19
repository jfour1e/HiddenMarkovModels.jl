export EmissionModel, AbstractResul, DriftDiffusionModel, UniformEmission, DDMResult, UniformResult, simulateDDM, rand

abstract type EmissionModel end

mutable struct DriftDiffusionModel <: EmissionModel
    B::Float64 # Boundary Separation
    v::Float64 # Drift Rate
    a₀::Float64 # Initial Accumulation: parameterized as a fraction of B
    τ::Float64 # Non-decision time
    σ::Float64 # Noise--set to 1.0 be default for identifiability
end

function DriftDiffusionModel(;
    B::Float64=5.0, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.5, # Initial Accumulation
    τ::Float64=0.0, # Non-decision time
    σ::Float64=1.0 # Noise--set to 1.0 be default for identifiability
) 
    return DriftDiffusionModel(B, v, a₀, τ, σ)
end

mutable struct UniformEmission <: EmissionModel
    a::Float64
    b::Float64 
end

abstract type AbstractResult end

"""
    DDMResult

A tuple of RT and choice. The first element is the RT, the second is the choice.
"""
struct DDMResult <: AbstractResult
    rt::Float64
    choice::Int
end

function DDMResult(;rt::Float64, choice::Int)
    return DDMResult(rt, choice)
end

"""
     UniformResult

Tuple of RT and choice for Uniform Emission 
"""
struct UniformResult <: AbstractResult
    rt::Float64
    choice::Int
end

function UniformResult(;rt::Float64, choice::Int)
    return UniformResult(rt, choice)
end

Base.eltype(::DriftDiffusionModel) = DDMResult
Base.eltype(::UniformEmission) = UniformResult

"""
Generate a single trial of a drift diffusion model
"""
function simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5, rng::AbstractRNG=Random.default_rng())
    @unpack B, v, a₀, τ, σ = model

    # initialize variables
    t = 0.0
    a = a₀ * B  # initial accumulation

    while a < B && a > 0
        # run the model forward in time
        if t < τ
            t += dt
        else
            a += v * dt + (σ * sqrt(dt) * randn(rng))
            t += dt
        end
    end

    if a >= B
        choice = 1  # upper boundary hit
    else
        choice = -1  # lower boundary hit
    end

    return DDMResult(t, choice)
end

function simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
    results = Vector{DDMResult}(undef, n)
    @threads for i in 1:n
        results[i] = simulateDDM(model, dt)
    end
    return results
end

"""
Uniform rand function 
"""
function Random.rand(rng::AbstractRNG, model::UniformEmission)
    rt = rand(rng, Uniform(model.a, model.b))      
    choice = rand(rng, Bool) ? 1 : -1                   
    return UniformResult(rt=rt, choice=choice)
end

"""
    Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)

Generate a single trial of the drift diffusion model using the Euler-Maruyama method--needed for HiddenMarkovModels.jl
"""
function Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)
    # Generate a single trial of the drift diffusion model
    return simulateDDM(model, 1e-6, rng)
end 