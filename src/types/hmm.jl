"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHMM
    "initial state probabilities"
    init::V
    "state transition probabilities"
    trans::M
    "observation distributions"
    dists::VD
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function HMM(init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector)
        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans)
        hmm = new{
            typeof(init),typeof(trans),typeof(dists),typeof(log_init),typeof(log_trans)
        }(
            init, trans, dists, log_init, log_trans
        )
        return hmm
    end
end

function Base.show(io::IO, hmm::HMM)
    return print(
        io,
        "Hidden Markov Model with:\n - initialization: $(hmm.init)\n - transition matrix: $(hmm.trans)\n - observation distributions: [$(join(hmm.dists, ", "))]",
    )
end

initialization(hmm::HMM) = hmm.init
log_initialization(hmm::HMM) = hmm.loginit
transition_matrix(hmm::HMM) = hmm.trans
log_transition_matrix(hmm::HMM) = hmm.logtrans
obs_distributions(hmm::HMM) = hmm.dists

## Fitting
function StatsAPI.fit!(
    hmm::HMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ) = fb_storage
    # Fit states
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]  # use ξ[t2] as scratch space since it is zero anyway
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]  # use ξ[t2] as scratch space since it is zero anyway
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    end
    fill!(hmm.init, zero(eltype(hmm.init)))
    fill!(hmm.trans, zero(eltype(hmm.trans)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        hmm.trans .+= ξ[t2]
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))
    # Fit observations
    for i in 1:length(hmm)
        fit_in_sequence!(hmm.dists, i, obs_seq, view(γ, i, :))
    end
    # Update logs
    hmm.loginit .= log.(hmm.init)
    mynonzeros(hmm.logtrans) .= log.(mynonzeros(hmm.trans))
    # Safety check
    @argcheck valid_hmm(hmm)
    return nothing
end

"""
    wfpt(t, v, B, z, τ, err=1e-8)

Calculate the Wiener First Passage Time (WFPT) density at time t for a drift diffusion model
with drift rate v, boundary separation B, starting point α₀, non-decision time τ, and error tolerance err.

This implementation follows the algorithm described in Navarro & Fuss (2009).
"""
function wfpt(t::TB, v::TV, B::TA, w::TT, τ::TS; err::Float64=1e-12
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real, TS<:Real}
    # Check for valid inputs (pass t = 0 for sigmoid later)
    if t < τ
        return 1e-16
    end
    
    # Use normalized time and relative start point
    tt = (t - τ) / (B^2)
    
    # Calculate number of terms needed for large t version
    if π * tt * err < 1  # if error threshold is set low enough
        kl = sqrt(-2 * log(π * tt * err) / (π^2 * tt))  # bound
        kl = max(kl, 1 / (π * sqrt(tt)))  # ensure boundary conditions met
    else  # if error threshold set too high
        kl = 1 / (π * sqrt(tt))  # set to boundary condition
    end
    
    # Calculate number of terms needed for small t version
    if 2 * sqrt(2 * π * tt) * err < 1  # if error threshold is set low enough
        ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * π * tt) * err))  # bound
        ks = max(ks, sqrt(tt) + 1)  # ensure boundary conditions are met
    else  # if error threshold was set too high
        ks = 2  # minimal kappa for that case
    end
    
    # Compute f(tt|0,1,w)
    p = 0.0  # initialize density
    if ks < kl  # if small t is better...
        K = ceil(Int, ks)  # round to smallest integer meeting error
        for k in -floor(Int, (K-1)/2):ceil(Int, (K-1)/2)  # loop over k
            p += (w + 2 * k) * exp(-((w + 2 * k)^2) / 2 / tt)  # increment sum
        end
        p /= sqrt(2 * π * tt^3)  # add constant term
    else  # if large t is better...
        K = ceil(Int, kl)  # round to smallest integer meeting error
        for k in 1:K
            p += k * exp(-(k^2) * (π^2) * tt / 2) * sin(k * π * w)  # increment sum
        end
        p *= π  # add constant term
    end
    
    # Convert to f(t|v,B,w)
    density = p * exp(-v * B * w - (v^2) * (t - τ) / 2) / (B^2)
    return max(density, 1e-12)  # ensure non-negative density (occasionaly generates neg values e.g., -1e-21) (maybe return +ϵ instead?)
end

"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::AbstractVector{<:Real}=ones(length(x)))
    @unpack B, v, a₀, τ, σ = model
    
    # Define negative log-likelihood function for optimization
    function neg_log_likelihood(params)
        # We optimize B, drift rate, and a₀ as a fraction of B
        B_temp, v_temp, τ_temp = params
        
        # Early return for invalid boundary (must be positive)
        if B_temp < 0
            return convert(typeof(B_temp), Inf)
        end
        
        # Calculate log-likelihood using the raw parameters version
        ll = 0.0
        for i in 1:length(x)
            ll += w[i] * logdensityof(B_temp, v_temp, a₀, τ_temp, σ, x[i].rt, x[i].choice)
        end
        
        # Return negative since optimizers typically minimize
        return -ll
    end
    
    # Set up optimization
    initial_params = [B, v, τ]
    
    # Add bounds - a₀_frac must be between -1 and 1
    lower_bounds = [0.001, -Inf, 1e-3]
    upper_bounds = [50.0, 10.0, 5.0]
    
    # Optimize using L-BFGS-B to respect the bounds
    result = optimize(neg_log_likelihood, lower_bounds, upper_bounds, initial_params, Fminbox(LBFGS()), autodiff=:forward)
    
    # Extract the optimized parameters
    optimal_params = Optim.minimizer(result)
    
    # Update the model with new parameter estimates
    model.B = optimal_params[1]
    model.v = optimal_params[2]
    # model.a₀ = optimal_params[3]
    model.τ = optimal_params[3]
    
    return model
end

"""
Multiple Dispatch for robustness, 
"""

function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{Any}, w::AbstractVector{<:Real})
    x_ddm = DDMResult[]
    w_ddm = Float64[]
    for (xi, wi) in zip(x, w)
        if xi isa DDMResult
            push!(x_ddm, xi)
            push!(w_ddm, wi)
        end
    end
    return fit!(model, x_ddm, w_ddm)
end

function StatsAPI.fit!(model::UniformEmission, x::Vector{UniformResult}, w::AbstractVector{<:Real})
    # Compute weighted min and max RTs
    rts = getfield.(x, :rt)
    weighted_mean = sum(w .* rts) / sum(w)
    weighted_std = sqrt(sum(w .* (rts .- weighted_mean).^2) / sum(w))
    
    # Update model bounds (safely)
    model.a = minimum(rts) - 0.05 * weighted_std
    model.b = maximum(rts) + 0.05 * weighted_std
    return model
end

function StatsAPI.fit!(model::UniformEmission, x::Vector{Any}, w::AbstractVector{<:Real})
    x_filtered = filter(xi -> xi isa UniformResult, x)
    x_cast = UniformResult[xi for xi in x_filtered]
    return fit!(model, x_cast, w[1:length(x_cast)])
end