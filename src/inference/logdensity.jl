######################
# HMM loglikelihoods #
######################

"""
$(SIGNATURES)

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hmm`, integrating over all possible state sequences.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    _, logL = forward(hmm, obs_seq, control_seq; seq_ends, error_if_not_finite=false)
    return sum(logL)
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the the joint loglikelihood of `obs_seq` and `state_seq` for `hmm`.
"""
function joint_logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Initialization
        init = initialization(hmm)
        logL += log(init[state_seq[t1]])
        # Transitions
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t + 1])
            logL += log(trans[state_seq[t], state_seq[t + 1]])
        end
        # Observations
        for t in t1:t2
            dists = obs_distributions(hmm, control_seq[t])
            logL += logdensityof(dists[state_seq[t]], obs_seq[t])
        end
    end
    return logL
end

#########################
# Emission Densities    #
#########################


# Register types with DensityInterface
DensityInterface.DensityKind(::DriftDiffusionModel) = HasDensity()
DensityInterface.DensityKind(::UniformEmission) = HasDensity()

"""
    DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a drift diffusion model given a DDMResult.
"""
function DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)
    @unpack B, v, a₀, τ, σ = model
    @unpack rt, choice = x
    
    return logdensityof(B, v, a₀, τ, σ, rt, choice)
end

function logdensityof(
    B::TB, v::TV, a₀::TA, τ::TT, σ::TS, rt::Float64, choice::Int
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real, TS<:Real}
    if rt <= 0
        return -Inf
    end

    T = promote_type(TB, TV, TA, TT, TS)
    B, v, a₀, τ, σ = T(B), T(v), T(a₀), T(τ), T(σ)

    # determine which version of the wpft to use: upper or lower boundary
    v, w = choice == 1 ? -v : v, choice == 1 ? 1 - a₀ : a₀

    # calculate the Wiener first passage time density
    density = wfpt(rt, v, B, w, τ)
    logdens = log(density)

    # check if density is Inf (i.e., log(0)) and return a very large value if so
    return isfinite(logdens) ? logdens : -1e16
end


"""
    DensityInterface.logdensityof(model::UniformEmission, x::UniformResult)

Calculate log-likelihood of uniform emission.
"""
function DensityInterface.logdensityof(model::UniformEmission, x::UniformResult)
    return logdensityof(model, x.rt)
end

function logdensityof(model::UniformEmission, rt::Float64)
    if model.a <= rt <= model.b
        width = model.b - model.a
        return width > 0 ? -log(width) : -1e16
    else
        return -Inf
    end
end

##################################
# Fallbacks for HMM #
##################################
PENALTY = -1e6 # define penalty to avoid state degeneracy

function logdensityof(model::DriftDiffusionModel, x::AbstractResult)
    return x isa DDMResult ? DensityInterface.logdensityof(model, x) : PENALTY
end

function logdensityof(model::UniformEmission, x::AbstractResult)
    return x isa UniformResult ? DensityInterface.logdensityof(model, x) : PENALTY
end