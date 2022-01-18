using DifferentialEquations
using Plots
using DelimitedFiles
using Random

dir_data = "/Users/vinco/Work/Data/labquakesde/synthetic/"

function synthetic_eqs(dstate, state, p, t)
	x, y, z, u = state
	κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄ = p
    expx = exp(x)
    dstate[3] = -ρ*(β₂*x + z)*expx
    dstate[4] = -α -γ*u -dstate[3]
	dstate[1] = (expx*((β₁-1)*x*(1+λ*u) + y - u) + κ*(1-expx) - dstate[4]*(1+λ*y)/(1+λ*u)) / (1 + λ*u + ν*expx)
    dstate[2] = κ*(1 - expx) - ν*expx*dstate[1]
end

function σ_synthetic_eqs(dstate,state,p,t)
    κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄ = p
    dstate[1] = 0.0
    dstate[2] = ϵ₂
    dstate[3] = 0.0
    dstate[4] = ϵ₄
end

v₀ = 10e-6       # Loading velocity [m/s]
β₁ = 1.23         # b₁/a₀ [non-dim]
k  = 14.8*1e9    # Spring stiffness [Pa/m]
L₁ = 3.5*1e-6    # Critical distance for θ₁ [m]
L₂ = 30*1e-6
ρ = L₁/L₂
G = 44e9         # Rigidity modulus of quartz [Pa]
ρᵥ = 2.62e3      # Density of quartz [kg/m^3]
cₛ = sqrt(G/ρᵥ)
η₀ = G/(2*cₛ)
η = 35*η₀
a = 0.01 #0.01*13.6/17.4        # Friction direct effect [non-dim]
μ₀ = 0.64        # Static friction coefficient [non-dim]
βₐ = 7.65*1e-6 # Compressibility of air (https://encyclopedia2.thefreedictionary.com/compressibility+of+air)
βₘ = 1e-11 # Compressibility of Quartz (Pimienta et al., 2017, JGR, fig. 12)
ϕ₀ = 0.075 # Reference porosity
β = ϕ₀*(βₐ+βₘ)
ϵ = 2.55e5*β #2.34e5*β
p₀ = 1.01325e5
c₀ = (v₀/L₁)*1e-4
γ = c₀*L₁/v₀

λ = a/μ₀

σₙ₀ = Vector{Float64}(undef, 14)
σₙ₀[1] = 13.600*1e6     # Reference normal stress [Pa]
σₙ₀[2] = 14.000*1e6     # Reference normal stress [Pa]
σₙ₀[3] = 14.408*1e6     # Reference normal stress [Pa]
σₙ₀[4] = 15.001*1e6     # Reference normal stress [Pa]
σₙ₀[5] = 15.400*1e6     # Reference normal stress [Pa]
σₙ₀[6] = 16.389*1e6     # Reference normal stress [Pa]
σₙ₀[7] = 17.003*1e6     # Reference normal stress [Pa]
σₙ₀[8] = 17.379*1e6     # Reference normal stress [Pa]
σₙ₀[9] = 17.909*1e6     # Reference normal stress [Pa]
σₙ₀[10] = 20.002*1e6     # Reference normal stress [Pa]
σₙ₀[11] = 21.985*1e6     # Reference normal stress [Pa]
σₙ₀[12] = 23.012*1e6     # Reference normal stress [Pa]
σₙ₀[13] = 24.017*1e6     # Reference normal stress [Pa]
σₙ₀[14] = 24.976*1e6     # Reference normal stress [Pa]
exp_names = ["b724", "b722", "b696", "b726", "b694", "b697", "b693", "b698", "b695", "b728", "b721", "b725", "b727", "i417"]

ϵₜ = Vector{Float64}(undef, 2)
ϵₛ = Vector{Float64}(undef, 2)
ϵₜ[1] = 0
ϵₜ[2] = 1 * 0.004*1e6
ϵₛ[1] = 0
ϵₛ[2] = 1 * 0.006*1e6
diffeq_type = ["ODE", "SDE"]

for ee = 1:2
    for ii = 1:14
        Random.seed!(42)
        print("\n")
        print(ii)
        α = (c₀*p₀*L₁) / (v₀*λ*σₙ₀[ii])
        τ₀ = μ₀*σₙ₀[ii]
        β₂ = -ϵ/(λ*β*σₙ₀[ii])
        κ = (k*L₁) / (a*σₙ₀[ii])
        ν = η*v₀ / (a*σₙ₀[ii])
        ϵ₂ = ϵₜ[ee]/ (a*σₙ₀[ii])
        ϵ₄ = (1/λ)*ϵₛ[ee]/σₙ₀[ii]
        print(" σₙ₀ = ")
        print(σₙ₀[ii]/1e6)
        print(" MPa     ")
        p = (κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄)
        prob_sde_synthetic_eqs = SDEProblem(synthetic_eqs,σ_synthetic_eqs,[0.05,0.0,0.0,0.0],(0.0,5000),p)
        sol = solve(prob_sde_synthetic_eqs, saveat=0.01*v₀/L₁)
        
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/sol_t.txt"), sol.t)
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/sol_u.txt"), sol.u)
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/a.txt"), a)
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/sigman0.txt"), σₙ₀[ii])
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/L1.txt"), L₁)
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/v0.txt"), v₀)
        writedlm(string(dir_data,diffeq_type[ee],"/",exp_names[ii],"/tau0.txt"), τ₀)
    end
end