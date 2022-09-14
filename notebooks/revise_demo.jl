### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 01b97702-3449-11ed-2756-a905b4e759e6
begin
	import Pkg
	Pkg.add(path="..")
	#Pkg.add("/mnt/c/Users/Murri/Documents/SWC/HLT/HLT_Analysis/")
	Pkg.add("Revise")

end

# ╔═╡ ddda6f0a-1445-4aa0-abae-0dd90f0dd6e6
begin
	using Revise
	import HLT_Analysis as ha
end

# ╔═╡ ee37494c-8143-4663-8188-45c1f86e872b
begin
	ha.foo()
end

# ╔═╡ 27a0237c-ec64-4c99-978c-762be4e6e42a
begin
path = "../../Data/base_HLT/"
fig_path = "../../Analysis/"
ha.parse_folder(path,400)
end

# ╔═╡ Cell order:
# ╠═01b97702-3449-11ed-2756-a905b4e759e6
# ╠═ddda6f0a-1445-4aa0-abae-0dd90f0dd6e6
# ╠═ee37494c-8143-4663-8188-45c1f86e872b
# ╠═27a0237c-ec64-4c99-978c-762be4e6e42a
