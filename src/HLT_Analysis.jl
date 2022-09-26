module HLT_Analysis

using CSV
using DataFrames
using LinearAlgebra

function parse_exp(exp_folder::String)
    pt_files = joinpath.(exp_folder,readlines(joinpath(exp_folder,"participants.txt")))
    return mapreduce(x->CSV.read(x,DataFrame),vcat,pt_files)
end

function parse_exp_pt(exp_folder::String)
	pt_files = joinpath.(exp_folder,readlines(joinpath(exp_folder,"participants.txt")))
	num_pt = length(pt_files)
	pt_data = Array{DataFrame,1}(undef,num_pt)
	for (ind,pt_file) in enumerate(pt_files)
		pt_data[ind] = CSV.read(pt_file,DataFrame)
	end
	return pt_data
end
		

function parse_folder(exp_folder::String,num_trials::Int)
    dir_files = readdir(exp_folder)
    output_file = joinpath(exp_folder,"participants.txt")
	output_file_bad = joinpath(exp_folder,"null_participants.txt")
    touch(output_file)
	touch(output_file_bad)
    pt_file = open(output_file,"w")
	pt_file_bad = open(output_file_bad,"w")
    for file in dir_files
        if match(r".csv",file) !== nothing
            cur_file = CSV.read(joinpath(exp_folder,file),DataFrame)
            cur_trial_num = nrow(cur_file)
            if (cur_trial_num === num_trials) & ("Feedback" in names(cur_file))
               println(pt_file,file) 
			else
			   println(pt_file_bad,file) 
			end
        end
    end
	close(pt_file)
	close(pt_file_bad)
end

function partition_data(df::DataFrame,segments::Int)
    unique_pt = unique(df.PROLIFIC_PID)
    num_pt = length(unique_pt)
    block_length = size(df,1) / num_pt
    segment_length = block_length ÷ segments
    remainder = block_length - (segment_length * segments)
    df_store  = Array{DataFrame,1}(undef,segments)
    blocks = fill(segment_length,segments)
    blocks[lastindex(blocks)] = last(blocks) + remainder
    for i in 1:segments
        cur_trials  = (collect(1:blocks[i]) .+ sum(blocks[1:(i-1)])) .- 1
        df_store[i] = df[in(cur_trials).(df."trials.thisN"),:]
    end
    return df_store
end
    
function row_means(density::Matrix,low_bound,high_bound)
    num_cols = size(density,2)
    bin_edges= LinRange(low_bound,high_bound,num_cols+1)
    values   = collect((bin_edges[1:lastindex(bin_edges)-1] + bin_edges[2:lastindex(bin_edges)])/2)
    means    = 1 ./sum(density,dims=2) .* (density * values)
    return means,values
end

function column_means(density::Matrix,low_bound,high_bound)
    num_cols = size(density,2)
    bin_edges= LinRange(low_bound,high_bound,num_cols+1)
    values   = collect((bin_edges[1:lastindex(bin_edges)-1] + bin_edges[2:lastindex(bin_edges)])/2)
    means    = 1 ./sum(density,dims=1) .* (transpose(values) * density)
    return means,values
end

function conditional(joint::Matrix,sample::Vector)
    return joint * Diagonal(1 ./ sample)
end

function psychometric(df::DataFrame)
	psych1 = df.Psych_1
	psych2 = df.Psych_2
	diff = psych2 - psych1
	feedback = df.Feedback
	psych1_vals = unique(psych1)
	dif_list  = Vector{Float64}()
	psych_list= Vector{Float64}()
	score_list= Vector{Int}()
	for cur_psych1 in psych1_vals
		cur_points = findall(x->x==cur_psych1,psych1)
		cur_diffs = diff[cur_points]
		cur_feedback = feedback[cur_points]
		for cur_diff in unique(cur_diffs)
			cur_points = findall(x->x==cur_diff,cur_diffs)
			set_feedback = cur_feedback[cur_points]
			score = sum(set_feedback)
			append!(dif_list,cur_diff)
			append!(psych_list,cur_psych1)
			append!(score_list,score)
		end
	end
	scatter(psych_list,dif_list,zcolor=score_list,xlabel="Sample freq exponent",ylabel="Difference",xlim=[10,12],lab="pair")
end

function test_revise()
    println("working?")
end

end
