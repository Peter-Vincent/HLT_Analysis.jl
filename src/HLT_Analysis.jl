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

function partition_data(exp_folder::String,segments::Int,min_corr::Float64 = 0.15)
    _,pt_data,_ = pool_pt(exp_folder,min_corr)
    num_pt = length(pt_data)
    split_data = partition_data.(pt_data,segments)
    data_sets = Array{DataFrame,1}(undef,segments)
    seg_store = Array{DataFrame,1}(undef,num_pt)
    for i in 1:segments
        for pt in 1:num_pt
            seg_store[pt] = split_data[pt][i]
        end
        data_sets[i] = reduce(vcat,seg_store)
    end
    return data_sets
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
    score_list= Vector{Float64}()
    for cur_psych1 in psych1_vals
        cur_points = findall(x->x==cur_psych1,psych1)
        cur_diffs = diff[cur_points]
        cur_feedback = feedback[cur_points]
        for cur_diff in unique(cur_diffs)
            cur_points = findall(x->x==cur_diff,cur_diffs)
            set_feedback = cur_feedback[cur_points]
            score = sum(set_feedback) / length(set_feedback)
            append!(dif_list,cur_diff)
            append!(psych_list,cur_psych1)
            append!(score_list,score)
        end
    end
    scatter(psych_list,dif_list,zcolor=score_list,xlabel="Sample freq exponent",ylabel="Difference",xlim=[10,12],lab="pair",markersize=15)
end

function row_mean_modes(sample_freq::Vector,response_freq::Vector,
                        bin_vals::Vector,kernel::Float64)
    num_bins = length(bin_vals) - 1
    bin_mids = (bin_vals[2:end] .+ bin_vals[1:end-1])./2
    mean_resps = Array{Float64,1}(undef,num_bins)
    mean_samps = Array{Float64,1}(undef,num_bins)
    mode_samps = Array{Float64,1}(undef,num_bins)
    for i in 1:num_bins
        inds = (response_freq .> bin_vals[i]) .& (response_freq .< bin_vals[i+1])
        mean_resps[i] = mean(response_freq[inds])
        mean_samps[i] = mean(sample_freq[inds])
        mode_values   = kde(sample_freq[inds],bin_mids,kernel)
        max_inds      = findall(mode_values .== maximum(mode_values))
        mode_samps[i] = bin_mids[max_inds[1]]
    end
    mean_resps,mean_samps,mode_samps,bin_mids
end

function row_mean_modes(sample_freq::Vector,response_freq::Vector,
                        bin_vals::Vector,kernel::Float64,iterations::Int)

end
    
    
function gauss_smooth(points::Vector,mean::Float64,kernel::Float64)
    return exp.(-(1/2).*(((mean .- points)./kernel).^2))
end
        
function kde(values::Vector,eval_points::Vector,kernel::Float64)
    num_evals = length(eval_points)
    kde_values = Array{Float64,1}(undef,num_evals)
    for i in 1:num_evals
        weights = gauss_smooth(values,eval_points[i],kernel)
        kde_values[i] = dot(weights,values)
    end
    return kde_values 
end

    
function decision_rule(df::DataFrame,num_bins::Int=25,kernel = 0.25)
    sample_freq =  Vector(df.Sample_Freq_Exp)
    response_freq =Vector(df.Match_End_Freq_Exp)
    bin_vals = Vector(LinRange(10,12,num_bins + 1))
    mean_resps, mean_samps, mode_samps, mode_points = row_mean_modes(sample_freq,response_freq,bin_vals,kernel)
    mean_error = mean_resps .- mean_samps
    mode_error = mean_resps .- mode_samps
    plot(mean_resps,mean_error,lw=5,c="orange",xlabel="Response value",ylabel="Prediction error",label="means")
    plot!(mode_points,mode_error,lw=5,c="blue",label="Modes")
end

function decision_rule(df_store::Vector{DataFrame},num_bins::Int=25,kernel=0.25)
    bin_vals = Vector(LinRange(10,12,num_bins + 1))
    mean_resps = Array{Float64,1}(undef,num_bins)
    mean_samps = Array{Float64,1}(undef,num_bins)
    num_segs = length(df_store)
    # create the empty plot
    fig = plot(xlabel="Response value",ylabel="Prediction error",title="Learning across time")
    for seg in 1:num_segs
        df = df_store[seg]
        sample_freq =  Vector(df.Sample_Freq_Exp)
        response_freq =Vector(df.Match_End_Freq_Exp)
        mean_resps, mean_samps, mode_samps, mode_points = row_mean_modes(sample_freq,response_freq,bin_vals,kernel)
        mean_error = mean_resps .- mean_samps
        mode_error = mode_points .- mode_samps
        plot!(mean_resps,mean_error,lw=5,label = seg)
        plot!(mode_points,mode_error,lw=5,label=seg)
    end
    return fig
end


function find_prior(df::DataFrame,num_bins::Int=25)
    
end
    
function summary(df::DataFrame,num_bins::Int = 25)
    sample_freq =  Vector(df.Sample_Freq_Exp)
    response_freq =Vector(df.Match_End_Freq_Exp)
    resp_time     = Vector(df.Match_End_Time - df.Match_Start_Time)
    resp_time[resp_time .<=0] .= 0
    summary_hist = marginalhist(sample_freq,response_freq,xlabel="log Sample freq",ylabel="log response freq",bins=LinRange(10,12,num_bins))
    summary_scatter = scatter(sample_freq,response_freq,mode="markers",xlabel="log Sample freq",ylabel="log response freq",markersize=5 .*sqrt.(resp_time))
    l = @layout [a;b]
    plot(summary_hist,summary_scatter,layout=l,size=(800,1600))
end

function performance(df::DataFrame)
    sample_freq =  Vector(df.Sample_Freq_Exp)
    response_freq =Vector(df.Match_End_Freq_Exp)
    pearson_corr = Statistics.cor(sample_freq,response_freq)
    return pearson_corr
end

function pool_pt(path::String,threshold::Float64=0.15)
    pt_data = parse_exp_pt(path)
    correlations = performance.(pt_data)
    valid_data = pt_data[correlations .> threshold]
    full_data = reduce(vcat,valid_data)
    return correlations,valid_data,full_data
end

function time_histogram(df::DataFrame,num_bins::Int = 25,thresh::Int = 10)
    sample_freq   = Vector(df.Sample_Freq_Exp)
    response_freq = Vector(df.Match_End_Freq_Exp)
    resp_time     = Vector(df.Match_End_Time - df.Match_Start_Time)
    resp_time[resp_time .<=0] .= 0
    bin_vals      = LinRange(10,12,num_bins + 1)
    time_matrix   = zeros(num_bins,num_bins)
    for resp_bins in 1:num_bins
        for samp_bins in 1:num_bins
            samp_inds = (sample_freq .> bin_vals[samp_bins]) .& (sample_freq .< bin_vals[samp_bins+1])
            resp_inds = (response_freq .> bin_vals[resp_bins]) .& (response_freq .< bin_vals[resp_bins+1])
            bin_inds = samp_inds .& resp_inds
            if sum(bin_inds) > thresh
                time_matrix[resp_bins,samp_bins] = mean(resp_time[bin_inds])
            end
        end
    end
    return heatmap(time_matrix)
end

end
