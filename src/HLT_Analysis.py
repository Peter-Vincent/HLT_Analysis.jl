import glob
import math
import os
import pickle as pkl
import re
from copy import copy
from functools import partial

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pygam as pgm
import sklearn
from jax import grad, jit, lax, random, value_and_grad, vmap
from jax.scipy import special as jspec
from scipy import linalg as sla
from scipy import optimize as sopt
from scipy import stats
from sklearn import linear_model
from tqdm import tqdm

matplotlib.use('Agg')
from itertools import compress

import imageio as imageio
import pylab as pl
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set()
FIG_DIR = "../figs/"
DATA = "../../Data/"

class handler:
    def __init__(self,exp_folder) -> None:
        self.exp_folder = exp_folder
        self.val_grad_row_means_errors = jit(value_and_grad(self.row_means_error,1))

    def parse_folder(self,num_trials):
        dir_files = os.listdir(os.path.join(DATA,self.exp_folder))
        output_file = os.path.join(DATA,self.exp_folder,"participants.txt")
        output_file_bad = os.path.join(DATA,self.exp_folder,"null_participants.txt")
        pt_file = open(output_file,"w")
        pt_file_bad = open(output_file_bad,"w")
        for file in dir_files:
            if ".csv" in file:
                try:
                    cur_file = pd.read_csv(os.path.join(DATA,self.exp_folder,file))
                    cur_trial_num = len(cur_file)
                except:
                    pt_file_bad.write(file + "\n")
                    continue
                if (cur_trial_num >= num_trials) & ("Feedback" in cur_file.columns):
                    pt_file.write(file + "\n")
                else:
                    pt_file_bad.write(file + "\n")
        pt_file.close()
        pt_file_bad.close()

    def parse_exp(self):
        with open(os.path.join(DATA,self.exp_folder,'participants.txt')) as f:
            lines = f.readlines()
        pt_files = [pd.read_csv(os.path.join(DATA,self.exp_folder,x.rstrip())) for x in lines]
        return pd.concat(pt_files)
    
    def parse_exp_pt(self):
        with open(os.path.join(DATA,self.exp_folder,'participants.txt')) as f:
            lines = f.readlines()
        return [pd.read_csv(os.path.join(DATA,self.exp_folder,x.rstrip())) for x in lines]
    
    def parse_log_pt(self):
        with open(os.path.join(DATA,self.exp_folder,'participants.txt')) as f:
            lines = f.readlines()
        

    def split_data(self,all_data,start_trials):
        start_data = all_data[all_data["trials.thisRepN"] < start_trials]
        end_data   = all_data[all_data["trials.thisRepN"] >= start_trials]
        return start_data, end_data
    
    def filter_perf(self,pt_data,thresh=0.15):
        good_data = []
        for data in pt_data:
            cur_samps = data["Sample_Freq_Exp"]
            cur_resps = data["Match_End_Freq_Exp"]
            corr = stats.pearsonr(cur_samps,cur_resps)
            if corr[0] >= thresh:
                good_data.append(data)
        return good_data

    def split_good_data(self,start_trials = 200,thresh = 0.15):
        pt_data = self.parse_exp_pt()
        filt_data = pd.concat(self.filter_perf(pt_data,thresh))
        start_data,end_data = self.split_data(filt_data,start_trials)
        return start_data,end_data

    def disp_means(self,data,num_bins=51):
        samps = data["Sample_Freq_Exp"].to_numpy()
        resps = data["Match_End_Freq_Exp"].to_numpy()
        
        bins = jnp.linspace(10,12,num = num_bins + 1,endpoint=True)
        fig,axs = plt.subplots(1,1,figsize=(8,8))
        data,_,_,_ = axs.hist2d(samps,resps,bins)
        col_means = []; col_inds = []
        row_means = jnp.empty(num_bins); row_inds = jnp.empty(num_bins)
        for i in range(num_bins):
            # Col means first
            binned_samps = (samps > bins[i]) & (samps < bins[i + 1])
            if sum(binned_samps) > 0:
                cur_samps = samps[binned_samps]
                cur_resps = resps[binned_samps]
                col_inds.append(jnp.mean(cur_samps))
                col_means.append(jnp.mean(cur_resps))
            # Row means now
            binned_resps = (resps > bins[i]) & (resps < bins[i + 1])
            cur_samps = samps[binned_resps]
            cur_resps = resps[binned_resps]
            row_means = row_means.at[i].set(jnp.mean(cur_samps))
            row_inds  = row_inds.at[i].set(jnp.mean(cur_resps))
        axs.plot(col_inds,col_means,lw=2,c='r',label="Column means")
        axs.plot(row_means,row_inds,lw=2,c='y',label="Row means")
        axs.legend()
        return fig
    
    def disp_decisions(self,data,kernel = 0.1, num_bins = 51):
        samps = data["Sample_Freq_Exp"].to_numpy()
        resps = data["Match_End_Freq_Exp"].to_numpy()
        fig,axs = plt.subplots(1,1,figsize=(8,8))
        bins = jnp.linspace(10,12,num = num_bins + 1,endpoint=True)
        data,_,_,_ = axs.hist2d(samps,resps,bins)
        axs.axhline(11,lw=2,c='k')
        axs.axvline(11,lw=2,c='k')
        col_inds,col_means,row_means,row_inds,row_modes,row_medians = self.gen_decision(samps,resps,kernel,num_bins)
        axs.plot(col_inds,col_means,lw=2,c='r',label="Column means")
        axs.plot(row_means,row_inds,lw=2,c='y',label="Row means")
        axs.plot(row_modes,row_inds,lw=2,c='b',label="Row modes")
        axs.plot(row_medians,row_inds,lw=2,c='m',label="Row medians")
        axs.legend()
        return fig

    def gen_decision(self,samps,resps,kernel,num_bins):
        bins = jnp.linspace(10,12,num = num_bins + 1,endpoint=True)
        col_means = []; col_inds = []
        row_means = jnp.empty(num_bins); row_inds = jnp.empty(num_bins)
        row_modes = jnp.empty(num_bins); mode_array = jnp.expand_dims(jnp.linspace(10,12,1000),0)
        row_medians = jnp.empty(num_bins)
        for i in range(num_bins):
            # Col means first
            binned_samps = (samps > bins[i]) & (samps < bins[i + 1])
            if sum(binned_samps) > 0:
                cur_samps = samps[binned_samps]
                cur_resps = resps[binned_samps]
                col_inds.append(jnp.mean(cur_samps))
                col_means.append(jnp.mean(cur_resps))
            # Row means now
            binned_resps = (resps > bins[i]) & (resps < bins[i + 1])
            cur_samps = samps[binned_resps]
            cur_resps = resps[binned_resps]
            row_means = row_means.at[i].set(jnp.mean(cur_samps))
            row_inds  = row_inds.at[i].set(jnp.mean(cur_resps))
            # Row modes
            smoothed  = jnp.sum(jnp.exp((-1/2) * ((mode_array - jnp.expand_dims(cur_samps,1))/kernel)**2),axis=0)
            row_modes = row_modes.at[i].set(mode_array[0,jnp.argmax(smoothed)])
            # Row medians
            row_medians = row_medians.at[i].set(np.median(cur_samps))
        return col_inds,col_means,row_means,row_inds,row_modes,row_medians
    
    def compare_decisions(self,samps1,resps1,samps2,resps2,kernel=0.1,num_bins=51):
        col_inds,col_means,row_means,row_inds,row_modes,row_medians = self.gen_decision(samps1,resps1,kernel,num_bins)
        col_inds2,col_means2,row_means2,row_inds2,row_modes2,row_medians2 = self.gen_decision(samps2,resps2,kernel,num_bins)
        fig,axs = plt.subplots(2,2,figsize = (16,16))
        axs[0,0].plot(col_inds,col_means,lw=2,c='r',label="Set 1")
        axs[0,0].plot(col_inds2,col_means2,lw=2,c='g',label="Set 2")
        axs[0,0].legend()
        axs[0,0].set_title("Column means")
        
        axs[0,1].plot(row_means,row_inds,lw=2,c='r',label="Set 1")
        axs[0,1].plot(row_means2,row_inds2,lw=2,c='g',label="Set 2")
        axs[0,1].legend()
        axs[0,1].set_title("Row means")

        axs[1,0].plot(row_modes,row_inds,lw=2,c='r',label = "Set 1")
        axs[1,0].plot(row_modes2,row_inds2,lw=2,c='g',label="Set 2")
        axs[1,0].legend()
        axs[1,0].set_title("Row modes")

        axs[1,1].plot(row_medians,row_inds,lw=2,c='r',label="Set 1")
        axs[1,1].plot(row_medians2,row_inds2,lw=2,c='g',label="Set 2")
        axs[1,1].legend()
        axs[1,1].set_title("Row medians")
        return fig

    def return_arrays(self,dataframe):
        return dataframe["Sample_Freq_Exp"].to_numpy(), dataframe["Match_End_Freq_Exp"].to_numpy()

    def fit_mean_prior(self,data,bin_range = [10.5,11.5],num_bins = 26,lr = 1e-2,max_iter = 1e7,min_error = 1e-6, min_step = 1e-10,report = 100):
        bins = jnp.linspace(bin_range[0],bin_range[1],num_bins + 1)
        s_mids = (bins[:-1] + bins[1:])/2
        data,_,_ = jnp.histogram2d(data["Sample_Freq_Exp"].to_numpy(),data["Match_End_Freq_Exp"].to_numpy(),bins)
        data = data.T
        llhs = self.gen_llhs(data)
        prior = jnp.reshape(jnp.ones(num_bins)/num_bins,(num_bins,1))
        start_prior = copy(prior)
        step_size = 1.0
        error = 1
        counter = 0
        error_store = [0]
        log_prior = jnp.log(prior)
        while (counter < max_iter) & (step_size > min_step) & (error > min_error):
            error,grad = self.val_grad_row_means_errors(llhs,log_prior,s_mids)
            log_prior -= (grad) * lr
            log_prior  = jnp.log(jnp.exp(log_prior) / jnp.sum(jnp.exp(log_prior)))
            if jnp.mod(counter,report) == 0:
                error_store.append(error)
                step_size = np.abs(error_store[-1] - error_store[-2])
                print(f"Iteration -- {counter},   Error -- {error:.3f},   Step size -- {step_size.squeeze():.3f}")
            counter += 1
        del error_store[0]
        posterior = self.gen_posterior(llhs,jnp.exp(log_prior))
        means     = self.mat_row_means(posterior,s_mids)
        return start_prior,jnp.exp(log_prior),error_store,s_mids,posterior,means

    def disp_opt_results(self,start_prior,prior,error_store,s_mids,posterior,means):
        fig,axs = plt.subplots(4,1,figsize=(8,16))
        axs[0].plot(s_mids,start_prior,lw=1,c='k',label="Start prior")
        axs[0].plot(s_mids,prior,lw=2,c='b',label="Fit prior")
        axs[0].set_xlabel("Sample value")
        axs[0].set_ylabel("Prior")
        axs[0].legend()
        axs[1].plot(error_store,lw=2,c='r')
        axs[2].set_title("posterior")
        axs[2].imshow(posterior,origin="lower")
        axs[3].plot(s_mids,means - s_mids,lw=2)
        axs[3].set_title("Posterior expectations")
        return fig

    def gen_posterior(self,llh,prior):
        unnorm_post = llh @ jnp.diag(prior.squeeze())
        return self._row_norm(unnorm_post,prior.shape[0])
    
    def mat_row_means(self,density,s_vals):
        return density @ s_vals

    def row_means_error(self,llhs,log_prior,s_vals):
        posterior = self.gen_posterior(llhs,jnp.exp(log_prior))
        errors    = self.mat_row_means(posterior,s_vals) - s_vals
        return errors @ errors.T

    def _col_norm(self,mat,mat_shape):
        col_normaliser = jnp.tile(jnp.reshape(jnp.sum(mat,axis=0),(1,mat_shape)),(mat_shape,1))
        return mat / col_normaliser
    
    def _row_norm(self,mat,mat_shape):
        row_normaliser = jnp.tile(jnp.reshape(jnp.sum(mat,axis=1),(mat_shape,1)),(1,mat_shape))
        return mat / row_normaliser

    def gen_llhs(self,hist):
        mat_shape = hist.shape[0]
        llh_unnorm = self._col_norm(hist,mat_shape)
        return self._row_norm(hist,mat_shape)

    def alt_hist_results(self,x_density,y_density,x_label,y_label,title,num_bins=51,diag=False):
        # Set up the figure structure
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        # Construct the elements
        diag_min = np.min(np.concatenate([x_density,y_density]))
        diag_max = np.max(np.concatenate([x_density,y_density]))
        diagonal = np.linspace(diag_min,diag_max,180)
        #ax.plot(diagonal,diagonal,c='k',ls='--',lw=2,alpha = 0.5)
        bins = np.linspace(10,12,num_bins+1,endpoint=True)
        max_hist = diag_max
        data,_,_,_ = ax.hist2d(x_density,y_density,bins)
        x_hist,x_bins = np.histogram(x_density,bins)
        x_hist = x_hist * (max_hist / np.max(x_hist))
        y_hist,y_bins = np.histogram(y_density,bins)
        y_hist = y_hist * (max_hist / np.max(y_hist))
        ax_histx.bar(x_bins[:-1], x_hist, align="edge",color='b',linewidth=0,label=x_label,width = np.diff(x_bins))
        ax_histy.barh(y_bins[:-1], y_hist, height = np.diff(y_bins), align="edge",color='y',linewidth=0,label=y_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax_histx.set_title(title)
        # Make things look a little prettier
        
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        ax_histx.spines['bottom'].set_linewidth(2)
        ax_histx.get_yaxis().set_ticks([])
        
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.spines['left'].set_linewidth(2)
        ax_histy.get_xaxis().set_ticks([])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        

        x_values = np.empty(num_bins); y_values = np.empty(num_bins)
        for bin_num in range(num_bins):
            x_vals = np.logical_and(x_density>bins[bin_num],x_density<=bins[bin_num+1])
            x_values[bin_num] = jnp.mean(x_density[x_vals])
            y_values[bin_num] = jnp.mean(y_density[x_vals])
        ax.plot(diagonal,diagonal,c='r',lw=1,alpha=1,ls='--')
        ax.plot(x_values,y_values,c='w',lw=4,alpha=1,ls='--',label="Mean response")
        #ax.legend()
        print(f"data_max = {np.max(data)}")
        data = data / np.sum(data)
        max_prob = np.max(data); min_prob = np.min(data)
        col_fig = pl.figure(figsize=(9,1.5))
        img = pl.imshow(data)
        pl.gca().set_visible(False)
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])
        pl.colorbar(orientation="horizontal", cax=cax,ticks=[min_prob,max_prob])
        return fig, col_fig

    def simple_glm(self,df):
        ## This is probably trash
        samps,resps = self.return_arrays(df)
        match_s     = df["Match_Freq_Exp"]
        num_trials  = resps.shape[0]
        des_mat     = jnp.ones((num_trials,3))
        des_mat = des_mat.at[:,1].set(samps)
        des_mat = des_mat.at[:,2].set(match_s)
        model   = linear_model.LinearRegression().fit(des_mat,resps)
        return model.coef_,model.score(des_mat,samps)
    
    def psychometric_mat(self,df):
        tone1    = jnp.array(df["Psych_1"].to_numpy())
        tone2    = jnp.array(df["Psych_2"].to_numpy())
        feedback = jnp.array(df["Feedback"].to_numpy())
        fig,axs = plt.subplots(2,1,sharex=True,figsize=(16,16),gridspec_kw={'height_ratios' : [3,1]})
        uni_tone1= jnp.unique(tone1)
        tone1_perf   = jnp.zeros(uni_tone1.shape[0])
        t1_list  = []
        dif_list = []
        col_list = []
        for tone1_ind, cur_tone1 in enumerate(uni_tone1):
            cur_trials = tone1 == cur_tone1
            cur_tone2  = tone2[cur_trials]
            tone1_perf = tone1_perf.at[tone1_ind].set(jnp.sum(feedback[cur_trials])/jnp.sum(cur_trials))
            uni_tone2  = jnp.unique(cur_tone2)
            cur_difs   = jnp.empty(uni_tone2.shape[0])
            cur_perf   = jnp.empty(uni_tone2.shape[0])
            for tone2_ind, cur_tone2 in enumerate(uni_tone2):
                cur_set = cur_trials & (tone2 == cur_tone2)
                cur_difs = cur_difs.at[tone2_ind].set(cur_tone2 - cur_tone1)
                cur_perf = cur_perf.at[tone2_ind].set(jnp.sum(feedback[cur_set]) / jnp.sum(cur_set))
            t1_list.append(jnp.full_like(cur_difs,cur_tone1))
            dif_list.append(cur_difs)
            col_list.append(cur_perf)
        sc1 = axs[0].scatter(jnp.concatenate(t1_list),jnp.concatenate(dif_list),c=jnp.concatenate(col_list),s=1500,edgecolors='k',linewidths=2,cmap="seismic",vmin=0.5,vmax=1)
        axs[1].plot(uni_tone1,tone1_perf,lw=2,label="Performance")
        axs[1].legend()
        axs[1].set_xlabel("Tone 1")
        axs[1].set_ylabel("Performance")
        axs[0].set_ylabel("Tone 2 - Tone 1")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('top', size='5%', pad=0.05)
        fig.colorbar(sc1, cax=cax, orientation='horizontal')
        return fig

    def demo_results_rows(self,x_density,y_density,x_label,y_label,title,num_bins=51,diag=False):
        # Set up the figure structure
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        # Construct the elements
        diag_min = np.min(np.concatenate([x_density,y_density]))
        diag_max = np.max(np.concatenate([x_density,y_density]))
        diagonal = np.linspace(diag_min,diag_max,180)
        #ax.plot(diagonal,diagonal,c='k',ls='--',lw=2,alpha = 0.5)
        bins = np.linspace(10,12,num_bins+1,endpoint=True)
        max_hist = diag_max
        data,_,_,_ = ax.hist2d(x_density,y_density,bins)
        x_hist,x_bins = np.histogram(x_density,bins)
        x_hist = x_hist * (max_hist / np.max(x_hist))
        y_hist,y_bins = np.histogram(y_density,bins)
        y_hist = y_hist * (max_hist / np.max(y_hist))
        ax_histx.bar(x_bins[:-1], x_hist, align="edge",color='b',linewidth=0,label=x_label,width = np.diff(x_bins))
        ax_histy.barh(y_bins[:-1], y_hist, height = np.diff(y_bins), align="edge",color='y',linewidth=0,label=y_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax_histx.set_title(title)
        # Make things look a little prettier
        
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        ax_histx.spines['bottom'].set_linewidth(2)
        ax_histx.get_yaxis().set_ticks([])
        
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.spines['left'].set_linewidth(2)
        ax_histy.get_xaxis().set_ticks([])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        

        x_values = np.empty(num_bins); y_values = np.empty(num_bins)
        row_means= np.empty(num_bins)
        row_mean_locs= np.empty(num_bins)
        for bin_num in range(num_bins):
            x_vals = np.logical_and(x_density>bins[bin_num],x_density<=bins[bin_num+1])
            y_vals = np.logical_and(y_density>bins[bin_num],y_density<=bins[bin_num + 1])
            row_means[bin_num]     = jnp.mean(x_density[y_vals])
            row_mean_locs[bin_num] = jnp.mean(y_density[y_vals])
            x_values[bin_num] = jnp.mean(x_density[x_vals])
            y_values[bin_num] = jnp.mean(y_density[x_vals])
        #ax.plot(diagonal,diagonal,c='r',lw=1,alpha=1,ls='--')
        ax.plot(x_values,y_values,c='w',lw=4,alpha=1,ls='--',label="Mean response")
        ax.plot(row_means,row_mean_locs,c='y',lw=5)
        #ax.legend()
        print(f"data_max = {np.max(data)}")
        data = data / np.sum(data)
        max_prob = np.max(data); min_prob = np.min(data)
        col_fig = pl.figure(figsize=(9,1.5))
        img = pl.imshow(data)
        pl.gca().set_visible(False)
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])
        pl.colorbar(orientation="horizontal", cax=cax,ticks=[min_prob,max_prob])
        return fig, col_fig

class models:

    def __init__(self,pt_df):
        self.pt_df = pt_df
        self.full_df = pd.concat(self.pt_df)
        self.p_correct = self._get_psychometric(self.full_df)
        

    def _extract_arrays(self,df):
        self.samps = jnp.array(df["Sample_Freq_Exp"].to_numpy())
        self.resps = jnp.array(df["Match_End_Freq_Exp"].to_numpy())
        self.match = jnp.array(df["Match_Freq_Exp"].to_numpy())
        self.f1    = jnp.array(df["Psych_1"].to_numpy())
        self.f2    = jnp.array(df["Psych_2"].to_numpy())
        self.trial = jnp.array(df["trials.thisRepN"].to_numpy())
        self.feedback = df["Feedback"].to_numpy()
        self._answers()

    def _answers(self):
        pass


        

class GLM:
        
    def simp_design_mat(self,d,psych_num_back = 4,samp_num_back = 4, match_num_back = 4):
        pass
    
    def glm(self):
        reg = linear_model.TweedieRegressor(power=0)

class GAM(models):
    def simp_inter_des_mat(self):
        num_pt = len(self.pt_df)
        pt_des_mats = []
        pt_y        = []
        print("Constructing design matrix...")
        for pt_ind in tqdm(range(num_pt)):
            cur_pt = self.pt_df[pt_ind]
            self._extract_arrays(cur_pt)
            mean_psych_stim = jnp.mean(jnp.concatenate((self.f1,self.f2)))
            num_trials  = len(cur_pt)
            cur_des_mat = jnp.ones((num_trials -1, 3))
            cur_y       = jnp.empty(num_trials - 1)
            cur_des_mat = cur_des_mat.at[:,0].set(self.f2[self.trial > 0] - self.f1[self.trial > 0])
            d_inf    = self.f1[self.trial > 0] - mean_psych_stim
            one_back = self.trial < (num_trials - 1)
            d1       = self.resps[one_back] - self.f1[self.trial > 0]
            cur_des_mat = cur_des_mat.at[:,1].set(d1)
            cur_des_mat = cur_des_mat.at[:,2].set(d_inf)
            pt_des_mats.append(cur_des_mat)
            for i in range(1,num_trials):
                cur_y = cur_y.at[i-1].set(self.p_correct[(float(self.f1[i]),float(self.f2[i]))])
            pt_y.append(cur_y)

        des_mat = jnp.concatenate(pt_des_mats)
        pt      = jnp.concatenate(pt_y)
        return des_mat,pt
    
    def simp_des_mat(self):
        num_pt = len(self.pt_df)
        pt_des_mats = []
        pt_y        = []
        print("Constructing design matrix...")
        for pt_ind in tqdm(range(num_pt)):
            cur_pt = self.pt_df[pt_ind]
            self._extract_arrays(cur_pt)
            mean_psych_stim = jnp.mean(jnp.concatenate((self.f1,self.f2)))
            num_trials  = len(cur_pt)
            cur_des_mat = jnp.ones((num_trials -1, 3))
            cur_y       = jnp.empty(num_trials - 1)
            cur_des_mat = cur_des_mat.at[:,0].set(self.f2[self.trial > 0] - self.f1[self.trial > 0])
            d_inf    = self.f1[self.trial > 0] - mean_psych_stim
            one_back = self.trial < (num_trials - 1)
            d1       = (self.f2[one_back] + self.f1[one_back])/2 - self.f1[self.trial > 0]
            cur_des_mat = cur_des_mat.at[:,1].set(d1)
            cur_des_mat = cur_des_mat.at[:,2].set(d_inf)
            pt_des_mats.append(cur_des_mat)
            for i in range(1,num_trials):
                cur_y = cur_y.at[i-1].set(self.p_correct[(float(self.f1[i]),float(self.f2[i]))])
            pt_y.append(cur_y)

        des_mat = jnp.concatenate(pt_des_mats)
        pt      = jnp.concatenate(pt_y)
        return des_mat,pt


    def run_3_gam(self,X,y):
        gam = pgm.LinearGAM(pgm.s(0) + pgm.s(1) + pgm.s(2)).fit(X,y)
        gam.summary()
        fig,axs = plt.subplots(3,1,figsize=(4,12))
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue

            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

            axs[i].plot(XX[:, term.feature], pdep)
            axs[i].plot(XX[:, term.feature], confi, c='r', ls='--')
            axs[i].set_title(repr(term))
        return fig


    def _get_psychometric(self,df):
        self._extract_arrays(df)
        tone1    = self.f1
        tone2    = self.f2
        feedback = self.feedback
        uni_tone1= jnp.unique(tone1)
        p_correct = {}
        for tone1_ind, cur_tone1 in enumerate(uni_tone1):
            cur_trials = tone1 == cur_tone1
            valid_tone2  = tone2[cur_trials]
            uni_tone2  = jnp.unique(valid_tone2)
            for tone2_ind, cur_tone2 in enumerate(uni_tone2):
                cur_set = cur_trials & (tone2 == cur_tone2)
                p_correct[(cur_tone1,cur_tone2)] = jnp.sum(feedback[cur_set]) / jnp.sum(cur_set)
        return p_correct
            

def main():
    data_handler = handler("base_psych_uniform")
    data_handler.parse_folder(400)
    start_data,end_data = data_handler.split_good_data()
    all_data = data_handler.parse_exp()
    x_data,y_data = data_handler.return_arrays(all_data)
    psych_plot = data_handler.psychometric_mat(all_data)
    psych_plot.savefig("psych_fig.eps")
    
    
    



if __name__ == "__main__":
    main()