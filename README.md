# HLT_Analysis

This repo contains the analysis codes for the Hierarchical Learning Task, which was active at some point between 2021 and 2022.

A full copy of the directory, including codes and data, can be found on `ceph/akrami/Peter/HLT`

## Experiments
The experiments for this project were performed online using PsychoJS, hosted on Pavlovia.  An example experiment is [here](https://run.pavlovia.org/PeterVincent96/base_uniform_hlt)

All data can be downloaded from the git repository where the codes are hosted.  It is not backed-up in this repository.  It can also be accessed on the external harddrive at the folloing path - `D:\Old_Laptop_Final_Backup_18_11_2024\SWC\HLT\Data`

## Analysis codes

There are two locations for the anlysis codes.  The first is in the `notebooks` directory.  This contains a Pluto notebook (Julia) called `analysis_nb.jl`.  However, these analyses are outdated.  I started doing analysis in Julia but pretty quickly realised there were major issues with auto-diff in Julia so I decided to move on.  The julia source code for this notebook are in the `src` directory, in the `src/HLT_Analysis.jl` file.  For the python analysis, which extends a lot further, the code is in the `src/HLT_Analysis.py` file.  The `src` directory also includes some basic figures

## State of the project

This project ground to a halt when we realised the unbounded nature of the data makes it impossible to estimate the likelihoods, since we can not normalise the distributions.  This means we could not use many of the techniques we used in the Orientation project.  

However, the project still has scope.  The underlying task design is interesting and some good work could probagly done on inference and statistical learning.  Collecting data in this experiment is pretty easy, and the core analysis codes are already there.

