
# settings.py

# def init():
global M
global m
global pot
global b
global a
global x0
global c

# m=0.001
# M=1.5
##############################
## SPRING POTENTIAL RESULTS ##
##############################
## easier problem 
#####################
## with parameters
# m=0.001
# M=1.5
# pot="spring"
# a=1
# b=1
# x0=0.5
# c=0.1
# dt_list= [0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6]
# T=100
# tau=0.1
# n_samples=5000123
# x=0
# range_bins=[-10,10] # range to plot samples on histograms
# nbins=50
# range_int=[-10,10] # integration range

## very hard problem 
#####################
## with parameters
m=0.001
M=1.5
pot="spring"
a=10
b=0.1
x0=0.5
c=0.1
dt_list= [0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6]
#dt_list= [0.0007,0.0009,0.001,0.003,0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5]
T=100 
tau=0.1
n_samples=10000000
x=0
range_bins=[-1,1] # range to plot samples on histograms
nbins=50
range_int=[-10,10] # integration range 


# Spring potential 
# pot="spring"
# a=10
# b=0.1
# x0=0.5
# c=0.1

# Parameters for extra precise results on accuracy with results 
## Easy - spring potential 
# pot="spring"
# a=1
# b=1
# x0=0.5
# c=0.1

# # with results 
# ## set up others values
# T=100
# tau=0.1
# n_samples=10000000
# h=0.05
# Nt= int(T*1/h)+1
# # m=0.001
# # M=1.5 
# x=0
# range_bins=[-1,1] # range to plot samples on histograms
# nbins=50
# range_int=[-10,10] # integration range 

