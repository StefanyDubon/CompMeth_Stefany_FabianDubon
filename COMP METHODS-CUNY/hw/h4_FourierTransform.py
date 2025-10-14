"""HW 4 """
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import zeros 
from cmath import exp,pi 
from numpy import concat

"""HW4 FOURIER ANAYSIS - TIME SERIES ANALYSIS"""

hdul = fits.open("tic0000120016.fits")
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']


#Plot the dtat from the fits file to see what epoch of observations to use
plt.plot(times, fluxes, color= "blue")
plt.xlabel("times")
plt.ylabel("fluxes")
plt.title("Time Series Plot")
# plt.xlim(3084, 3090)
# plt.show()

#create a mask so that I can use logical operators to reeduce the data to the chosen range
np.shape(times),np.shape(fluxes), np.shape(ferrs)
#logical operators)

np.shape(times),np.shape(fluxes), np.shape(ferrs)
#logical operators)
mask = (times >=2342) & (times<=2344)
mask

times_epoch= times[mask]
fluxes_epoch = fluxes[mask]
ferrs_epoch= ferrs[mask] 

# print("times epoch", len(times_epoch))
# print("fluxes epoch", len(fluxes_epoch), fluxes_epoch)

plt.plot(times_epoch, fluxes_epoch)
plt.xlabel("time")
plt.ylabel("flux")
plt.title("TIC 120016: 2340-2345 epoch ")
# plt.show()

    
#perform the fourier analysis
"""Discrete Fourier transform aka DFT. The COEFFICIENTS are defined as:
        c_k = N_yk = sum(upper : N-1, lower n=0)exp(-i(2pikx_n)/L)"""

def dft(y):               # perform Discrete Fourier Transform on vector y using loop
    N = len(y)
    N_real = N//2+1             # this is for when thereal-valued signals, since its spectrum is symmetrical 
                                #it only takes the coefficitents for the first half. 
    c = np.zeros(N,dtype='complex')
    for k in range(N):           #use the whole array so that I can reconstruct the full signal whrn I do the Inverse
        for n in range(N):
            c[k] += y[n]*np.exp(-2j*np.pi*k*n/N)
    return c


#I'm making another mask for a smaller range, where there is not missing information so 
#I'll use that to do the first fourier transform, and then use the other range for when I do the linear interpolation

# mask2 = (times >=1500) & (times<=1650)
# mask
# #ns = no space between the points 
# times_ns= times[mask2] 
# fluxes_ns = fluxes[mask2]
# ferrs_ns= ferrs[mask2] 

c = dft(fluxes_epoch)
# print("fourier transform", len(c), c)

#we need the magnitude square ck^2 of the fourier coeffcients for the power spectrum
#a power spectrum is a plot of the absolute values of the coefficients |ck| 
#and it shows the relative contribution of waves of each frequency

abs_c = np.abs(c)**2
print("magnitude squared" ,abs_c[0:2])

k = np.arange(len(c))
# print(k)
# abs_c[0] = 0
plt.plot(k,abs_c)
# plt.axvline(1, c="r")
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title("Power Spectrum")
# plt.xlim(-1,200)
plt.ylim(0, 1000)


""" Now that we have the power spectrum, we have to see the few coefficients I can use 
to capture the behaviour of the eclipse binary loking at the inverse transform
THE FORMULA FOR INVERSE DFTIS : 
                        y_n= 1/2 sum[ upper n-1, lower k=0] c_k exp((i2pikn)/(N))"""

def inverseft(c):
    N = len(c)
    y = np.zeros(N, dtype= 'complex')
    for n in range (N):
        for k in range(N):
            y[n] += c[k]*np.exp(2j*pi*k*n/N)
    y/= N #divide the total sum once for each y[n]
    return y
        
    
fluxes_inverse = inverseft(c)

# print("Inverse ft fluxes", fluxes_inverse)

#use the real numpy built in fuction to only take the real part so that there the array is not complex
fluxes_invreal = np.real(fluxes_inverse)


#Plot all 4 graphs (Original time series, FFT, power spectrum, Inverse FFT, and FFT vs IFFT for this epoch in the data set
# print("real fluxes for inverse fourier transform ", len(fluxes_invreal), fluxes_invreal)
print(len(times_epoch), len(fluxes_invreal))
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#Original time series
axs[0, 0].plot(times_epoch, fluxes_epoch, color="blue")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Flux")
axs[0, 0].set_title("TIC 120016: 2340-2344 Epoch")

# Power Spectrum
axs[0, 1].plot(k, abs_c, color="green")
axs[0, 1].set_xlabel("k")
axs[0, 1].set_ylabel(r"$|c|^2$")
axs[0, 1].set_title("Power Spectrum")
axs[0, 1].set_ylim(0, 1000)

#Inverse FFT
axs[1, 0].plot(times_epoch, fluxes_invreal, color="red")
axs[1, 0].set_xlabel("time")
axs[1, 0].set_ylabel("Flux")
axs[1, 0].set_title("Inverse Fourier Transform")

# Original vs FFT
axs[1, 1].plot(times_epoch, fluxes_epoch, label="Original", color="blue")
axs[1, 1].plot(times_epoch, fluxes_invreal, '--', label="Reconstructed (Inverse FFT)", color="red")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Flux")
axs[1, 1].set_title("Original vs Inverse FFT")
axs[1, 1].legend()
fig.suptitle('TIC 120016: 2342-2344 Epoch', fontsize=16)
plt.tight_layout()
plt.show()




"""Filling the missing time steps with linear interpolation"""

#before doing the interpolation, I need to find the missing time steps in the portion that has lots of data
#Make a new mask, this time for the range I need to make the x argument needed to use the
#interpolation function built in numpy later on

mask_interp= (times>=2342) & (times<=2344)
print(len(mask_interp))
times_interp = times[mask_interp]
fluxes_interp = fluxes[mask_interp]

print("length of data points in the chosen range", len(times_interp))
print("times_interp length", len(times_interp), "fluxes_inter length ", len(fluxes_interp)  )
# plt.plot(times[mask_interp], fluxes[mask_interp])


t_max= np.max(times_interp)
t_min = np.min(times_interp)

#compute the difference between the first and second point
time_step = times_interp[1] - times_interp[0]
print("expected time step ", time_step)

missing_ts= []
t = t_min

# while t <= t_max:
#     expected_p.append(t)
#     t += time_step

#its better to loop throught the data so that i can find the gaps
for i in range(len(times_interp)-1):
    t_current = times_interp[i]
    t_next = times_interp[i+1]

    t_expected = t_current + time_step
    while t_expected < t_next - 1e-8:   #small tolerance
        missing_ts.append(t_expected)
        t_expected += time_step


print(f"Number of missing points detected: {len(missing_ts)}")
if len(missing_ts) > 0:
    print("First few missing times:", missing_ts[:10])

#Plot the original data and highlight gaps ---
plt.figure(figsize=(8,4))
plt.plot(times_interp, fluxes_interp, 'o-', label='Original Data', color='blue')

# Highlight missing gaps
for t in missing_ts:
    plt.axvline(t, color='red', alpha=0.5, linestyle='--')

plt.xlabel('Time')
plt.ylabel('Flux')
plt.title('Detected Missing Points (Red Lines)')
plt.legend()
plt.tight_layout()
# plt.show()

# expected_p = np.array(expected_p)
# print(len(expected_p))
# plt.plot(times_interp, expected_p)
# print(expected_p)

""""Now, that we know where the missing points are, I can use the np.interp to fill those missing points with
        linear interpolation"""

#to make that x array, take the max data set, min data set and (THHIS WAS WRONG)
#then for the step sixe, get the difference between them, and divde by the length of data

# t_max= np.max(times_interp)
# t_min = np.min(times_interp)
# step = ((max-min)/len(times[mask_interp]))
# print("max",max , "min", min, step)
# # min =2342
# x_inter = np.arange(min, max, ((max-min)/len(times[mask_interp])))

# print(len(x_inter))
# # np.interp()


merged_times = np.concatenate((times_interp, missing_ts))  #combine the original range of the times array with the missing time steps array
                                                            #because the 

merged_fluxes_li = np.interp(merged_times, times_interp, fluxes_interp) #used the merged array to get the L.I. for the missing time steps

# print("merged arrays", len(merged_times), mergmerged_times)


#THIS IS ANOTHER WAY TO CREATE THE TIMES ARRAY USED TO FIND THE LINEAR INTERPOLATION FOR THE MISSING TIME STEPS 
#(don't need to first find the location of the missing times steps)
times_uniform = np.arange(times_interp[0], times_interp[-1]+time_step, time_step)
fluxes_uniform = np.interp(times_uniform, times_interp, fluxes_interp)

#I printed all the statements cuz I wanted to make sure I would get the same results for both methods of getting the LI
print("time interp", len(times_interp), times_interp[:10])
print("merged time arrays", len(merged_times), merged_times[:10])
print("times uniform", len(times_uniform), times_uniform[:10])

print("flux interp", len(fluxes_interp), fluxes_interp[:10])
print("merged fluxes LI", len(merged_fluxes_li), merged_fluxes_li[:10])
print("fluxes uniform", len(fluxes_uniform), fluxes_uniform[:10])


plt.figure(figsize=(8,4))
plt.plot(times_interp, fluxes_interp, 'o--', label='Original Data', color='blue', markersize= 11 )
plt.plot(merged_times, merged_fluxes_li,   "s", label= " Linear interpolation Points", color = "orange", markersize= 6)
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title('Time series epoch filled in with Linear Interpolation ')
plt.legend()
plt.tight_layout()
plt.show()

"""Finally, redo the Fourier analysis """
c2_dft_interp= dft(merged_fluxes_li)

# print("fourier transform for the LI data ", len(c2_dft_interp), c2_dft_interp)

abs_c2 = np.abs(c2_dft_interp)**2
# print(abs_c2[0:2])

k2 = np.arange(len(c2_dft_interp))
print(k2)
# abs_c2[0] = 0
plt.plot(k2,abs_c2)
# plt.axvline(1, c="r")
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title("Power Spectrum for Time series with interpolation")
# plt.xlim(-1,200)
plt.ylim(0, 1000)
# plt.show()

li_fluxes_inverse = inverseft(c2_dft_interp)

print("fluxes for inverse fourier transform ", len(li_fluxes_inverse), len(k), len(times_epoch))

#use the real numpy built in fuction to only take the real part so that there the array is not complex
li_fluxes_invreal = np.real(li_fluxes_inverse)
print("real fluxes for inverse fourier transform ", len(fluxes_invreal))

plt.plot(merged_times, li_fluxes_invreal, 'o')
plt.plot(times_epoch, fluxes_epoch, "o--")
# plt.show()

"""Finally, redo the Fourier analysis """
c2_dft_interp= dft(merged_fluxes_li)

# print("fourier transform for the LI data ", len(c2_dft_interp), c2_dft_interp)

abs_c2 = np.abs(c2_dft_interp)**2


k2 = np.arange(len(c2_dft_interp))
# print(k2)
# abs_c2[0] = 0
plt.plot(k2,abs_c2)
# plt.axvline(1, c="r")
plt.ylabel('$|c|^2$')
plt.xlabel('k')
plt.title("Power Spectrum for Time series with interpolation")
# plt.xlim(-1,200)
plt.ylim(0, 1000)
# plt.show()

li_fluxes_inverse = inverseft(c2_dft_interp)

print("fluxes for inverse fourier transform ", len(li_fluxes_inverse), len(k), len(times_epoch))

#use the real numpy built in fuction to only take the real part so that there the array is not complex
li_fluxes_invreal = np.real(li_fluxes_inverse)
print("real fluxes for inverse fourier transform ", len(fluxes_invreal))

plt.plot(merged_times, li_fluxes_invreal, 'o')
plt.plot(times_epoch, fluxes_epoch, "o--")
# plt.show()

#PLOT SAME 4 GRAPHS BUT USING THE LI FFT
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#LI original time series
axs[0, 0].plot(merged_times,merged_fluxes_li, 'o--', color="blue")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Flux")
axs[0, 0].set_title("TIC 120016: Epoch with Linear Interpolation")

#LI power spectrum
axs[0, 1].plot(k2, abs_c2, color="green")
axs[0, 1].set_xlabel("k")
axs[0, 1].set_ylabel(r"$|c|^2$")
axs[0, 1].set_title("Power Spectrum")
axs[0, 1].set_ylim(0, 1000)

#LI IFFT
axs[1, 0].plot(merged_times, merged_fluxes_li, 'o--', color="red")
axs[1, 0].set_xlabel("time")
axs[1, 0].set_ylabel("Flux")
axs[1, 0].set_title("Inverse LI Fourier Transform")

#Original LI vs LI IFFT
axs[1, 1].plot(merged_times, merged_fluxes_li, 'o',label="Original", color="blue")
axs[1, 1].plot(merged_times, li_fluxes_invreal, '--', label="Reconstructed (Inverse FFT)", color="red")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Flux")
axs[1, 1].set_title("Original vs Inverse LI FFT")
axs[1, 1].legend()

fig.suptitle('TIC 120016: 2442-244 Epoch with Linear Interpolation', fontsize=16)
plt.show()

"""Let's try in different section of the data series"""

mask_e2 = (times >=1600) & (times<=1608)


times_epoch2= times[mask_e2]
fluxes_epoch2 = fluxes[mask_e2]
ferrs_epoch2= ferrs[mask_e2]

epochc2= dft(fluxes_epoch2)

abs_epochc2 = np.abs(epochc2)**2


k_epoch2 = np.arange(len(epochc2))

fluxes_inverse_epochc2 = inverseft(epochc2)

# print("Inverse ft fluxes", fluxes_inverse_epochc2)

#use the real numpy built in fuction to only take the real part so that there the array is not complex
fluxes_invreal_epochc2 = np.real(fluxes_inverse_epochc2)

print(len(times_epoch2), len(fluxes_invreal_epochc2))
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#Original time series
axs[0, 0].plot(times_epoch2, fluxes_epoch2, color="blue")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Flux")
axs[0, 0].set_title("TIC 120016: 1600-1608 Epoch")

# Power Spectrum
axs[0, 1].plot(k_epoch2, abs_epochc2, color="green")
axs[0, 1].set_xlabel("k")
axs[0, 1].set_ylabel(r"$|c|^2$")
axs[0, 1].set_title("Power Spectrum")
axs[0, 1].set_ylim(0, 1000)

#Inverse FFT
axs[1, 0].plot(times_epoch2, fluxes_invreal_epochc2, color="red")
axs[1, 0].set_xlabel("time")
axs[1, 0].set_ylabel("Flux")
axs[1, 0].set_title("Inverse Fourier Transform")

# Original vs FFT
axs[1, 1].plot(times_epoch2, fluxes_epoch2, label="Original", color="blue")
axs[1, 1].plot(times_epoch2, fluxes_invreal_epochc2, '--', label="Reconstructed (Inverse FFT)", color="red")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Flux")
axs[1, 1].set_title("Original vs Inverse FFT")
axs[1, 1].legend()

fig.suptitle('TIC 120016: 1600- 1608 Epoch', fontsize=16)
plt.tight_layout()
plt.show()

mask_e3 = (times >=3084) & (times<=3090)


times_epoch3= times[mask_e3]
fluxes_epoch3 = fluxes[mask_e3]
ferrs_epoch3= ferrs[mask_e3]

epochc3= dft(fluxes_epoch3)

abs_epochc3 = np.abs(epochc3)**2


k_epoch3 = np.arange(len(epochc3))

fluxes_inverse_epochc3 = inverseft(epochc3)

# print("Inverse ft fluxes", fluxes_inverse_epochc2)

#use the real numpy built in fuction to only take the real part so that there the array is not complex
fluxes_invreal_epochc3 = np.real(fluxes_inverse_epochc3)

print(len(times_epoch3), len(fluxes_invreal_epochc3))
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#Original time series
axs[0, 0].plot(times_epoch3, fluxes_epoch3, color="blue")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Flux")
axs[0, 0].set_title("TIC 120016: 3084 - 3090 Epoch")

# Power Spectrum
axs[0, 1].plot(k_epoch3, abs_epochc3, color="green")
axs[0, 1].set_xlabel("k")
axs[0, 1].set_ylabel(r"$|c|^2$")
axs[0, 1].set_title("Power Spectrum")
axs[0, 1].set_ylim(0, 1000)

#Inverse FFT
axs[1, 0].plot(times_epoch3, fluxes_invreal_epochc3, color="red")
axs[1, 0].set_xlabel("time")
axs[1, 0].set_ylabel("Flux")
axs[1, 0].set_title("Inverse Fourier Transform")

# Original vs FFT
axs[1, 1].plot(times_epoch3, fluxes_epoch3, label="Original", color="blue")
axs[1, 1].plot(times_epoch3, fluxes_invreal_epochc3, '--', label="Reconstructed (Inverse FFT)", color="red")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Flux")
axs[1, 1].set_title("Original vs Inverse FFT")
axs[1, 1].legend()

fig.suptitle('TIC 120016: 3084 - 3090 Epoch', fontsize=16)
plt.tight_layout()
plt.show()