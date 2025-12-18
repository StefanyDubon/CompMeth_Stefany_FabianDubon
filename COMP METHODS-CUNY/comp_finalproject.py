"""
    Main idea of the Project: Use the titlted ring model to create 2D velocity
    field fields to simulate the rortation curve of a spiral galaxy. 
    First, create a mock spiral galaxy, then use the model with 
    MCMC to find the best fit parameters and their uncertainties.
"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math 
import scipy
import emcee
from scipy.stats import binned_statistic
from numpy import cos, sin, arctan2
from astropy.io import fits
from matplotlib.patches import Ellipse
import corner

"""
    Tilted-ring model.
This model is based on the assumption that the emitting material is confined to 
a thin disc and that the kinematics is dominated by the rotational motion, so 
that each ring has a constant circular velocity Vrot(R), depending only on 
the distance R from the centre. 
In the model the disc is therefore broken down into a number of concentric rings 
with different radii, inclinations, positions angles and rotation velocities 

To do the fitting of 2D velocity fields, I used the formula
Vlos(R)= Vsys + Vrot(R) cos θsin i, where Vsys is
the systemic velocity and θis the azimuthal angle, measured in the
plane of the galaxy (θ= 0 for major axis), related to the inclination i
and the position angle φ, measured on the plane of the sky
"""

############# SET PARAMETERS ###############
rng = np.random.default_rng()


##################################################################################################################
        ## Main parameters used # 
        #inclination = the angle of the disk relative to the line of sight
        #position angle = the angle of the galaxy's major axis n the plane of the sky
        #X,Y: sky coordinates of the rotation centre of the galaxy
        #theta = the azimuthal angle
        #Vrot = the circular velocity at distance R from the center
        #r = the distance from the center along the disk, for which vel are measured to build the rotation core
####################################################################################################################

#initialize number generator
# rng = default_rng()
np.random.seed(1)
#set initial parameters

#Galaxy parameters
inc = np.deg2rad(60.0) #radians -At 90 edge-on, 0 face on
pa = np.deg2rad(30.0)   #radians -At 0 full approaching vel, 90 (side) no radial vel, 180 (receding)full receding vel
vsys = 1000.0          #systematic vel km/s

#Rotation curve parameters
vmax = 200.0          #maximum rotation velocity km/s
r0 = 5.0              #Scale radius (kpc

#obs parameters
n_points = 4000        #number of observed points
sigma_v = 10.0         #vel uncertainty km/s
r_max = 15.0           #max radius kpc

#Ring binning parameters
n_rings= 10            #number of radial rings
r_bins = np.linspace(0, r_max, n_rings+1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

# MCMC parameters
n_walkers = 40              # Number of MCMC walkers
n_steps = 20000              # Number of MCMC steps per walker
n_burn = 1000               # Burn-in steps to discard
thin = 10                   # Thinning factor

print("="*70)
print("TILTED-RING GALAXY ROTATION CURVE MODELING WITH MCMC")
print("="*70)
print(f"\nConfiguration:")
print(f"  Number of points: {n_points}")
print(f"  Number of rings: {n_rings}")
print(f"  Velocity noise: {sigma_v} km/s")
print(f"  MCMC walkers: {n_walkers}")
print(f"  MCMC steps: {n_steps}")



##############################################################################
            #Get rotation curve#
##############################################################################

def rotation_curve(r):
    #rotation curve
    return vmax *(1- np.exp(-r/r0))

##############################################################################
            #Generate mock galaxy#
##############################################################################

print("\nGenerating mock galaxy observations...")

#random sky-plane coord
x = rng.normal(-r_max, r_max, n_points)
y = rng.normal(-r_max, r_max, n_points)
print("x",x)
#Rotate by the position angle 
x_pa = x * cos(pa) + y * sin(pa)
y_pa = -x * sin(pa)+ y * cos(pa)
print(x_pa)

#Deproject to Galaxy Plane
r_gal= np.sqrt( x_pa**2 + (y_pa / cos(inc))**2)
theta_gal = arctan2( y_pa / cos(inc), x_pa)
# print("r_gal", r_gal)

#filter points within max radius
mask_r = r_gal < r_max
# print(mask_r)
r_gal = r_gal[mask_r]
theta_gal = theta_gal[mask_r]
x_pa = x_pa[mask_r]
y_pa = y_pa[mask_r]

n_points=len(r_gal)

#compute the Vlos vel using the Vlos = Vsys +Vrot *cos(theta)*sin(inc)
vrot = rotation_curve(r_gal)
vlos = vsys + vrot *cos(theta_gal)*sin(inc)

#add noise to the mock data so it can be "realistic"
vlos_noise = vlos + np.random.normal(0.0, sigma_v, n_points)
print(len(vlos_noise), len(vlos))
print(f"Generated {n_points} valid observations")

##############################################################################
            #Put bin dta into rings#
##############################################################################

#assing each point to a ring 
ring_index = np.digitize(r_gal, r_bins) -1
ring_index = np.clip(ring_index, 0, n_rings-1)


#now calculate points per ring for the diagnostics (mcmc)
# for i in range(n_rings):
#     points_per_ring = np.array([np.sum(ring_index == i)])
#     points_per_ring += points_per_ring
# print(f"\nPoints per ring: min={np.min(points_per_ring)}, "
#     f"max={np.max(points_per_ring)}, mean={points_per_ring.mean():.1f}")

points_per_ring = np.array([np.sum(ring_index == i) for i in range(n_rings)]) #vectorize it to make it faster
print(f"\nPoints per ring: min={points_per_ring.min()}, "
      f"max={points_per_ring.max()}, mean={points_per_ring.mean():.1f}")

##############################################################################
            #Tilted-ring model#
##############################################################################

def vlos_model(vrot_params, r, theta, ring_idx):
    
    """
         MAIN GOAL: Compute the line of sight velocity 
         using tilted-rind model
    """

    v_model = np.full_like(r, vsys)


    for i in range(len(vrot_params)):
        mask = ring_idx ==i
        if np.any(mask):
            v_model[mask] = vsys + vrot_params[i]*cos(theta[mask])*sin(inc)

    return v_model

##############################################################################
#Define the likelihood, prior, and posterior to later run them wiith MCMC#
##############################################################################

def log_likelihood(vrot_params, r, theta, ring_idx, v_obs, sigma):

    """
    This is the lof likelihood assuming Gaussian errors
    """
    v_model = vlos_model(vrot_params, r, theta, ring_idx) #get the line of sight vel model
    residuals  = v_obs - v_model

    chi_squared = np.sum((residuals/sigma)**2)

    #include normalization constant
    log_l = -0.5 *(chi_squared + n_points * np.log(2 * np.pi * sigma**2))

    return log_l

def log_prior(vrot_params):
    """
        uniform prior with physical bounds. The rotation velocities must be between 0 and 500 km//s
    """

    if np.any(vrot_params < 0) or np.any(vrot_params > 400):
        return -np.inf
    return 0.0
        
def log_probability(vrot_params, r, theta, ring_idx, v_obs, sigma):
    """
         this is the log posterior probabilty ==> log prior +log likelihood
         
    """
    lp = log_prior(vrot_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(vrot_params, r, theta, ring_idx, v_obs, sigma)


##############################################################################
            #  Run MCMC with emcee #
##############################################################################

print("\n" + "="*70)
print("RUNNING MCMC")
print("="*70)

ndim = n_rings

#initialize walkers with small perturbations around reasonable guess
initial_guess = 150.0
initial_positions = initial_guess + 10.0 *np.random.randn(n_walkers, ndim)
initial_positions= np.clip(initial_positions, 0, 400)

#Create the sampler using emcee

sampler = emcee.EnsembleSampler( 
    n_walkers,
    ndim,
    log_probability,
    args = (r_gal, theta_gal, ring_index, vlos_noise, sigma_v)
)

#Run MCMC 
print(f"Running {n_steps} steps with {n_walkers} walkers...")
sampler.run_mcmc(initial_positions, n_steps, progress= True)
print("\nMCMC sampling complete!")

##############################################################################
        # Analyze the results #
##############################################################################

print("\n" + "="*70)
print("ANALYZING RESULTS")
print("="*70)

samples = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
print(f"Total samples after burn-in and thinning: {samples.shape[0]}")

#calculate statistics for each ring
vrot_median = np.median(samples, axis = 0)
vrot_mean = np.mean(samples, axis= 0)
vrot_std = np.std(samples, axis=0)
vrot_low = np.percentile(samples, 16, axis=0)
vrot_high = np.percentile(samples, 84, axis=0)

# values at ring centers
vrot_center = rotation_curve(r_centers)

#calculate chi-squared for the best fit
vmodel_best = vlos_model(vrot_median, r_gal, theta_gal, ring_index)
chi_squared = np.sum((vlos_noise - vmodel_best)**2 / sigma_v**2)
reduced_chi_squared = chi_squared / (n_points - ndim)

print(f"\nGoodness of fit:")
print(f"  chi-squared = {chi_squared:.1f}")
print(f"  Reduced chi-squared = {reduced_chi_squared:.3f}")
print(f"     How we can interpret it: ~1.0 = EXCELLENT FIT!")
print(f"                     <0.8 = might be overfitting")
print(f"                     >1.5 = poor fit, model issues")
print(f"  DOF = {n_points - ndim}")


# Print results for each ring
print(f"\n RECOVERED ROTATION VELOCITIES - KEY RESULT! ")
print(f"This shows how accurately MCMC recovered the true rotation curve:")
print(f"{'Ring':<6} {'R (kpc)':<10} {'True':<10} {'Median':<12} {'±1sigma':<15} {'Offset':<10}")
print("-" * 70)
for i in range(n_rings):
    offset = vrot_median[i] - vrot_center[i]
    offset_pct = 100 * offset / vrot_center[i]
    print(f"{i:<6} {r_centers[i]:<10.2f} {vrot_center[i]:<10.1f} "
          f"{vrot_median[i]:<12.1f} ±{vrot_std[i]:<14.1f} "
          f"{offset:>6.1f} ({offset_pct:>5.1f}%)")
# Calculate overall accuracy
mean_offset_pct = np.mean(np.abs(100 * (vrot_median -  vrot_center) / vrot_center))
print(f"\n AVERAGE RECOVERY ACCURACY: {mean_offset_pct:.1f}% ")


# Compute autocorrelation time for convergence diagnostics
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nAutocorrelation time:")
    print(f"  Mean: {np.mean(tau):.1f} steps")
    print(f"  Max: {np.max(tau):.1f} steps")
    print(f"  Effective samples per walker: {n_steps / np.mean(tau):.1f}")
except:
    print("\nWarning: Could not compute autocorrelation time")

# Acceptance fraction
acceptance = np.mean(sampler.acceptance_fraction)
print(f"\nMean acceptance fraction: {acceptance:.3f}")
if acceptance < 0.2 or acceptance > 0.5:
    print("  Warning: Acceptance fraction outside optimal range [0.2, 0.5]")


####################################
        # Plots #
####################################

#plot 1: rotation curve. Probably the most important result/plot since it shows the recovered rotation curve with mcmc

fig1, ax = plt.subplots(figsize= (6,8)) 
ax.errorbar(
    r_centers,
    vrot_median,
    yerr = [vrot_median - vrot_low, vrot_high - vrot_median],
    fmt = 'o',
    color = 'blue',
    markersize = 8,
    capsize = 5,
    capthick = 2,
    label = "MCMC Recovered (68%)",
    zorder = 3
)
r_smooth = np.linspace(0, r_max, 200)  #
ax.plot(r_smooth, rotation_curve(r_smooth), 'r-', linewidth = 2, label="True Model", zorder = 2)
ax.scatter(r_centers, vrot_center, color = 'red', s=100, marker= 'x', linewidth = 3, label = 'True at Ring Centers', zorder= 4)
ax.set_xlabel("Radius (kpc)", fontsize = 12, fontweight = 'bold')
ax.set_ylabel("Rotation Velocity (km/s)", fontsize=12, fontweight='bold')
ax.set_title("Recovered Rotation Curve", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, r_max + 0.5)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('recovered_rotation_curve.png', dpi=300, bbox_inches='tight')


#plot 2: Observed velocity field
fig2, ax2 = plt.subplots(figsize=(6,8))
scatter = ax2.scatter(x_pa, y_pa, c=vlos_noise, s=1, cmap = 'RdBu_r', vmin= vsys-250, vmax= vsys+250, alpha= 0.5) 
ax2.set_xlabel("X (kpc)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Y (kpc)", fontsize=12, fontweight='bold')
ax2.set_title("Observed Velocity Field", fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('V_los (km/s)', fontsize=11)
ax2.grid(True, alpha=0.3)
plt.savefig('observed_vel_field.png', dpi=300, bbox_inches='tight')

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 8))

# Plot 3: the velocity field with the rings.  A visual representation of hot tilted-ring model works (10 concentric rings)
scatter = ax3.scatter(x_pa, y_pa, c=vlos_noise, s=10, 
                    cmap='RdBu_r', vmin=vsys-250, vmax=vsys+250, 
                    alpha=0.6, edgecolors='none')

# Overlay the rings
from matplotlib.patches import Ellipse

for i, r_edge in enumerate(r_bins[1:]):  # Skip r=0
    a = r_edge                  # semi-major axis
    b = r_edge * np.cos(inc)    # semi-minor axis (projection)

    ellipse = Ellipse(
        xy=(0, 0),
        width=2 * a,
        height=2 * b,
        angle=np.rad2deg(pa),
        edgecolor='black',
        facecolor='none',
        linewidth=1.5,
        linestyle='--',
        alpha=0.7
    )

    ax3.add_patch(ellipse)

    # Optional: label ring
    if i < len(r_centers):
        x_label = a * np.cos(pa)
        y_label = a * np.sin(pa)
        ax3.text(
            x_label,
            y_label,
            f'Ring {i}',
            fontsize=9,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )

    

# Add ring centers as well (optional)
for i, r_c in enumerate(r_centers):
    circle_center = plt.Circle((0, 0), r_c, color='gray', 
                              fill=False, linewidth=1, 
                              linestyle=':', alpha=0.5)
    ax3.add_patch(circle_center)

# Mark the center
ax3.plot(0, 0, 'k*', markersize=20, label='Galaxy Center')

# Add position angle line
pa_length = r_max * 0.8
ax3.plot([0, pa_length * np.cos(pa)], 
       [0, pa_length * np.sin(pa)], 
       'g-', linewidth=3, label=f'PA = {np.rad2deg(pa):.0f}°')

# Add major axis line (perpendicular to PA)
ax3.plot([0, pa_length * np.cos(pa + np.pi/2)], 
       [0, pa_length * np.sin(pa + np.pi/2)], 
       'orange', linewidth=2, linestyle='--', label='Minor Axis')

ax3.set_xlabel("X (kpc)", fontsize=14, fontweight='bold')
ax3.set_ylabel("Y (kpc)", fontsize=14, fontweight='bold')
ax3.set_title(f"Mock Galaxy Velocity Field with Ring Structure\n(i={np.rad2deg(inc):.0f}°, PA={np.rad2deg(pa):.0f}°)", 
            fontsize=16, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3, linestyle=':')
ax3.legend(loc='upper right', fontsize=11)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('V_los (km/s)', fontsize=12, fontweight='bold')
cbar.ax.axhline(vsys, color='yellow', linewidth=2, linestyle='--', label='V_sys')

# Set limits
ax3.set_xlim(-r_max - 1, r_max + 1)
ax3.set_ylim(-r_max - 1, r_max + 1)

plt.tight_layout()
plt.savefig('galaxy_velocity_field_with_rings.png', dpi=300, bbox_inches='tight')
print("Saved: galaxy_velocity_field_with_rings.png")



# Plot 4: RESIDUALS - Shows quality of recovery 
print("vrot median", vrot_median, "vrot centers", vrot_center) #includes noise (stdv = 10)
fig3, ax4 = plt.subplots(figsize=(6,8))
residuals = vrot_median - vrot_center
ax4.errorbar(
    r_centers,
    residuals,
    yerr=[vrot_std, vrot_std],
    fmt='o',
    color='purple',
    markersize=8,
    capsize=5,
    capthick=2
)
ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.fill_between(r_centers, -sigma_v, sigma_v, 
                  alpha=0.2, color='gray', label=f'±{sigma_v} km/s (noise level)')
ax4.set_xlabel("Radius (kpc)", fontsize=12, fontweight='bold')
ax4.set_ylabel("Residual (km/s)", fontsize=12, fontweight='bold')
ax4.set_title(" Recovery Residuals", 
             fontsize=14, fontweight='bold', color='darkgreen')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')


#Plot 5: Corner Plot
try:
    # Plot corner for first 6 rings 
    n_corner = min(6, n_rings)
    labels = [f'V_{i} ({r_centers[i]:.1f} kpc)' for i in range(n_corner)]
    
    fig5 = corner.corner(
        samples[:, :n_corner],
        labels=labels,
        truths=vrot_center[:n_corner],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.1f',
        color='steelblue',
        truth_color='red'
    )
    fig5.suptitle('Posterior Distributions (First 6 Rings)', 
                  fontsize=16, fontweight='bold', y=1.0)
    plt.savefig('corner_plot1.png', dpi=300, bbox_inches='tight')
    print("Saved: corner_plot1.png")
except Exception as e:
    print(f"Could not create corner plot: {e}")

plt.show()
    













