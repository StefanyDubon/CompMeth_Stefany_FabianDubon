import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random 
import matplotlib.animation as animation

"""Track a photon in a slab """


def track_photon(n_photons, slab_width, mean_free_path):
    rng = np.random.default_rng()
    # start = 0.0    #intial position of the photon
    # r = 0.0 #intital distance traveled
    path = []
    albedo = 0.9 #probability thatv a photon is scattered at an interaction. 
                    #leaves opportunity for the photon to be absorbed

    max_steps = 1000
    n_photons= n_photons 

    transmitted = 0.0
    reflected= 0.0 
    absorbed = 0
    scatter_times = []
    all_paths= []

    # the exponential distribution: P(r) = 1/l*e^(-r/l), here x is going to be the distance
    #to draw a random distance we can rearange that equatio to: x = l*ln(1-R)


    first_angle = random.uniform(np.pi/2, np.pi)
    

    for i in range(n_photons):

        x, y = 0.0, 0.0    #intial position in an x,y plane
        path_x= [x]
        path_y= [y]
        n_scatter = 0.0

        is_first_step = True

        for step in range(max_steps):
            #random distamce and direction
            r = -mean_free_path *np.log(1 - rng.random())

            if is_first_step:     #so that the first photon can go forward

                theta = rng.uniform(-np.pi/2, np.pi/2)
                is_first_step = False

            else: 
                theta =  2*np.pi*rng.random()

            x_new= x + r*np.cos(theta)
            y_new= y+r*np.sin(theta)


            #Check and keep count on the number of photons that transmitted or scattered or absorb

            #if the photon transmitted
            if x_new > slab_width:
                transmitted += 1
                path_x.append(x_new)
                path_y.append(y_new)
                break

            #if the photon got reflected
            if x_new < 0:
                reflected += 1
                path_x.append(x_new)
                path_y.append(y_new)
                break

            #if the photon is still inside the slan 
            n_scatter += 1

            x, y = x_new, y_new
            path_x.append(x)
            path_y.append(y)

            #ramdomly deide on absorption or scatterimg 
            if rng.random() < (1.0 - albedo):
                absorbed += 1
                break

        else:
            absorbed += 1

        scatter_times.append(n_scatter)
        all_paths.append([path_x.copy(), path_y.copy()])

#         return scatter_times, path_x, path_y
# # Results
    print("Simulated photons:", n_photons)

    print(transmitted)
    print(f"Transmitted: {transmitted} photons")
    print(f"Reflected: {reflected} photons")
    print(f"Absorbed: {absorbed} photons")
    print(f"Mean number of interactions (for simulated photons): {np.mean(scatter_times):.2f}")

    return scatter_times, path_x, path_y, transmitted, reflected, absorbed, all_paths
                
            
                
            
        
"""ANIMATION FOR PHOTONS GOING THROUGH THE SLAB"""

# Run your simulation
scatter_times, path_x, path_y, transmitted, reflected, absorbed, all_paths = \
    track_photon(n_photons=10000, slab_width=1000, mean_free_path=150)

# Get first 5 photons
n_photons_to_animate = 20
paths_to_animate = all_paths[:n_photons_to_animate]

# Find plot limits
all_x = [x for path_x, path_y in paths_to_animate for x in path_x]
all_y = [y for path_x, path_y in paths_to_animate for y in path_y]

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(xlim=(min(all_x)-200, max(all_x)+200), 
              ylim=(min(all_y)-100, max(all_y)+100))
ax.set_xlabel('Position (m)')
ax.set_ylabel('Position (m)')
ax.grid(True, alpha=0.3)

# Draw slab boundaries
ax.axvline(x=0, color='black', linewidth=2, label = "slab entrance")
ax.axvline(x=1000, color='black', linewidth=2, label = "slab exit")
ax.legend()

# Create 5 photons with different colors
colors = plt.cm.rainbow(np.linspace(0, 1, n_photons_to_animate))
photons = []
trails = []

for color in colors:
    photon = plt.Circle((0, 0), 20, fc=color)
    trail, = ax.plot([], [], color=color, linewidth=1, alpha=0.5)
    photons.append(photon)
    trails.append(trail)

# Find longest path
max_steps = max(len(px) for px, py in paths_to_animate)

def init():
    for photon, (px, py) in zip(photons, paths_to_animate):
        photon.center = (px[0], py[0])
        ax.add_patch(photon)
    for trail in trails:
        trail.set_data([], [])
    return photons + trails

def animate(i):
    for photon, trail, (px, py) in zip(photons, trails, paths_to_animate):
        if i < len(px):
            photon.center = (px[i], py[i])
            trail.set_data(px[:i+1], py[:i+1])
        else:
            photon.center = (px[-1], py[-1])
            trail.set_data(px, py)
    return photons + trails

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                              frames=max_steps, interval=100, blit=True)

# Save as gif
writergif = animation.PillowWriter(fps=10)
anim.save('photon_paths.gif', writer=writergif)

plt.show()
    
"""Now do it with the sun"""
def photon_around_sun(max_steps):

    rng = np.random.default_rng()

    sigma =6.652*(10**-25)  
    R_sun = 6.96*(10**10) #solar radius
    c = 3*10**10  #speed of light
    r= 0.0  #distance from the sun
    s= 0.0  #total distance traveled by the atom 

    n_scatter = 0


    for step in range(max_steps):
        
        
        #First calculate the Sun's electron density as a afunction of radius
        n = 2.5*(10**26)*np.exp(-r/(0.096*R_sun))
    
        #now the mean free path for the sun
        l_sun = 1/(n * sigma )
        # print(l_sun)

        #draw random distance from the exponential distribution withe the scale l_sun
        distance = -l_sun *np.log(1 - rng.random())
        
        #next update the total path length 
        s += distance

        #0 to 180 angle the photon was scatter
        theta = np.acos(2*rng.random()-1) 

        #now we have to update r using the law of cosines since the photon isn't always moving radialt 

        r_new = np.sqrt(r**2+distance**2 +2*r*distance*np.cos(theta))

        r= r_new 

        
        # Finally check if the photon escaped
        if r >= R_sun:
            time_years = (s / c) / (365.25 * 24 * 3600)
            print(f"Photon escaped!")
            print(f"Total distance s: {s:.2e} cm")
            print(f"Final radius r: {r:.2e} cm")
            print(f"Time to escape: {time_years:.1f} years")
            print(f"Number of scatters: {n_scatter}")
            break
        
   
        n_scatter += 1
        

        
        #now the loop repeats so it'll get the new n and updated r
    
    else:
        print(f"Photon did not escape within {max_steps} steps")
    print(r, s, n_scatter)
    return r, s, n_scatter

# Run the simulation
sun_photon = photon_around_sun(1000000)

    

