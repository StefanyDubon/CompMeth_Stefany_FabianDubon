import argparse
import math 
import numpy as np
import sys
import matplotlib.pyplot as plt

def determine_time( h, g = 9.81):

    """This function will determine the time it takes a ball to hit the ground
    with a given heigght (h) and g being a keyword using this formula:
        h = 1/2gt^2
    
    """
    
    t= np.sqrt((2*h)/g)


    return t

"""This part was for when we were using sys.argv"""

def plot_hxt(h, g=9.81, pltshow = True):

    """Plots: Given an array of height values with the given value in the center, the different times it would take the ball to hit the ground depending on the height"""

    height = np.linspace(0, h*2, 100)



    times = []

    for h_val in height:
        time = determine_time(h_val, g)
        times.append(time)
    if pltshow:
        plt.plot(height, times)
        plt.xlabel("Height (m)")
        plt.ylabel("time (s)")
        plt.title("Time vs Height)")
        plt.show()
    return times, height
  

# h = float(sys.argv[1])
# print(func(h))
# # print(func(4))
# print(sys.argv[1])

if __name__ == "__main__":
#set up the argsparse Arguments 
    parser = argparse.ArgumentParser(description="Do Something")
    parser.add_argument('h', type=float, help='This is the height of object thrown above ground')
    parser.add_argument("--grav", type=float, default= 9.8, help="Earth's gravitational acceleration")
    parser.add_argument("--make_plot", action = "store_true", default = None, help = "This argument determines if you want to make a plot showing time as a function of height. To make this plot enter as True ")
    args = parser.parse_args()

    h = args.h
    grav = args.grav
    make_plot = args.make_plot

    if h>0:

        print(f"{(determine_time(h, grav)):.2f}") #f' and 2.f is to determine the decimal points
    
    else:
        raise ValueError("The height value needs to be bigger than zero")

    if make_plot: 
        plots = plot_hxt(h, grav)
    else:
        raise ValueError("You have to provide the argument: --make_plot in order to plot different times it would take a ball to hit the ground as a function of a range of different height values(the one you give will be the center)")