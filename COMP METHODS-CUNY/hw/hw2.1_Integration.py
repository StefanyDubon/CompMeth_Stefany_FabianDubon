import numpy as np
import matplotlib.pyplot as plt
import math 
import argparse

"""HW.2 

    E(x) = int 0 to x e^(-t^2)dt 

A) Write a program to calculate E(x) for values of x from 0 to 3 
in steps of 0.1. Choose for yourself what method you will use for 
performing the integral and a suitable number of slices.

b) When you are convinced your program is working, extend it 
further to make a graph of E(x) as a function of x

Note that there is no known way to perform this particular integral 
analytically, so numerical approaches are the only way forward.


"""
##USING TRAPEZOIDAL RULE
def E(t):     #This is the a defined function to integrate
    return np.exp(-t**2)

def Trapezoidal(a, b, N):
    # N=1000 #number of slices 
    # a= 0    #lower limit of integration
    # b=3     #higher limit of integration s  


    if a > b:
        raise ValueError(" 'a' must be less than 'b' ")

    x_values = np.arange(0,3.1,0.1) # a=0, bb= 3.1 to be inclusive and in 0.1 step size
    E_values = []
    for x in x_values:   ##double for loop so it will take an x vaule from the x_value-array in 
                        #steps 0f 0.1 and do the integral 
        s = 0.5*E(a) +0.5*E(x)
        h = (x-a)/N 
        for k in range(1, N):
            s += E(a+k*h)
        val= h*s 
        E_values.append(val)


    print (f"X values: {x_values}")
    print (f"E(x) values: {E_values}")

    return x_values, E_values

##########################################################
#### New: Simpson's rule####
###################################################

def Simpsonrule(a, b, N):
    # N=1000 #number of slice
    # a = 0
    # b= 3

    if a > b :
        raise ValueError("a must be less than b")
    
    x1_vals = np.arange(0, 3.1, 0.1)
    simp_vals = []

    for x in x1_vals:
        s = (E(a)+ E(x))
        h = (x-a)/N
        for k in range(1, N//2+1):
            s += 4*(E(a+(2*k-1)*h))
        for j in range(1, N//2):
            s+= 2*(E(a+2*j*h))
        vals = (h/3)*s 
        simp_vals.append(vals)

    print (f"X values: {x1_vals}")
    print (f"E(x) values: {simp_vals}")

    return x1_vals, simp_vals



"""WE CAN ALSO DEFINE A FUNCTION OF THE TRAPEZOIDAL SO THAT 
A,B AND N ARE ARBITRARY NUMBERS """

def TrapRule(a,b,N):  #user gives values the upper (a), lower(b) and number of Slices
    s1 = 0.5*E(a) +0.5*E(b)
    h1 = (b-a)/N 
    for k in range (1, N):
        s1 += E(a+k*h1)
    return h1*s1 
###This is if we want to make the user input the arbitrary numbers
# since we are not doing in .1 steps, the answer will be a single number for this
# a1 = float(input("Enter lower limit (a): "))
# b1 = float(input("Enter higher limit (b): "))
# N1 = int(input("Enter number of slices (N): "))
# e = TrapRule(a1, b1, N1)

#############
##New
###########

##Make plots of E(x) as a function of x
def plot(a, b, N, make_plot= 'None'):

    if make_plot == 'Trapezoidal':
        fig, ax = plt.subplots(figsize =(6,8))
        x_values, E_values = Trapezoidal(a, b, N)
        ax.plot(x_values, E_values, color= 'red')
        ax.set_xlabel(" x ")
        ax.set_ylabel(" E(x)")
        ax.set_title("Plot of E(x) as a function of x using Trapezoidal rule")
        plt.show()
    
    elif make_plot == "Simpsons":
        fig, ax = plt.subplots(figsize =(6,8))
        x1_vals, simp_vals = Simpsonrule(a, b, N)
        ax.plot(x1_vals, simp_vals, color= 'blue')
        ax.set_xlabel(" x ")
        ax.set_ylabel(" E(x)")
        ax.set_title("Plot of E(x) as a function of x using Simpsons rule")
        plt.show()

    elif make_plot == "Both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Call functions to get data
        x_values, E_values = Trapezoidal(a, b, N)
        x1_vals, simp_vals = Simpsonrule(a, b, N)

        ax1.plot(x_values, E_values, 'r-', linewidth=2, label='E(x) using Trapezoidal Rule')
        ax1.set_xlabel('x')  # Note: set_xlabel, not xlabel
        ax1.set_ylabel('E(x)')  # Note: set_ylabel, not ylabel  
        ax1.set_title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')  # Note: set_title, not title
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(x1_vals, simp_vals, 'b-', linewidth=2, label='E(x) using Simpsons Rule')
        ax2.set_xlabel('x')
        ax2.set_ylabel('E(x)')
        ax2.set_title('Error Function: E(x) = ∫₀ˣ e^(-t²) dt')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.show()




    # x_values, E_values = Trapezoidal(a1, b1, N1)
    # plt.plot(x_values, E_values, color='g')
    # plt.xlabel("x")
    # plt.ylabel("E(x)")
    # plt.title("E(x) as a function of x for E(x)= e^(-t^2) from 0 to 3")
    # plt.grid()
    # plt.show()


if __name__ == "__main__":   #new
    #set up the arguments with argsparse
    parser = argparse.ArgumentParser(description= "Do Something")
    parser.add_argument('a', type =float, help=' We will calculate E(x) fro values a (input 0)')
    parser.add_argument('b', type = float, help= 'We will calculate E(x)for values until b (input 3)')
    parser.add_argument('N', type = int, help=' Choose the numeber of slices')
    parser.add_argument('choose_method', type = str, default='Trapezoid', help='String that controls which method to use to calculate E(x) --- insert Trapezoid or Simpsons (Default is Trapezoid) ')
    parser.add_argument('--make_plot', type=str, default='None', help = 'Insert Trapezoid, Simpsons or Both to plot E(x) as a function of x using Trapezoid method, Simpsoms method, or plot both to compare' )

    args = parser.parse_args()

    #define args
    a = args.a
    b = args.b
    N = args.N
    choose_method = args.choose_method
    make_plot = args.make_plot

    if choose_method == 'Trapezoidal':
        x_values, E_values = Trapezoidal(a, b, N)
    elif choose_method == "Simpsons":
        x1_vals, simop_vals = Simpsonrule(a, b, N)

    #make plot if requested 
    if make_plot != 'None':
        plot(a, b, N, make_plot)


"""
UPDATED: I added another method, and made different plots so they can be compared 
PS: I did use aI in the part of two loops, becuase I was getting the 
same value for all x values, and so I asked what the problem was, and it was cuz
I hard coded the b value (3) instead of doing x in x_values which is 0-3.1 in 0.1 steps. """