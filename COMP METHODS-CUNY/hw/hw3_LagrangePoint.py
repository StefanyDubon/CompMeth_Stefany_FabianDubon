import numpy as np
import math
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u

""" Write a program that uses either Newtons method or the secant method to solve for the distance r 
from the Earth to the L1 point. Compute a solution accurate to at least four significant figures.
Distance r from the center of the Earth to the L1 point: GM/r^2- Gm/(R-r)^2 = w^2r """

def f(r):      #distance r from the center of the Earth to the L1 function
    
    G= const.G   #Newton's gravitational constant 
    M_earth= const.M_earth   #Earth's mass
    m_moon = 7.348e22 * u.kg     #moon mass
    R= 3.844e8 * u.m          #earth-moon distance
    w = 2.662e-6 * (1/u.s)    #omega's value

    L1 =((G*M_earth)/((r*u.m)**2)) - ((G*m_moon)/((R-(r*u.m))**2)) - (w**2)*(r*u.m) 

    return L1.value


""" For this part, I tried doing the Newton's method without having to do the derivative analytically, intead I tried 
using the fact that f'(x) = f(x)/delta_x, with delta being a chosen value of steps.
I then tried to find the new x value by using this formula
            x' = x-delta_x = """
#I'm not sure what I'm missing but I don't think that the value I'm getting is right
#and after discussing what other classmates did, I've decide to just compute the derivative
#analytically and seing if that works. 
#######PS:I'm leaving this here so that it shows I tried to solve it in different ways
# def Newton_meth(r0, tol= 1e-10, max_iter= 1000):
#     r= r0
#     delta_r = 0.1
#     num_iter = 0
#     for i in range(max_iter):
#         r_d = f(r)/delta_r
#         r_new = r- r_d
#         if abs(r_new - r) < tol:
#             num_iter+= 1
#             return r_new  # return only when converged
#         r = r_new
#         # print(r)
#     return r
# r_val = Newton_meth(0.5)
# print("r", r_val)


##Derivative of f(r)
def f_deriv(r):
    G= const.G   #Newton's gravitational constant 
    M_earth= const.M_earth   #Earth's mass
    m_moon = 7.348e22 * u.kg     #moon mass
    R= 3.844e8 * u.m          #earth-moon distance
    w = 2.662e-6 * (1/u.s) #omega's value

    L_deriv =((-2*G*M_earth)/((r*u.m)**3)) - ((2*G*m_moon)/((R-(r*u.m))**3)) - (w**2) 

    return L_deriv.value

"""Use Newton method so that the new guess for x' would be
                x'= x-delta_x = - (x)/f'x)"""
def Newton_meth(r0, tol= 1e-8,  max_iter = 10000):
    r= r0
    for i in range(max_iter):
        lag_func = f(r)
        deriv_func = f_deriv(r)
        delta_r = lag_func/deriv_func
        r_new= r - delta_r
        if abs(r_new - r) < tol:
            return r_new  # return only when converged
        r = r_new
     
    return r
r_val = Newton_meth(3e8)
print(f"The distance r from from the Earth to the L1 point is: {r_val:.4f} m")


###Plot the function, so that we can estimate/guess the starting value of r
x_vls= np.linspace(1e8, 3.8e8, 1000)
r_values = []

for r in x_vls:
    r_values.append(f(r))

##We can make it so that the user will see the plot first, and then input the starting r value
x_vls= np.linspace(1e8, 3.8e8, 1000)
r_values = []

for r in x_vls:
    r_values.append(f(r))

plt.plot(x_vls, r_values, color= "pink", label = " $L_1$ Lagrange Function")
plt.title("$L_1$ Newton's Method" )
plt.xlabel(" r (m)")
plt.ylabel("f(r)")
# plt.scatter(r_val, 0, s = 50, color = 'blue', label =" r ")
plt.grid()
plt.legend(loc= "upper right")
plt.show()

#raise an error message if the user inputs a number outside what's allow, or a string
R= 3.844e8 * u.m 
try: 
    r_guess = float(input("Pleaser input the starting r value: "))
    if r_guess <=0 or r_guess >=R.value: 
        raise ValueError(f"r must be betweenn 0 and {R:.2e} ")
except ValueError as e:
    print("Invalud input:", e)
    exit(1)

r_val2 = Newton_meth(r_guess)
print(f"The distance r from from the Earth to the L1 point is: {r_val2:.4f} m")

