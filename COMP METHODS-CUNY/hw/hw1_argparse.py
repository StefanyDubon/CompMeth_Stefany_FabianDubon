import argparse
import math 
import numpy as np
import sys

def determine_time( h, g = 9.81):

    """This function will determine the time it takes a ball to hit the ground
    with a given heigght (h) and g being a keyword using this formula:
        h = 1/2gt^2
    
    """
    
    t= np.sqrt((2*h)/g)


    return t

"""This part was for when we were using sys.argv"""
# h = float(sys.argv[1])
# print(func(h))
# # print(func(4))
# print(sys.argv[1])


parser = argparse.ArgumentParser(description="Do Something")
parser.add_argument('h', type=float, help='This is the height of object thrown above ground')
parser.add_argument("--grav", type=float, default= 9.8, help="Earth's gravitational acceleration")
args = parser.parse_args()

print(f"{(determine_time(args.h, args.grav)):.2f}") #f' and 2.f is to determine the decimal points