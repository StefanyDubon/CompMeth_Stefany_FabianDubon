import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

"""
HW6: Exerise 8.8 SPACE GARBAGE
We have a rod of mass M and length L
Equations of motion for the position (x,y) of the ball bearing in the xy plane
d^2x/dt^2 = -GM(x/r^2sqrt(r^2+1/4L^2)
d^2y/dt^2 = -GM(y/r^2sqrt(r^2+1/4L^2)
"""
#initial conditions:
G=1.0 #Gravitational constant 
M=10.0 #Mass of the rod
L=2.0 #Length of the rod

def deriv(state, t): 
    """
    state: [x, y, vx, vy]
    The four first-order equations returned are: 
    dx_dt--> velocity in the x-direction
    dy_dt--> velocity in the y-direction
    dvx_dt--> acceleration in the x-direction
    dvy_dt--> acceleration in the y-direction

    r--> the distance the ball is bearing bearing from the center of the rod
    
    """
    x, y, vx, vy =state

    #represent velocity
    dx_dt = vx
    dy_dt = vy


    r = np.sqrt(x**2+y**2) #calculate the distance 
    

    if r<1e-10:    #To avoid division by zero
        return np.array([vx, vy, 0, 0])
        
    #represent acceleration
    dvx_dt = -G*M*(x/(r**2*(np.sqrt(r**2+(L**2/4)))))
    dvy_dt = -G*M*(y/(r**2*(np.sqrt(r**2+(L**2/4)))))
    
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])
    
    

    
def fourth_order_runge_kutta(f, state, h, t ):

    """
    Perform single step of 4th order Runge-Kuta 
    """
    k1 = h*f(state, t)
    k2 = h*f(state+0.5*k1,t +0.5*h)
    k3 = h*f(state+0.5*k2,t +0.5*h)
    k4 = h*f(state+k3,t +h)

    return state +(k1+2*k2+2*k3+k4)/6

def solve_RK4(x0= 1.0 ,y0 = 0.0, vx0 = 0.0, vy0 = 1.0, t_start=0.0, t_end= 10.0, h=0.01):
    """
    Compute the RK4 using:
    # x0= 1.0  #inital condition for x
    # y0=0.0   #initial condition for y
    # vx0 = 0.0 #velocity in the x-direction
    # vy0 = 1.0 #velocity in the y-direction
    # t_start= 0
    # t_end = 10.0
    # h = 0.01  #time step 
    """
    state = np.array([x0, y0, vx0, vy0])
    #create empty lists
    x_vals = []
    y_vals = []
    times = []
    t= t_start
    while t <= t_end:
        x_vals.append(state[0])
        y_vals.append(state[1])
        times.append(t)

        state = fourth_order_runge_kutta(deriv, state, h, t)
        t +=h

    x_vals= np.array(x_vals)
    y_vals = np.array(y_vals)
    times= np.array(times)
    print(type(x_vals))
    
    print(f"Simulation complete! Generated {len(x_vals)} points from t={t_start} to t={t_end}")
    return x_vals, y_vals, times

# # Call the function
# x_vals, y_vals, times = solve_RK4()
# print(f"Returned {len(x_vals)} points")

def static_and_animate_plot(x_vals, y_vals, x0= 1.0 ,y0 = 0.0, M= 10.0, L=2.0): #I use cloude to debug this part because 
                                                                                #the end position was too far from the inital one( it was 80 in x direction))

    """
    Make the Static plot using Plotly 
    """

    fig_static = go.Figure()

    # Add the orbit trajectory
    fig_static.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Orbit',
        line=dict(color='green', width=2),
    ))

    # Add start point
    fig_static.add_trace(go.Scatter(
        x=[x0],
        y=[y0],
        mode='markers',
        name='Start',
        marker=dict(color='blue', size=12, symbol='circle'),
    ))

    # Add end point
    fig_static.add_trace(go.Scatter(
        x=[x_vals[-1]],
        y=[y_vals[-1]],
        mode='markers',
        name='End',
        marker=dict(color='pink', size=12, symbol='circle'),
    ))

    # Add the rod
    fig_static.add_trace(go.Scatter(
        x=[0, 0],
        y=[-L/2, L/2],
        mode='lines',
        name=f'Rod (M={M}, L={L})',
        line=dict(color='red', width=8),
    ))

    # Update layout
    fig_static.update_layout(
        title='Ball bearing orbiting around a rod in space',
        xaxis=dict(title='x', scaleanchor='y', scaleratio=1),
        yaxis=dict(title='y'),
        width=800,
        height=800,
        showlegend=True
    )

    fig_static.show()


    """
    Make the Animate Plot suing Plotly
    """
    
    x_a= x_vals 
    y_a = y_vals
    t_a = times

    #Make frames for animation
    frames =[]
    for i in range(len(x_a)):
        frame_data = [
            # Add start point
            go.Scatter(
            x=[x0],
            y=[y0],
            mode='markers',
            name='Start',
            marker=dict(color='blue', size=12, symbol='circle'),
            showlegend=True
            ),
    
            # # Add the orbit trajectory don't really need to since the trajectory will show it
            # go.Scatter(
            # x=x_vals,
            # y=y_vals,
            # mode='lines',
            # name='Orbit',
            # line=dict(color='green'),
            # showlegend= True
            # ),
            #the trail
            go.Scatter(
                x = x_a[:i+1],
                y = y_a[:i+1],
                mode = 'lines',
                name = 'Trail',
                line= dict(color= "purple", width = 2),
                showlegend= True
            ),
            #the current position of the ball bearing
            go.Scatter(
                x = [x_a[i]],
                y = [y_a[i]],
                mode = 'markers',
                name= 'Ball bearinf',
                marker = dict(color= 'green', size= 15, symbol= 'circle'),
                showlegend= True 
            ),
            #Add the rod
            go.Scatter(
                x= [0,0], 
                y= [-L/2, L/2],
                mode= 'lines',
                name=f'Rod (M={M}, L={L})',
                line=dict(color='red', width=8),
                showlegend= True
            )
        ]

        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title_text=f'Time: {t_a[i]:.2f} s'
            )
        ))

    # Create initial figure
    fig_animated = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Add play and pause buttons
    fig_animated.update_layout(
        title='Animated Ball bearing orbiting around a rod in space',
        xaxis=dict(
            title='x',
            range=[min(x_vals), max(x_vals)],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='y',
            range=[min(y_vals), max(y_vals)]
        ),
        width=800,
        height=800,
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ],
                x=0.1,
                y=0,
                xanchor='left',
                yanchor='top'
            )
        ],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{t_a[k]:.1f}',
                    'method': 'animate'
                }
                for k, f in enumerate(frames)
            ],
        }]
    )

    fig_animated.show()

    return fig_static, fig_animated


# static_plot()
if __name__== "__main__":

    x_vals, y_vals, times = solve_RK4()
    static_and_animate_plot(x_vals, y_vals)



        
    


    