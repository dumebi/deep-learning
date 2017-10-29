from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #iniialize error a 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        #get the x value
        x = points[i, 0]
        # get the y value
        y = points[i, 0]
        #get the difference, square i, ad it to the total
        totalError += (y - (m * x + b)) ** 2

    #get the average
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iiterations):
    b = starting_b
    m = starting_m
    #perform gradient descent
    for i in range(num_iiterations):
        #update b and m with new accurate b and m by performing
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b,m]

def step_gradient(current_b, current_m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        #starting points for our gradients
        x = points[i, 0]
        y = points[i, 1]
        #directions with respect to b and m
        #computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2 / N) * x * (y - ((current_m * x) + current_b))

    #update our b and m values using this partial derivatives
    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)

    return [new_b, new_m]


def run():
    #step 1 - Collect data
    points = genfromtxt('data.csv', delimiter=',')

    #step 2 - Define hyperparameters

    #how fast should our model converge
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    #step 3 - train model
    print 'stating gadient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print 'running'

    print 'ending gadient descent after {0} iterations  at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == "__main__":
    run()