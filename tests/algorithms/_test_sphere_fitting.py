# coding=utf-8

"""scikit-surgerycalibration sphere-fitting tests"""

import numpy
from sksurgerycalibration.algorithms import sphere_fitting

def test_fit_sphere_least_squares():
    """Test that it fits a sphere with some gaussian noise"""
    x_centre = 1.74839
    y_centre = 167.0899222
    z_centre = 200.738829

    radius = 7.543589

    #some arrays to fit data to
    coordinates = numpy.ndarray(shape=(1000, 3), dtype=float)

    #fill the arrays with points uniformly spread on
    #a sphere centred at x,y,z with radius radius
    #first seed the random generator so we get consistent test behaviour
    numpy.random.seed(seed=0)
    for i in range(1000):
        #make a random vector
        x_vector = numpy.random.uniform(-1.0, 1.0)
        y_vector = numpy.random.uniform(-1.0, 1.0)
        z_vector = numpy.random.uniform(-1.0, 1.0)

        #scale it to length radius
        length = numpy.sqrt((x_vector)**2 + (y_vector)**2 + (z_vector)**2)
        factor = radius / length

        coordinates[i, 0] = x_vector*factor + x_centre
        coordinates[i, 1] = y_vector*factor + y_centre
        coordinates[i, 2] = z_vector*factor + z_centre

    parameters = [0.0, 0.0, 0.0, 0.0]
    result = sphere_fitting.fit_sphere_least_squares(coordinates, parameters)
    numpy.testing.assert_approx_equal(result.x[0], x_centre, significant=10)
    numpy.testing.assert_approx_equal(result.x[1], y_centre, significant=10)
    numpy.testing.assert_approx_equal(result.x[2], z_centre, significant=10)
    numpy.testing.assert_approx_equal(result.x[3], radius, significant=10)
