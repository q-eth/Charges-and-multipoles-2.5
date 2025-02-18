
import numpy as np
import scipy.integrate as integrate

def load_data(filename):
    with open(filename, 'r') as f:
        data = [float(x) for x in f.read().split()]
    charge = {'q': data[0], 'x': data[1], 'y': data[2], 'z': data[3]}
    plane_z = data[4]
    circle = {'x': data[5], 'y': data[6], 'r': data[7]}
    return charge, plane_z, circle

def image_charge(charge, plane_z):
    q_img = -charge['q']
    z_img = 2 * plane_z - charge['z']
    return {'q': q_img, 'x': charge['x'], 'y': charge['y'], 'z': z_img}

def surface_charge_density(q_img, x_q, y_q, z_q, x, y):
    epsilon_0 = 8.85e-12
    r = np.sqrt((x - x_q) ** 2 + (y - y_q) ** 2 + z_q ** 2)
    E_n = q_img / (4 * np.pi * epsilon_0 * r ** 3) * z_q
    sigma = -epsilon_0 * E_n
    return sigma

def monte_carlo_integrate(q_img, x_q, y_q, z_q, circle, num_samples=100000):
    """ Численное интегрирование методом Монте-Карло """
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    r = np.sqrt(np.random.uniform(0, 1, num_samples)) * circle['r']

    x_samples = circle['x'] + r * np.cos(theta)
    y_samples = circle['y'] + r * np.sin(theta)
    sigma_samples = surface_charge_density(q_img, x_q, y_q, z_q, x_samples, y_samples)
    Q_ind = np.mean(sigma_samples) * (np.pi * circle['r'] ** 2)

    return Q_ind

def main():
    filename = 'input.txt'
    charge, plane_z, circle = load_data(filename)
    image = image_charge(charge, plane_z)
    print(f'Координаты заряда изображения: x = {image["x"]}, y = {image["y"]}, z = {image["z"]}')
    induced_charge = monte_carlo_integrate(image['q'], image['x'], image['y'], image['z'], circle)
    print(f'Суммарный индуцированный заряд в круге: {induced_charge:.4e} Кл')

if __name__ == "__main__":
    main()
