# Initialisation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# Radius data
radius_data1 = np.array([59.513027, 66.368389, 69.506166, 71.700852, 74.735925, 76.007795, 76.695256, 78.039551, 79.598386, 80.075493, 80.079624, 81.495363, 82.598934, 83.172805, 84.919440, 85.155809, 85.718415, 85.920147, 86.000243, 87.120934, 87.507943, 87.786892, 87.060409, 88.086562, 88.113767, 88.436448, 89.082172, 89.139957, 89.210538, 89.356928, 90.312937, 90.406557, 90.754907, 90.88941, 91.096467])
radius_data2 = np.array([59.810888, 65.360919, 69.088934, 72.594631, 74.694093, 76.440186, 78.133942, 78.618537, 79.753566, 80.745559, 81.422723, 81.634563, 82.861597, 84.335873, 85.362055, 85.537714, 86.159399, 86.873675, 86.918131, 87.003533, 87.126402, 87.155440, 87.224911, 87.394479, 87.433936, 87.626938, 87.701465, 87.834029, 87.963874, 88.153147, 88.209880, 88.542036, 88.86527, 89.082038, 89.214132])
radius_data3 = np.array([58.200029, 64.826353, 69.332991, 73.504694, 74.295033, 77.506575, 78.413291, 79.952682, 81.339708, 81.938359, 82.528196, 82.807452, 83.378999, 84.521468, 84.507216, 85.064265, 85.247146, 85.900079, 86.475709, 86.776052, 87.158810, 87.343755, 87.448085, 87.822712, 88.140434, 88.311032, 88.619312, 88.970210, 89.373613, 89.754486, 89.900430, 90.116608, 90.288358, 90.711677, 90.989783])
time = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5])

# Plot experimental data
plt.figure()
plt.plot(time, radius_data1, "yo")
plt.plot(time, radius_data2, "go")
plt.plot(time, radius_data3, "mo")
plt.xlabel("time (s)")
plt.ylabel("radius (µm)")
plt.title("Experimental Data")
plt.show()

# Compute U
def solve_for_U(R, T):
    delta_R = np.diff(R)
    delta_T = np.diff(T)
    return np.abs(delta_R / delta_T)

U1 = solve_for_U(radius_data1, time)
U2 = solve_for_U(radius_data2, time)
U3 = solve_for_U(radius_data3, time)

# Compute H for volume conservation
def volume_equation(H, R):
    return (np.pi / 6) * H * (3 * R**2 + H**2) - 7.6e3

def solve_for_H(R, H_guess=20):
    H_solution = []
    for r in R:
        H_value = fsolve(volume_equation, H_guess, args=(r,), maxfev=10000)[0]
        H_solution.append(H_value)
    return np.array(H_solution)

# Compute contact angle
def solve_for_theta(R):
    H = solve_for_H(R, H_guess=20)
    return (np.pi/2) - np.arctan((R**2 - H**2)/(2*R*H))

TH1 = solve_for_theta(radius_data1)[1:]
TH2 = solve_for_theta(radius_data2)[1:]
TH3 = solve_for_theta(radius_data3)[1:]

# Prepare data arrays
Data_1 = np.column_stack((U1, TH1))
Data_2 = np.column_stack((U2, TH2))
Data_3 = np.column_stack((U3, TH3))

# Mean and standard deviation
U_combined = np.column_stack((U1, U2, U3))
U_average = np.mean(U_combined, axis=1)
U_unc = np.std(U_combined, axis=1)

TH_combined = np.column_stack((TH1, TH2, TH3))
TH_average = np.mean(TH_combined, axis=1)
TH_unc = np.std(TH_combined, axis=1)

plt.scatter(TH_average, U_average)
plt.errorbar(TH_average, U_average, xerr=TH_unc, yerr=U_unc, fmt='x', capsize=5)
plt.xlabel('Contact angle (rads)')
plt.ylabel('Mean speed (µm/s)')
plt.title('Mean speed vs mean contact angle')
plt.grid(True)
plt.show()

# Linear model and fitting
def linear_model(x, A, B):
    return A * (x - B)

Data_THS = np.column_stack((U_average, TH_average**2, U_unc))
Data_THC = np.column_stack((U_average, TH_average**3, U_unc))

def calculate_initial_guesses(data):
    data_sorted = data[np.argsort(data[:,1])]
    y, x = data_sorted[:,0], data_sorted[:,1]
    yint = np.interp(0, x, y)
    grad = (np.max(y)-np.min(y)) / (np.max(x)-np.min(x))
    return [grad, (yint/grad)]

def fit_linear_model(data, law_name, P1, P2):
    initial_guesses = calculate_initial_guesses(data)
    y, x, unc = data[:,0], data[:,1], data[:,2]
    popt, pcov = curve_fit(linear_model, x, y, sigma=unc, p0=initial_guesses)
    unc_fit = np.sqrt(np.diag(pcov))
    print(law_name, 'Has parameters', P1, '=', popt[0], '±', unc_fit[0], 'and', P2, '=', popt[1], '±', unc_fit[1])
    return popt, pcov

popt_squared, pcov_squared = fit_linear_model(Data_THS,'De Gennes law','U₀','θ₀²')
popt_cubed, pcov_cubed = fit_linear_model(Data_THC,'Cox-Voinov law','U₀','θ₀³')

def fit_and_plot(data, popt, plot_title, x_axis_title, X_ERR):
    y, x = data[:,0], data[:,1]
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = linear_model(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r', label='Fitted Model')
    plt.scatter(x, y, color='blue', label='Data')
    plt.errorbar(x, y, xerr=X_ERR, yerr=U_unc, fmt='x', color='blue', capsize=5)
    plt.xlabel(x_axis_title)
    plt.ylabel('Mean speed (µm/s)')
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()

fit_and_plot(Data_THS, popt_squared, 'Mean speed vs Contact angle squared', 'Contact angle squared (rads)**2', 2*TH_average*TH_unc)
fit_and_plot(Data_THC, popt_cubed, 'Mean speed vs Contact angle cubed', 'Contact angle cubed (rads)**3', 3*(TH_average**2)*TH_unc)

# Reduced chi-squared
def reduced_chi_squared(y, popt, y_err, x, num_params):
    residuals = (y - linear_model(x, *popt)) / y_err
    chi_sq = np.sum(residuals**2)
    rdc = chi_sq / (len(y) - num_params)
    print("Reduced chi-squared =", rdc)
    plt.scatter(x, residuals)
    plt.xlabel('x')
    plt.ylabel('Residuals / Uncertainty')
    plt.grid()
    plt.show()
    return rdc

theta_squared_rdc = reduced_chi_squared(U_average, popt_squared, U_unc, TH_average**2, 2)
theta_cubed_rdc = reduced_chi_squared(U_average, popt_cubed, U_unc, TH_average**3, 2)

def contour_plot(x, y, parameters, uncertanities, y_label, plot_title):
    # Extract best-fit parameters for A and B
    A_best, B_best = parameters

    # Define a range for parameters to try
    A_range = np.linspace(A_best * 0.25, A_best * 1.75, 100)
    B_range = np.linspace(B_best * 0.25, B_best * 1.75, 100)
    
    # Create a grid of (A, B) values.
    A_grid, B_grid = np.meshgrid(A_range, B_range)

    # Prepare an array to store the chi-squared values at each grid point.
    chi2 = np.zeros_like(A_grid)

    # Loop over the grid and compute the sum of squared residuals
    for i, B_val in enumerate(B_range):
        for j, A_val in enumerate(A_range):
            # Evaluate the model with the current A and B.
            model_curve = linear_model(x, A_val, B_val)
            residuals = y - model_curve
            chi2[i, j] = np.sum(residuals / uncertanities)**2

    # Create the contour plot.
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(A_grid, B_grid, chi2, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Sum of Squared Residuals')
    plt.xlabel('U₀')
    plt.ylabel(y_label)
    plt.title(plot_title)
    
    # Plot the best-fit parameters on the contour.
    plt.plot(A_best, B_best, 'r*', markersize=12, label='Best Fit')
    plt.legend()
    plt.show()


contour_plot(TH_average**2, U_average, popt_squared, U_unc, y_label = 'θ₀²', plot_title = 'Contour plot for theta squared fit')
contour_plot(TH_average**3, U_average, popt_cubed, U_unc, y_label = 'θ₀³', plot_title = 'Contour plot for theta cubed fit')
