import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tqdm 
from concurrent.futures import ProcessPoolExecutor
import uproot
import pandas as pd

def exponential_fit(x, a, b):
    """Exponential function for fitting."""
    return a * np.exp(b * x)

def fit_histogram_exponential(counts, bin_edges, range_fit=(100, 150)):
    """
    Fit an exponential to the bin counts of a histogram in a specific range.

    Parameters:
    - data: Array-like, data to be histogrammed.
    - bins: Integer, number of bins for the histogram.
    - range_fit: Tuple, range to fit the exponential function (default is (100, 150)).

    Returns:
    - popt: Optimal values for the parameters of the fit.
    - pcov: Covariance of popt.
    """

    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Select the bins and counts within the specified range
    mask = (bin_centers >= range_fit[0]) & (bin_centers <= range_fit[1])
    fit_x = bin_centers[mask]
    fit_y = counts[mask]

    # Fit the exponential function to the selected data
    popt, pcov = curve_fit(exponential_fit, fit_x, fit_y, p0=(1, -0.01))

    # Plot the fitted curve
    x_fit = np.linspace(range_fit[0], range_fit[1], 100)
    y_fit = exponential_fit(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit: a*exp(b*x)\n a={popt[0]:.2f}, b={popt[1]:.4f}')
    plt.scatter(bin_centers, counts)
    plt.legend()

    plt.title('Histogram with Exponential Fit in Range 100-150 GeV')
    plt.xlabel('Invariant Mass (GeV)')
    plt.ylabel('Number of Events')
    plt.show()

    return popt, pcov

def exponential_poly_fit(x, a, b, c):
    """Exponential function for fitting."""
    return a * np.exp(b * x + c * x**2)

def fit_histogram_exponential_poly(counts, bin_edges, range_fit=(100, 150), ign_window=[122,128]):
    """
    Fit an exponential to the bin counts of a histogram in a specific range.

    Parameters:
    - data: Array-like, data to be histogrammed.
    - bins: Integer, number of bins for the histogram.
    - range_fit: Tuple, range to fit the exponential function (default is (100, 150)).

    Returns:
    - popt: Optimal values for the parameters of the fit.
    - pcov: Covariance of popt.
    """

    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Select the bins and counts within the specified range
    mask = (bin_centers >= range_fit[0]) & (bin_centers <= range_fit[1]) & ((bin_centers <= ign_window[0]) | (bin_centers > ign_window[1])) 
    fit_x = bin_centers[mask]
    fit_y = counts[mask]

    # Fit the exponential function to the selected data
    popt, pcov = curve_fit(exponential_poly_fit, fit_x, fit_y, p0=(500000, -0.01, 0)) 

    # Plot the fitted curve
    x_fit = np.linspace(range_fit[0], range_fit[1], 100)
    y_fit = exponential_poly_fit(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit: a*exp(b*x)\n a={popt[0]:.2f}, b={popt[1]:.4f}, c={popt[2]:.4f}')
    plt.scatter(fit_x, fit_y, c='b')
    plt.scatter(bin_centers[np.invert(mask)], counts[np.invert(mask)],c='g')
    plt.legend()

    plt.title('Histogram with Exponential Fit in Range 100-150 GeV')
    plt.xlabel('Invariant Mass (GeV)')
    plt.ylabel('Number of Events')
    plt.show()

    return popt, pcov



def compute_invariant_mass(E, pt, eta, phi):
    """
    Compute the invariant mass of two leading photons in an event.
    """
    # Ensure there are at least two photons
    if len(E) < 2:
        return None

    # Sort photons by energy
    sorted_indices = np.argsort(E)[::-1]
    E, pt, eta, phi = E[sorted_indices], pt[sorted_indices], eta[sorted_indices], phi[sorted_indices]

    # Calculate px, py, pz from pt, eta, phi
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    # Calculate the invariant mass of the two leading photons
    E1, E2 = E[0], E[1]
    px1, px2 = px[0], px[1]
    py1, py2 = py[0], py[1]
    pz1, pz2 = pz[0], pz[1]

    if E1 > 0 and E2 > 0:  # Ensure energies are positive
        invariant_mass = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
        return invariant_mass / 1000  # Convert to GeV
    return None

def process_file(file_path):
    """
    Process a single file and return a DataFrame with relevant data.
    """
    # Open the file and read the branches
    with uproot.open(file_path) as file:
        tree = file['mini']

        # Read branches for photon data and additional event-level data
        photon_E = tree["photon_E"].array(library="np")
        photon_pt = tree["photon_pt"].array(library="np")
        photon_eta = tree["photon_eta"].array(library="np")
        photon_phi = tree["photon_phi"].array(library="np")
        photon_isTightID = tree["photon_isTightID"].array(library="np")
        photon_ptcone30 = tree["photon_ptcone30"].array(library="np")
        photon_etcone20 = tree["photon_etcone20"].array(library="np")
        runNumber = tree["runNumber"].array(library="np")
        eventNumber = tree["eventNumber"].array(library="np")
        jet_n = tree["jet_n"].array(library="np")
        lep_n = tree["lep_n"].array(library="np")
        met_et = tree["met_et"].array(library="np")
        met_phi = tree["met_phi"].array(library="np")

    # List to hold event data
    data = []

    # Loop over each event and compute necessary information
    for idx in range(len(photon_E)):
        E = photon_E[idx]
        pt = photon_pt[idx]
        eta = photon_eta[idx]
        phi = photon_phi[idx]
        
        # Filter only events with exactly two photons
        if len(E) != 2:
            continue
        
        # Compute the invariant mass of the two photons
        invariant_mass = compute_invariant_mass(E, pt, eta, phi)
        
        if invariant_mass is not None:
            # Collect data for the dataframe
            data.append({
                'runNumber': runNumber[idx],
                'eventNumber': eventNumber[idx],
                'invariant_mass': invariant_mass,
                'photon_isTightID': photon_isTightID[idx].tolist(),
                'photon_ptcone30': photon_ptcone30[idx].tolist(),
                'photon_etcon20': photon_etcone20[idx].tolist(),
                'jet_n': jet_n[idx],
                'lep_n': lep_n[idx],
                'met_et': met_et[idx],
                'met_phi': met_phi[idx]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def process_file(file_path):
    """
    Process a single file and return a DataFrame with relevant data.
    """
    # Open the file and read the branches
    with uproot.open(file_path) as file:
        tree = file['mini']

        # Read branches for photon data and additional event-level data
        photon_E = tree["photon_E"].array(library="np")
        photon_pt = tree["photon_pt"].array(library="np")
        photon_eta = tree["photon_eta"].array(library="np")
        photon_phi = tree["photon_phi"].array(library="np")
        photon_isTightID = tree["photon_isTightID"].array(library="np")
        photon_ptcone30 = tree["photon_ptcone30"].array(library="np")
        photon_etcone20 = tree["photon_etcone20"].array(library="np")
        photon_trigMatched = tree["photon_trigMatched"].array(library="np")  # Read 'photon_trigMatched'
        runNumber = tree["runNumber"].array(library="np")
        eventNumber = tree["eventNumber"].array(library="np")
        jet_n = tree["jet_n"].array(library="np")
        lep_n = tree["lep_n"].array(library="np")
        met_et = tree["met_et"].array(library="np")
        met_phi = tree["met_phi"].array(library="np")

    # List to hold event data
    data = []

    # Loop over each event and compute necessary information
    for idx in range(len(photon_E)):
        E = photon_E[idx]
        pt = photon_pt[idx]
        eta = photon_eta[idx]
        phi = photon_phi[idx]
        
        # Filter only events with exactly two photons
        if len(E) != 2:
            continue
        
        # Compute the invariant mass of the two photons
        invariant_mass = compute_invariant_mass(E, pt, eta, phi)
        
        if invariant_mass is not None:
            # Collect data for the dataframe
            data.append({
                'runNumber': runNumber[idx],
                'eventNumber': eventNumber[idx],
                'invariant_mass': invariant_mass,
                'photon_isTightID': photon_isTightID[idx].tolist(),
                'photon_ptcone30': photon_ptcone30[idx].tolist(),
                'photon_etcon20': photon_etcone20[idx].tolist(),
                'photon_trigMatched': photon_trigMatched[idx].tolist(),  # Add 'photon_trigMatched'
                'jet_n': jet_n[idx],
                'lep_n': lep_n[idx],
                'met_et': met_et[idx],
                'met_phi': met_phi[idx]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def process_files_parallel(file_paths):
    """
    Process multiple files in parallel and return a combined DataFrame.
    """
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Use tqdm to display progress bar
        results = list(tqdm.tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    # Concatenate all DataFrames
    combined_df = pd.concat(results, ignore_index=True)
    return combined_df