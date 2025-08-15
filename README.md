# Droplet Spreading Analysis

This repository contains a Python script for analyzing droplet spreading experiments.  
The script processes experimental radius vs. time data for multiple droplets, calculates spreading speeds and contact angles, fits physical models (De Gennes and Cox–Voinov laws), and compares model performance using statistical metrics.

---

## Features

- **Load Experimental Data**  
  Reads measured droplet radii over time from three experimental runs.

- **Speed Calculation**  
  Computes instantaneous spreading speed `U` from radius vs. time data.

- **Geometry Calculation**  
  Solves for droplet height `H` from the volume conservation equation and calculates contact angle `θ`.

- **Data Averaging**  
  Combines multiple runs to obtain mean values and uncertainties for both speed and contact angle.

- **Model Fitting**  
  Fits two models:  
  - **De Gennes Law:** \( U = U_0 (\theta^2 - \theta_0^2) \)  
  - **Cox–Voinov Law:** \( U = U_0 (\theta^3 - \theta_0^3) \)

- **Error Analysis**  
  - Plots data with error bars.
  - Computes reduced chi-squared for both models.
  - Generates residual plots to visually assess fit quality.

---

## Requirements

Install dependencies with:

```bash
pip install numpy matplotlib scipy
