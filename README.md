# Z Boson Mass Reconstruction (Gaussian Fit)

Numerical simulation and data analysis of a physical system using Python and scientific computing tools.

## 📌 Description
This project reconstructs the **invariant mass** of dimuon events from a CSV dataset and identifies the Z boson peak.
It includes histogram exploration (linear and log scale), peak detection, and a Gaussian fit to estimate the peak position and uncertainty.

## 🛠️ Tools
- Python
- NumPy, Pandas
- SciPy (peak finding + curve fitting)
- Matplotlib

## 📄 Report
A detailed analysis and discussion of the results is available in `report/analysis_report.pdf`.


## 📊 Methodology
- Compute invariant mass:  
  \( m = \sqrt{E^2 - (p_x^2 + p_y^2 + p_z^2)} \)
- Histogram exploration of the full spectrum
- Zoom into Z region (default 60–120 GeV)
- Peak detection (SciPy `find_peaks`)
- Gaussian fit around detected peak(s)

## ▶️ How to run
1) Install dependencies:
```bash
pip install -r requirements.txt
