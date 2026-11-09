# 📈 Task 2: Unemployment Analysis with Python (COVID-19 Economic Study)

## 📋 Project Overview
This project presents an empirical investigation into unemployment dynamics in India across 2019 and 2020. It evaluates the impact of the **COVID-19 nationwide lockdown**, examines regional and demographic variations (Rural vs. Urban), and provides data-driven policy recommendations.

## 📁 Folder Structure
```
Task_2_Unemployment_Analysis/
├── dataset/
│   ├── Unemployment in India.csv                # High-frequency Rural/Urban dataset (740 records)
│   └── Unemployment_Rate_upto_11_2020.csv       # State-wise macro geographic dataset (267 records)
├── outputs/
│   ├── figures/                                 # Exported analytical visualizations
│   │   ├── 01_national_unemployment_timeline.png
│   │   ├── 02_covid_lockdown_impact.png
│   │   ├── 03_rural_vs_urban_unemployment.png
│   │   ├── 04_top_bottom_states_unemployment.png
│   │   ├── 05_macro_region_breakdown.png
│   │   ├── 06_correlation_matrix.png
│   │   └── 07_state_month_heatmap.png
│   └── executive_unemployment_report.txt        # Automated executive economic policy report
├── unemployment_analysis.py                     # Full end-to-end data pipeline script
├── unemployment_analysis.ipynb                  # Interactive Jupyter Notebook
└── README.md                                    # Project documentation
```

## 🔍 Key Findings
1. **COVID-19 Lockdown Shock:**
   - Pre-Lockdown Baseline (May 2019 – Feb 2020): **9.51%**
   - Lockdown Peak Period (Mar 2020 – Jun 2020): **17.77%** (+8.26 percentage point jump)
   - Highest single regional spike: **75.85%** recorded in Puducherry in April 2020.
2. **Rural vs. Urban Divergence:**
   - **Urban Average:** **13.17%** (experienced higher volatility and deeper drops due to service and industrial freezes).
   - **Rural Average:** **10.32%** (cushioned by agricultural continuity and the MGNREGA scheme).
3. **State Variations:**
   - Highest average unemployment: Haryana, Tripura, Jharkhand, and Bihar.
   - Most resilient states: Meghalaya, Odisha, and Assam.

## 💡 Policy Recommendations
- **Urban Employment Guarantee:** Launch an urban counterpart to MGNREGA to protect gig workers and informal laborers.
- **MSME Working Capital:** Establish micro-credit support lines for small enterprises to prevent involuntary layoffs.
- **Skill Reshoring:** Expand digital and technical vocational education programs in high-vulnerability regions.

## 🚀 How to Run
```bash
# Execute standalone analysis
python unemployment_analysis.py

# Or inspect Jupyter Notebook
jupyter notebook unemployment_analysis.ipynb
```
