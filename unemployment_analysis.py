"""
Task 2: Unemployment Analysis with Python
CodeAlpha Data Science Internship
Author: CodeAlpha Intern
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_1 = os.path.join(DATASET_DIR, "Unemployment in India.csv")
CSV_2 = os.path.join(DATASET_DIR, "Unemployment_Rate_upto_11_2020.csv")


def clean_dataframe(df):
    """Clean column names, strip whitespace, handle missing values and parse dates."""
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    
    # Drop rows where Region or Date is null
    df = df.dropna(subset=['Region', 'Date']).copy()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Standardize Date format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Feature extraction
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b %Y')
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    return df


def load_datasets():
    print("=" * 65)
    print("STEP 1: Loading & Cleaning Unemployment Datasets")
    print("=" * 65)
    
    df1 = pd.read_csv(CSV_1)
    df1_clean = clean_dataframe(df1)
    print(f"Dataset 1 (Rural/Urban detailed): {df1_clean.shape[0]} records, {df1_clean.shape[1]} columns")
    
    df2 = pd.read_csv(CSV_2)
    # Rename Region.1 to Macro_Region
    if 'Region.1' in [c.strip() for c in df2.columns]:
        df2 = df2.rename(columns={c: 'Macro_Region' for c in df2.columns if c.strip() == 'Region.1'})
    df2_clean = clean_dataframe(df2)
    print(f"Dataset 2 (State & Macro-region with Lat/Long): {df2_clean.shape[0]} records, {df2_clean.shape[1]} columns")
    
    # Categorize Covid-19 Phases
    # Pre-Lockdown: before March 2020
    # Lockdown Peak: March 2020 to June 2020
    # Post-Lockdown / Recovery: July 2020 onwards
    def get_phase(date):
        if date < pd.Timestamp('2020-03-01'):
            return 'Pre-Lockdown (May 2019 - Feb 2020)'
        elif date <= pd.Timestamp('2020-06-30'):
            return 'Lockdown Peak (Mar 2020 - Jun 2020)'
        else:
            return 'Unlock / Recovery (Jul 2020 - Nov 2020)'
            
    df1_clean['Covid_Phase'] = df1_clean['Date'].apply(get_phase)
    df2_clean['Covid_Phase'] = df2_clean['Date'].apply(get_phase)
    
    return df1_clean, df2_clean


def analyze_and_plot(df1, df2, output_dir=FIGURES_DIR):
    print("\n" + "=" * 65)
    print("STEP 2: Generating In-Depth Visualizations & Trends")
    print("=" * 65)
    
    # -------------------------------------------------------------
    # 1. Timeline of Unemployment Rate in India
    # -------------------------------------------------------------
    monthly_trend = df1.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_trend['Date'], monthly_trend['Estimated Unemployment Rate (%)'],
             marker='o', color='#d62728', linewidth=2.5, markersize=7)
    
    # Annotate Lockdown peak
    peak_row = monthly_trend.loc[monthly_trend['Estimated Unemployment Rate (%)'].idxmax()]
    plt.annotate(
        f"Covid Lockdown Shock Peak\n{peak_row['Estimated Unemployment Rate (%)']:.1f}% ({peak_row['Date'].strftime('%b %Y')})",
        xy=(peak_row['Date'], peak_row['Estimated Unemployment Rate (%)']),
        xytext=(peak_row['Date'] - pd.Timedelta(days=70), peak_row['Estimated Unemployment Rate (%)'] + 4),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
        fontweight='bold', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.6)
    )
    plt.title("National Average Monthly Unemployment Rate Trend (2019 - 2020)", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Date", fontweight='bold', fontsize=12)
    plt.ylabel("Unemployment Rate (%)", fontweight='bold', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    p1 = os.path.join(output_dir, "01_national_unemployment_timeline.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")
    
    # -------------------------------------------------------------
    # 2. Covid-19 Phase Comparison (Pre-Lockdown vs Peak vs Recovery)
    # -------------------------------------------------------------
    phase_order = [
        'Pre-Lockdown (May 2019 - Feb 2020)',
        'Lockdown Peak (Mar 2020 - Jun 2020)',
        'Unlock / Recovery (Jul 2020 - Nov 2020)'
    ]
    # Filter only available phases in df1
    available_phases = [p for p in phase_order if p in df1['Covid_Phase'].values]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Covid_Phase', y='Estimated Unemployment Rate (%)', data=df1,
                order=available_phases, palette=['#2ca02c', '#d62728', '#1f77b4'])
    plt.title("Impact of Covid-19 Lockdown Phases on Unemployment Rate", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Pandemic Period", fontweight='bold', fontsize=12)
    plt.ylabel("Unemployment Rate (%)", fontweight='bold', fontsize=12)
    p2 = os.path.join(output_dir, "02_covid_lockdown_impact.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")
    
    # -------------------------------------------------------------
    # 3. Rural vs. Urban Unemployment Dynamics
    # -------------------------------------------------------------
    if 'Area' in df1.columns:
        area_trend = df1.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=area_trend, x='Date', y='Estimated Unemployment Rate (%)',
                     hue='Area', style='Area', markers=True, dashes=False, palette={'Rural': '#2ca02c', 'Urban': '#ff7f0e'}, linewidth=2.5)
        plt.title("Rural vs. Urban Unemployment Dynamics (Pre & Post Covid)", fontsize=15, fontweight='bold', pad=15)
        plt.xlabel("Date", fontweight='bold', fontsize=12)
        plt.ylabel("Unemployment Rate (%)", fontweight='bold', fontsize=12)
        plt.legend(title="Area Type", title_fontsize='11', fontsize='10')
        p3 = os.path.join(output_dir, "03_rural_vs_urban_unemployment.png")
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f"Saved: {p3}")
    
    # -------------------------------------------------------------
    # 4. State-Wise Analysis (Top 10 vs Bottom 10 States)
    # -------------------------------------------------------------
    state_avg = df2.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Top 10 Highest
    top10 = state_avg.head(10)
    sns.barplot(x=top10.values, y=top10.index, ax=axes[0], palette='Reds_r')
    axes[0].set_title("Top 10 States with Highest Avg Unemployment", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Average Unemployment Rate (%)", fontweight='bold')
    for i, v in enumerate(top10.values):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
        
    # Top 10 Lowest
    bottom10 = state_avg.tail(10).sort_values(ascending=True)
    sns.barplot(x=bottom10.values, y=bottom10.index, ax=axes[1], palette='Greens_r')
    axes[1].set_title("Top 10 States with Lowest Avg Unemployment", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Average Unemployment Rate (%)", fontweight='bold')
    for i, v in enumerate(bottom10.values):
        axes[1].text(v + 0.2, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
        
    plt.suptitle("State-by-State Unemployment Comparison in India (2020)", fontsize=16, fontweight='bold')
    p4 = os.path.join(output_dir, "04_top_bottom_states_unemployment.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")
    
    # -------------------------------------------------------------
    # 5. Macro-Regional Breakdown (North, South, East, West, North-East)
    # -------------------------------------------------------------
    if 'Macro_Region' in df2.columns:
        plt.figure(figsize=(10, 6))
        macro_avg = df2.groupby('Macro_Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
        sns.barplot(x=macro_avg.index, y=macro_avg.values, palette='Blues_r')
        plt.title("Unemployment Rate Across Macro Geographic Zones", fontsize=15, fontweight='bold', pad=15)
        plt.xlabel("Macro Region", fontweight='bold', fontsize=12)
        plt.ylabel("Mean Unemployment Rate (%)", fontweight='bold', fontsize=12)
        for idx, val in enumerate(macro_avg.values):
            plt.text(idx, val + 0.4, f"{val:.2f}%", ha='center', fontweight='bold')
        p5 = os.path.join(output_dir, "05_macro_region_breakdown.png")
        plt.savefig(p5, dpi=300)
        plt.close()
        print(f"Saved: {p5}")
        
    # -------------------------------------------------------------
    # 6. Correlation Matrix
    # -------------------------------------------------------------
    numeric_cols = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)'
    ]
    plt.figure(figsize=(8, 6))
    corr = df2[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", linewidths=1.0)
    plt.title("Correlation: Unemployment, Employment & Labour Participation", fontsize=13, fontweight='bold', pad=15)
    p6 = os.path.join(output_dir, "06_correlation_matrix.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")
    
    # -------------------------------------------------------------
    # 7. State vs Month Unemployment Heatmap (Covid Peak Visualizer)
    # -------------------------------------------------------------
    pivot_table = df2.pivot_table(
        index='Region',
        columns=df2['Date'].dt.strftime('%b %Y'),
        values='Estimated Unemployment Rate (%)',
        aggfunc='mean'
    )
    # Sort months chronologically
    sorted_cols = sorted(pivot_table.columns, key=lambda x: pd.to_datetime(x, format='%b %Y'))
    pivot_table = pivot_table[sorted_cols]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".1f", linewidths=0.5, cbar_kws={'label': 'Unemployment Rate (%)'})
    plt.title("State-wise Unemployment Heatmap Month-by-Month (2020)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Month", fontweight='bold', fontsize=12)
    plt.ylabel("State / Region", fontweight='bold', fontsize=12)
    p7 = os.path.join(output_dir, "07_state_month_heatmap.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"Saved: {p7}")


def generate_executive_insights(df1, df2):
    print("\n" + "=" * 65)
    print("STEP 3: Statistical Findings & Policy Insights")
    print("=" * 65)
    
    # National Metrics
    pre_covid_unemp = df1[df1['Covid_Phase'].str.contains('Pre-Lockdown')]['Estimated Unemployment Rate (%)'].mean()
    lockdown_unemp = df1[df1['Covid_Phase'].str.contains('Lockdown Peak')]['Estimated Unemployment Rate (%)'].mean()
    
    # Area Metrics
    rural_mean = df1[df1['Area'] == 'Rural']['Estimated Unemployment Rate (%)'].mean()
    urban_mean = df1[df1['Area'] == 'Urban']['Estimated Unemployment Rate (%)'].mean()
    
    # Maximum recorded shock
    max_rate = df2['Estimated Unemployment Rate (%)'].max()
    max_row = df2.loc[df2['Estimated Unemployment Rate (%)'].idxmax()]
    
    report = f"""
======================================================================
          UNEMPLOYMENT ANALYSIS & ECONOMIC POLICY REPORT
======================================================================

1. PANDEMIC SHOCK & RECOVERY DYNAMICS:
   - Baseline Pre-Lockdown Unemployment Rate (May 2019 - Feb 2020): {pre_covid_unemp:.2f}%
   - Lockdown Peak Unemployment Rate (Mar 2020 - Jun 2020): {lockdown_unemp:.2f}%
   - Absolute Shock Increase: +{lockdown_unemp - pre_covid_unemp:.2f}% percentage points.
   - Highest single recorded unemployment surge: {max_rate:.2f}% in {max_row['Region']} ({max_row['Date'].strftime('%b %Y')}).

2. RURAL VS. URBAN SECTOR DISPARITY:
   - Average Rural Unemployment: {rural_mean:.2f}%
   - Average Urban Unemployment: {urban_mean:.2f}%
   - Observation: Urban centers suffered higher unemployment and sharper volatility due to service and manufacturing lockdowns, whereas rural regions experienced resilience buffered by agriculture and the MGNREGA social safety net.

3. SOCIO-ECONOMIC & POLICY RECOMMENDATIONS:
   - Urban Employment Guarantee: Implement an urban safety-net scheme similar to MGNREGA to protect informal urban laborers during economic disruptions.
   - Small Business Liquidity: Provide collateral-free working capital micro-loans to MSMEs to prevent permanent layoffs during demand contractions.
   - Labour Force Re-skilling: Subsidize digital and technical skill development programs in high-unemployment northern and eastern states.
   - High-Frequency Data Monitoring: Deploy real-time regional dashboard systems for rapid targeted social assistance.
======================================================================
"""
    print(report)
    
    # Save text report to outputs
    report_path = os.path.join(OUTPUT_DIR, "executive_unemployment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Executive report saved to: {report_path}")


if __name__ == "__main__":
    df1, df2 = load_datasets()
    analyze_and_plot(df1, df2)
    generate_executive_insights(df1, df2)
    print("\nTask 2 execution completed successfully!")
