import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def build_notebook():
    print("Building notebook structure...")
    nb = nbf.v4.new_notebook()
    cells = []

    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Traffic Demand Prediction - Exploratory Data Analysis (EDA)\n\n"
        "This notebook provides a thorough Exploratory Data Analysis (EDA) for the Traffic Demand Prediction task.\n"
        "It loads the dataset, inspects missing values, analyzes the target variable (`demand`), visualizes individual features, "
        "runs correlation analyses, explores time patterns, and summarizes key insights."
    ))

    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Set plotting aesthetics\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['font.size'] = 11\n"
        "print('Libraries imported successfully.')"
    ))

    # Load Data
    cells.append(nbf.v4.new_code_cell(
        "# Load datasets\n"
        "train_df = pd.read_csv('dataset/train.csv')\n"
        "test_df = pd.read_csv('dataset/test.csv')\n\n"
        "print(f'Train dataset shape: {train_df.shape}')\n"
        "print(f'Test dataset shape: {test_df.shape}')\n"
        "print('\\nTrain columns:', list(train_df.columns))\n"
        "print('Test columns:', list(test_df.columns))"
    ))

    # Data Head and Types
    cells.append(nbf.v4.new_code_cell(
        "print('--- Train Data Sample ---')\n"
        "display(train_df.head())\n\n"
        "print('--- Data Types & Non-Null Counts ---')\n"
        "train_df.info()"
    ))

    # Section 2: Missing Values
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Check for Missing Values\n\n"
        "We identify missing values in both the training and test datasets."
    ))

    cells.append(nbf.v4.new_code_cell(
        "train_missing = train_df.isnull().sum()\n"
        "train_missing_pct = 100 * train_df.isnull().mean()\n"
        "missing_train_df = pd.DataFrame({'Missing Count': train_missing, 'Percentage (%)': train_missing_pct})\n"
        "print('--- Missing values in Train Data ---')\n"
        "display(missing_train_df[missing_train_df['Missing Count'] > 0])\n\n"
        "test_missing = test_df.isnull().sum()\n"
        "test_missing_pct = 100 * test_df.isnull().mean()\n"
        "missing_test_df = pd.DataFrame({'Missing Count': test_missing, 'Percentage (%)': test_missing_pct})\n"
        "print('--- Missing values in Test Data ---')\n"
        "display(missing_test_df[missing_test_df['Missing Count'] > 0])"
    ))

    # Section 3: Target Variable Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Understand the Target Variable (`demand`)\n\n"
        "We compute key statistics of the `demand` column and evaluate its skewness to decide if a log transformation is appropriate."
    ))

    cells.append(nbf.v4.new_code_cell(
        "print('Summary Statistics of demand:')\n"
        "display(train_df['demand'].describe())\n"
        "print(f'Median: {train_df[\"demand\"].median()}')\n"
        "print(f'Skewness: {train_df[\"demand\"].skew()}')\n"
        "print(f'Minimum value (non-zero check): {train_df[\"demand\"].min()}')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
        "# Original distribution\n"
        "sns.histplot(train_df['demand'], kde=True, bins=50, ax=axes[0], color='royalblue')\n"
        "axes[0].set_title('Distribution of Original Demand')\n"
        "axes[0].set_xlabel('Demand')\n"
        "axes[0].set_ylabel('Count')\n\n"
        "# Log distribution\n"
        "log_demand = np.log(train_df['demand'])\n"
        "sns.histplot(log_demand, kde=True, bins=50, ax=axes[1], color='teal')\n"
        "axes[1].set_title('Distribution of Log-Transformed Demand (np.log)')\n"
        "axes[1].set_xlabel('Log(Demand)')\n"
        "axes[1].set_ylabel('Count')\n\n"
        "plt.suptitle('Target Variable Distribution & Skewness Correction', fontsize=14)\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(f'Original skewness: {train_df[\"demand\"].skew():.4f}')\n"
        "print(f'Log-transformed skewness: {log_demand.skew():.4f}')"
    ))

    # Section 4: Feature Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Analyze Each Feature\n\n"
        "We break down the categorical features (`RoadType`, `Weather`, `LargeVehicles`, `Landmarks`), "
        "numerical features (`Temperature`, `NumberofLanes`), and geographical identifiers (`geohash`)."
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Categorical Columns: Value Counts & Bar Charts"
    ))

    cells.append(nbf.v4.new_code_cell(
        "cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']\n\n"
        "for col in cat_cols:\n"
        "    print(f'--- Column: {col} ---')\n"
        "    print('Unique counts (including NaNs):')\n"
        "    print(train_df[col].value_counts(dropna=False))\n"
        "    print() \n\n"
        "# Fill NaNs with 'Missing' and convert to string for plotting\n"
        "plot_df = train_df.copy()\n"
        "for col in cat_cols:\n"
        "    plot_df[col] = plot_df[col].fillna('Missing').astype(str)\n\n"
        "# Plot value counts for categorical variables\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
        "axes = axes.flatten()\n\n"
        "for i, col in enumerate(cat_cols):\n"
        "    sns.countplot(data=plot_df, x=col, ax=axes[i], order=plot_df[col].value_counts().index, palette='viridis')\n"
        "    axes[i].set_title(f'Count of {col}')\n"
        "    axes[i].set_xlabel(col)\n"
        "    axes[i].set_ylabel('Count')\n"
        "    axes[i].tick_params(axis='x', rotation=30)\n\n"
        "plt.suptitle('Distribution of Categorical Features', fontsize=14)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Plot average demand per category to understand relationships\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
        "axes = axes.flatten()\n\n"
        "for i, col in enumerate(cat_cols):\n"
        "    sns.barplot(data=plot_df, x=col, y='demand', ax=axes[i], order=plot_df[col].value_counts().index, palette='magma', errorbar=None)\n"
        "    axes[i].set_title(f'Average Demand by {col}')\n"
        "    axes[i].set_xlabel(col)\n"
        "    axes[i].set_ylabel('Average Demand')\n"
        "    axes[i].tick_params(axis='x', rotation=30)\n\n"
        "plt.suptitle('Average Demand by Categorical Features', fontsize=14)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Numerical Columns: Distributions & Box Plots"
    ))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n\n"
        "# Temperature Histogram\n"
        "sns.histplot(train_df['Temperature'].dropna(), kde=True, ax=axes[0, 0], color='coral', bins=30)\n"
        "axes[0, 0].set_title('Temperature Distribution (Histogram)')\n"
        "axes[0, 0].set_xlabel('Temperature')\n\n"
        "# NumberofLanes Countplot\n"
        "sns.countplot(data=train_df, x='NumberofLanes', ax=axes[0, 1], palette='Blues')\n"
        "axes[0, 1].set_title('NumberofLanes Distribution (Countplot)')\n"
        "axes[0, 1].set_xlabel('NumberofLanes')\n\n"
        "# Temperature vs Demand\n"
        "sns.scatterplot(data=train_df.sample(2000, random_state=42), x='Temperature', y='demand', alpha=0.3, ax=axes[1, 0], color='crimson')\n"
        "axes[1, 0].set_title('Demand vs Temperature (Sample of 2000)')\n"
        "axes[1, 0].set_xlabel('Temperature')\n"
        "axes[1, 0].set_ylabel('Demand')\n\n"
        "# NumberofLanes vs Demand Boxplot\n"
        "sns.boxplot(data=train_df, x='NumberofLanes', y='demand', ax=axes[1, 1], palette='Blues')\n"
        "axes[1, 1].set_title('Demand Distribution by NumberofLanes')\n"
        "axes[1, 1].set_xlabel('NumberofLanes')\n"
        "axes[1, 1].set_ylabel('Demand')\n\n"
        "plt.suptitle('Analysis of Numerical Features', fontsize=14)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Geohash Locations Analysis"
    ))

    cells.append(nbf.v4.new_code_cell(
        "print(f'Number of unique geohashes in Train: {train_df[\"geohash\"].nunique()}')\n"
        "print(f'Number of unique geohashes in Test: {test_df[\"geohash\"].nunique()}')\n\n"
        "train_geos = set(train_df['geohash'])\n"
        "test_geos = set(test_df['geohash'])\n"
        "overlap = train_geos.intersection(test_geos)\n"
        "print(f'Geohash overlap between Train & Test: {len(overlap)}')\n"
        "print(f'Geohashes in Test but not in Train: {len(test_geos - train_geos)}')\n\n"
        "# Top 10 busiest locations (highest total demand)\n"
        "top_10_demand = train_df.groupby('geohash')['demand'].sum().sort_values(ascending=False).head(10)\n"
        "print(\"\\n--- Top 10 Busiest Locations (by total demand sum) ---\")\n"
        "print(top_10_demand)\n\n"
        "plt.figure(figsize=(12, 5))\n"
        "sns.barplot(x=top_10_demand.index, y=top_10_demand.values, palette='copper')\n"
        "plt.title('Top 10 Busiest Geohash Locations (Total Demand Sum)')\n"
        "plt.xlabel('Geohash Location')\n"
        "plt.ylabel('Total Demand')\n"
        "plt.xticks(rotation=45)\n"
        "plt.show()"
    ))

    # Section 5: Correlation
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Correlation Analysis\n\n"
        "We map the categorical variables to numeric codes to investigate linear correlations with `demand`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "corr_df = train_df.copy()\n"
        "corr_df['RoadType_code'] = corr_df['RoadType'].astype('category').cat.codes\n"
        "corr_df['Weather_code'] = corr_df['Weather'].astype('category').cat.codes\n"
        "corr_df['LargeVehicles_code'] = corr_df['LargeVehicles'].astype('category').cat.codes\n"
        "corr_df['Landmarks_code'] = corr_df['Landmarks'].astype('category').cat.codes\n\n"
        "features_for_corr = ['demand', 'Temperature', 'NumberofLanes', 'day', \n"
        "                     'RoadType_code', 'Weather_code', 'LargeVehicles_code', 'Landmarks_code']\n\n"
        "corr_matrix = corr_df[features_for_corr].corr()\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, vmin=-1, vmax=1)\n"
        "plt.title('Correlation Matrix Heatmap')\n"
        "plt.show()\n\n"
        "print('Correlation of features with demand (sorted):')\n"
        "print(corr_matrix['demand'].sort_values(ascending=False))"
    ))

    # Section 6: Time Patterns
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Time Patterns\n\n"
        "We parse `timestamp` to study how traffic demand fluctuates by day and hour."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Parse timestamp\n"
        "def parse_time(df):\n"
        "    df_copy = df.copy()\n"
        "    time_split = df_copy['timestamp'].str.split(':', expand=True)\n"
        "    df_copy['Hour'] = time_split[0].astype(int)\n"
        "    df_copy['Minute'] = time_split[1].astype(int)\n"
        "    df_copy['TimeInMinutes'] = df_copy['Hour'] * 60 + df_copy['Minute']\n"
        "    return df_copy\n\n"
        "train_time = parse_time(train_df)\n\n"
        "print('Average demand by Day:')\n"
        "print(train_time.groupby('day')['demand'].mean())\n\n"
        "plt.figure(figsize=(8, 5))\n"
        "sns.boxplot(data=train_time, x='day', y='demand', palette='Set2')\n"
        "plt.title('Demand Distribution by Day (48 vs 49)')\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Hourly demand trends per day\n"
        "hourly_demand = train_time.groupby(['day', 'Hour'])['demand'].mean().reset_index()\n\n"
        "plt.figure(figsize=(12, 6))\n"
        "sns.lineplot(data=hourly_demand, x='Hour', y='demand', hue='day', marker='o', palette='Set1', linewidth=2.5)\n"
        "plt.title('Average Hourly Demand Pattern (Day 48 vs Day 49)')\n"
        "plt.xlabel('Hour of the Day')\n"
        "plt.ylabel('Mean Demand')\n"
        "plt.xticks(range(24))\n"
        "plt.grid(True, linestyle='--', alpha=0.5)\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Continuous demand timeline\n"
        "train_time['TimelineMinutes'] = (train_time['day'] - 48) * 1440 + train_time['TimeInMinutes']\n"
        "timeline_demand = train_time.groupby('TimelineMinutes')['demand'].mean().reset_index()\n\n"
        "def get_label(min_val):\n"
        "    d = 48 + min_val // 1440\n"
        "    m = min_val % 1440\n"
        "    h = m // 60\n"
        "    mn = m % 60\n"
        "    return f'D{d} {h:02d}:{mn:02d}'\n\n"
        "plt.figure(figsize=(15, 6))\n"
        "plt.plot(timeline_demand['TimelineMinutes'], timeline_demand['demand'], color='purple', alpha=0.8, linewidth=2)\n"
        "plt.axvline(x=1440, color='red', linestyle='--', label='Day 48/49 Transition')\n"
        "plt.title('Continuous Traffic Demand Timeline (Mean across all locations)')\n"
        "plt.xlabel('Timeline')\n"
        "plt.ylabel('Mean Demand')\n\n"
        "ticks = np.arange(0, train_time['TimelineMinutes'].max() + 1, 180)\n"
        "tick_labels = [get_label(t) for t in ticks]\n"
        "plt.xticks(ticks, tick_labels, rotation=45)\n"
        "plt.legend()\n"
        "plt.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Section 7: Key Findings Summary
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Key Findings Summary\n\n"
        "### Target Variable (`demand`)\n"
        "* **Right-Skewed Distribution**: The demand variable is heavily right-skewed with a skewness of ~3.73. Most traffic observations represent low demand values (median of ~0.048 vs mean of ~0.094).\n"
        "* **Log Transformation Benefit**: Applying a natural log transformation `np.log(demand)` results in a skewness of `-0.728`, which is much closer to normal, indicating log-transforming target might improve model accuracy.\n\n"
        "### Missing Values\n"
        "* Missing values are present in both the training and test sets in `RoadType` (~0.8% missing), `Temperature` (~3.2% missing), and `Weather` (~1.0% missing).\n\n"
        "### Feature Analysis\n"
        "* **RoadType**: `Residential` is the most common road type, but `Highway` road types exhibit higher average demand levels.\n"
        "* **Weather**: `Sunny` weather has the highest frequency and average demand, while `Snowy` weather is associated with lower demand levels.\n"
        "* **Large Vehicles & Landmarks**: Roads that allow large vehicles or have landmarks nearby see significantly higher average demand.\n"
        "* **Number of Lanes**: High positive correlation with demand—more lanes correlate with higher traffic volume.\n\n"
        "### Geohash Locations\n"
        "* Out of 1,249 training geohashes and 1,190 test geohashes, 1,180 overlap. **10 geohashes are unique to the test set**, which will require robust location-level generalization.\n\n"
        "### Time Patterns\n"
        "* **Hourly Peaks**: Strong daily rhythm with two distinct peak periods: **08:00 - 10:00** (morning rush) and **16:00 - 19:00** (evening rush).\n"
        "* **Low Demand Period**: Night hours from **00:00 to 05:00** see minimum traffic demand."
    ))

    nb['cells'] = cells
    nbf.write(nb, 'eda.ipynb')
    print("Notebook written to eda.ipynb")

    # Pre-execute notebook
    print("Running notebook execution via ExecutePreprocessor...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        nbf.write(nb, 'eda.ipynb')
        print("Notebook successfully executed and saved with all plots inline.")
    except Exception as e:
        print("Error executing notebook:", e)
        sys.exit(1)

if __name__ == '__main__':
    build_notebook()
