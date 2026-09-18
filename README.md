# ✈️ Flight Delay Analysis

## 📌 Project Overview

Flight delays are a common issue in the aviation industry and can be caused by several operational and external factors.

This project performs a **descriptive analysis of flight delay data** to identify patterns in departure and arrival delays, compare airline-level delay patterns, examine delay causes, and analyze cancellations and diversions.

The project uses **Python** for data cleaning, preprocessing, exploratory data analysis, and visualization, along with **Tableau Public** for interactive dashboard visualization.

---

## 🎯 Objectives

* Analyze flight departure and arrival delays.
* Identify major recorded causes of flight delays.
* Compare delay patterns across airlines.
* Examine flight cancellations and diversions.
* Analyze trends in the available flight data.
* Perform data cleaning and preprocessing.
* Create meaningful data visualizations.
* Build an interactive Tableau dashboard.
* Generate useful descriptive insights from the dataset.

---

## 📊 Dataset

**Dataset:** `Flight_delay.csv`

The dataset contains:

* **484,551 flight records**
* **29 columns**

### Important columns

| Column              | Description                    |
| ------------------- | ------------------------------ |
| `Date`              | Date of the flight             |
| `DayOfWeek`         | Day of the week                |
| `Airline`           | Airline operating the flight   |
| `FlightNum`         | Flight number                  |
| `Origin`            | Origin airport                 |
| `Dest`              | Destination airport            |
| `DepTime`           | Actual departure time          |
| `ArrTime`           | Actual arrival time            |
| `DepDelay`          | Departure delay in minutes     |
| `ArrDelay`          | Arrival delay in minutes       |
| `Distance`          | Flight distance                |
| `Cancelled`         | Cancellation status            |
| `Diverted`          | Diversion status               |
| `CarrierDelay`      | Carrier-related delay          |
| `WeatherDelay`      | Weather-related delay          |
| `NASDelay`          | National Airspace System delay |
| `SecurityDelay`     | Security-related delay         |
| `LateAircraftDelay` | Late aircraft-related delay    |

### Data Quality

* Missing values were identified in `Org_Airport` and `Dest_Airport`.
* `Org_Airport` had **1,177** missing values.
* `Dest_Airport` had **1,479** missing values.
* **2 duplicate records** were identified and removed during preprocessing.

---

## 🛠️ Tools & Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Visual Studio Code**
* **Tableau Public**
* **Git & GitHub**

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Duplicate Removal
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Python Visualizations
     ↓
Tableau Visualizations
     ↓
Interactive Dashboard
     ↓
Insights & Conclusion
```

---

## 🧹 Data Cleaning & Preprocessing

The dataset was loaded and inspected using Pandas.

The following steps were performed:

1. Loaded the CSV dataset.
2. Checked the number of rows and columns.
3. Examined column names and data types.
4. Checked for missing values.
5. Identified duplicate records.
6. Removed duplicate records.
7. Handled missing airport information.
8. Prepared relevant numerical and categorical fields.
9. Prepared the date field for time-based analysis.
10. Used the cleaned dataset for analysis and visualization.

---

## 📈 Exploratory Data Analysis

The analysis focuses on several important aspects of flight operations:

* Departure delays
* Arrival delays
* Airline-level delay patterns
* Delay causes
* Flight cancellations
* Flight diversions
* Airport-related patterns
* Time-based delay patterns

Python was used to explore the dataset and create analytical visualizations.

---

## 📊 Tableau Dashboard

An interactive dashboard was created using **Tableau Public**.

The dashboard contains a maximum of **six key visualizations** to present the most important findings without overcrowding the dashboard.

The dashboard focuses on:

* Flight delay patterns
* Airline-level analysis
* Arrival delay analysis
* Delay-related measures
* Operational flight information
* Time-based patterns

The dashboard provides an easy-to-understand visual representation of the analysis.

---

## 🔍 Key Insights

The project provides a descriptive understanding of flight-delay behavior across the available records.

The analysis considers multiple dimensions, including:

* Airlines
* Flight dates
* Departure delays
* Arrival delays
* Airports
* Cancellations
* Diversions
* Recorded causes of delays

The project demonstrates that flight delays can be examined from multiple operational and external perspectives rather than through a single variable.

---

## 📁 Project Structure

```text
Flight Delay Analysis/
│
├── Dataset/
│   └── Flight_delay.csv
│
├── flight_delay_visualization.py
│
├── README.md
│
└── Project Documentation/
    └── Flight_Delay_Analysis_Project_Documentation.docx
```

> Update the folder/file names above if your actual VS Code project structure is different.

---

## 🚀 Future Scope

* Extend the analysis with larger or more recent datasets.
* Perform predictive analysis for flight delays.
* Build airport-level and route-level analysis.
* Investigate seasonal and monthly delay patterns.
* Add more interactive Tableau filters.
* Integrate weather or other external datasets.
* Develop machine-learning models for delay prediction.

---

## 🎓 Project Outcome

This project demonstrates an end-to-end **Data Analytics workflow**, including:

**Data Cleaning → Data Analysis → Data Visualization → Dashboard Development → Insights**

It showcases practical skills in Python, Pandas, exploratory data analysis, visualization, Tableau, and data storytelling.

---

### Skills Demonstrated

`Python` · `SQL` · `Excel` · `Pandas` · `NumPy` · `Data Visualization` · `Tableau` · `EDA`
