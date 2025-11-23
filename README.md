# India-Hyper-Risk-Atlas-Map-X
India HyperRisk Atlas MapX is a state-level multi-hazard intelligence system combining ten hazards including earthquake, flood, cyclone, landslide, heatwave, drought, forest fire, air pollution and lightning. It generates a fused hazard score, radar profile and India-wide choropleth map.


India HyperRisk Atlas 10X

India HyperRisk Atlas 10X is a state-level multi-hazard intelligence dashboard designed to give a combined risk overview for all major natural hazards affecting India. The system integrates ten hazard indices, generates state-wise risk profiles, produces fused multi-hazard scores, displays a national choropleth map, and includes radar-based hazard fingerprints. It provides a compact, research-grade view of India’s environmental and disaster vulnerability.


>

Features

1. Multi-Hazard Integration

The application combines ten major hazards using normalized indices:

Earthquake

Flood

Cyclone

Tsunami

Landslide

Heatwave

Drought

Forest Fire

Air Pollution

Lightning


All values are user-provided or derived from open national datasets.

2. Fused Multi-Hazard Score

Each state receives a combined hazard score based on the mean value of all individual hazard indices. This provides a simplified overview of overall vulnerability.

3. Interactive Choropleth Map

A state-wise map of India shows the spatial distribution of multi-hazard intensity. The map automatically adapts to user-provided data and highlights high-risk zones.

4. Radar-Based Hazard Profile

Each state’s hazard fingerprint is shown using a radar chart, allowing comparison of hazard patterns across different regions.

5. Machine Learning Vulnerability Category

A Random Forest model uses all hazard indices to classify each state into:

Low

Moderate

High


This classification is based on fused hazard levels.

6. State-Wise Comparison Table

A sorted table displays all states by their overall hazard index for quick assessment and ranking.


Dataset Structure

The application uses a CSV file named india_state_multi_hazard.csv with the following structure:
state,eq_risk,flood_risk,cyclone_risk,tsunami_risk,landslide_risk,heatwave_risk,drought_risk,forestfire_risk,airpollution_risk,lightning_risk

Hazard values follow a normalized scale (0–100).
The file can be expanded to include additional states or updated with more accurate values.


>

GeoJSON Requirement

The application uses a GeoJSON file named india_states.geojson containing boundaries of Indian states.
Each feature must contain
properties.ST_NM
corresponding exactly to the state names used in the CSV file.
This ensures correct map rendering.


>

Installation

Clone the repository and install the required packages:

pip install -r requirements.txt

Run the application locally:

streamlit run app.py


>

Files Included

app.py – Main application file

requirements.txt – Python dependencies

india_state_multi_hazard.csv – Hazard index dataset

india_states.geojson – India state boundary file



>

How It Works

1. The CSV file is loaded and hazard values are processed.


2. A fused multi-hazard score is generated.


3. A Random Forest classifier predicts state vulnerability categories.


4. The GeoJSON file is read to render a choropleth map.


5. A radar chart and bar chart visualize hazard distribution for each state.


6. A comparison table shows the relative ranking of all states.




>

Use Cases

Multi-hazard risk assessment

Disaster management planning

Environmental vulnerability studies

Research projects in geoscience and climate science

Academic demonstrations for hazard integration methods



>

Purpose

This project serves as a research-oriented demonstration of multi-hazard integration for India, presenting a unified method to visualize where combined risks are highest and how different hazards interact across states. It helps communicate complex hazard information in a concise, visual and accessible form.


>

Author

Bikrant Kumar Mishra
GeoAI and Hazard Modelling
