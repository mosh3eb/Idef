---
title: 'IDEF: An Interactive Data Exploration Framework for Multi-dimensional Scientific Datasets'
tags:
  - Python
  - data visualization
  - interactive exploration
  - multi-dimensional data
  - scientific datasets
authors:
  - name: Author Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Institution Name
   index: 1
date: 17 May 2025
bibliography: paper.bib
---

# Summary

Scientific research increasingly involves the analysis of complex, multi-dimensional datasets. Effective exploration and visualization of these datasets is crucial for gaining insights and making discoveries. The Interactive Data Exploration Framework (IDEF) is a Python library designed to democratize access to advanced data visualization techniques, enabling researchers to interactively explore multi-dimensional scientific datasets through both programmatic and graphical interfaces. IDEF bridges the gap between powerful visualization capabilities and ease of use, allowing researchers to focus on scientific questions rather than visualization implementation details.

# Statement of Need

Modern scientific research generates increasingly complex datasets with multiple dimensions, variables, and relationships. Examples include climate models with spatial, temporal, and multiple physical variables; genomic data with thousands of features across multiple samples and conditions; and neuroimaging data with spatial, temporal, and functional dimensions. Exploring these datasets effectively requires specialized visualization techniques that can reveal patterns, relationships, and anomalies across multiple dimensions simultaneously.

While numerous visualization libraries exist in the Python ecosystem, they often present significant barriers to researchers:

1. **Steep learning curves**: Libraries like Matplotlib, Plotly, and Bokeh offer powerful capabilities but require substantial programming knowledge to create effective visualizations.

2. **Fragmentation**: Different libraries excel at different visualization types, forcing researchers to learn multiple APIs and integration approaches.

3. **Limited interactivity**: Many libraries provide limited interactive capabilities, especially for exploring relationships across multiple dimensions.

4. **Lack of guidance**: Researchers often struggle to determine which visualization techniques are most appropriate for their specific data characteristics.

IDEF addresses these challenges by providing:

1. **Unified interface**: A consistent API across multiple visualization types and backends.

2. **Dual interfaces**: Both programmatic (Python API) and graphical (web dashboard) interfaces to accommodate different workflows and expertise levels.

3. **Rich interactivity**: Built-in support for linked views, brushing, filtering, and dynamic updates.

4. **Intelligent suggestions**: Automatic recommendation of appropriate visualization techniques based on data characteristics.

5. **Extensibility**: A plugin architecture that allows researchers to add custom visualizations, data connectors, and analysis methods.

By lowering the barriers to effective data visualization, IDEF enables researchers across disciplines to gain deeper insights from their multi-dimensional datasets, potentially accelerating scientific discovery.

# Architecture and Features

IDEF follows a modular, layered architecture with clear separation of concerns:

## Data Layer

The data layer provides a unified representation of multi-dimensional scientific datasets, building upon the xarray library [@hoyer2017xarray] for labeled multi-dimensional arrays. Key components include:

- **Dataset class**: A wrapper around xarray.Dataset that adds visualization metadata and convenience methods.
- **Data connectors**: Adapters for loading data from various file formats (NetCDF, HDF5, CSV, etc.) and sources.
- **Transformation pipeline**: A framework for processing and transforming data through operations like selection, aggregation, and normalization.

## Visualization Layer

The visualization layer handles the creation and rendering of visual representations, supporting multiple visualization types:

- **2D visualizations**: Scatter plots, line plots, heatmaps, and contour plots.
- **3D visualizations**: 3D scatter plots, surface plots, and volume renderings.
- **Multi-dimensional techniques**: Parallel coordinates, small multiples, and dimensionality reduction visualizations.

The visualization layer leverages existing libraries like Plotly [@plotly] and HoloViews [@stevens2015holoviews] while providing a consistent interface and enhanced interactivity.

## Application Layer

The application layer manages user sessions, configuration, and high-level functionality:

- **Explorer class**: The main entry point for programmatic interaction with IDEF.
- **Session management**: Persistence of exploration state for reproducibility.
- **Export capabilities**: Generation of shareable outputs in various formats.

## User Interfaces

IDEF provides two complementary interfaces:

- **Web dashboard**: A browser-based interface built with Panel [@panel] that allows for interactive exploration without writing code.
- **Python API**: A programmatic interface for scripted analysis and integration with existing workflows.

## Extension System

The extension system enables customization and extension through plugins:

- **Custom visualizations**: Addition of new visualization types.
- **Data connectors**: Support for additional data formats and sources.
- **Analysis methods**: Integration of domain-specific analysis capabilities.

# Example Usage

The following examples demonstrate IDEF's capabilities for exploring multi-dimensional scientific datasets.

## Programmatic Interface

```python
from idef.ui.api import load_data, visualize, show

# Load a climate dataset
dataset = load_data("climate_data.nc")

# Create a heatmap of temperature distribution
heatmap = visualize(dataset, "heatmap",
                   x="longitude",
                   y="latitude",
                   z="temperature")

# Show the visualization
show(heatmap)

# Create a time series of temperature averaged over a region
region = dataset.select(latitude=slice(30, 45), longitude=slice(-120, -100))
avg_temp = region.transform(lambda ds: ds.mean(dim=["latitude", "longitude"]))

time_series = visualize(avg_temp, "line",
                       x="time",
                       y="temperature")

show(time_series)
```

## Web Dashboard

The web dashboard provides a graphical interface for interactive exploration:

```python
from idef.ui.api import dashboard

# Launch the dashboard
dashboard()
```

Through the dashboard, users can:
- Load datasets from various sources
- Create and customize visualizations
- Interactively explore relationships through linked views
- Export visualizations for sharing or publication

# Impact and Conclusion

IDEF aims to democratize access to advanced data visualization techniques, enabling researchers across disciplines to gain deeper insights from their multi-dimensional datasets. By providing both programmatic and graphical interfaces, IDEF accommodates different workflows and expertise levels, making powerful visualization capabilities accessible to a broader audience.

The framework's modular architecture and extension system allow for customization and growth, ensuring that IDEF can adapt to evolving research needs and visualization techniques. By lowering the barriers to effective data visualization, IDEF has the potential to accelerate scientific discovery across domains that deal with complex, multi-dimensional data.

# Acknowledgements

We acknowledge contributions from the open-source community and the developers of the libraries upon which IDEF builds, including xarray, Plotly, HoloViews, Panel, and others.

# References
