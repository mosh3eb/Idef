# Interactive Data Exploration Framework (IDEF)

A Python framework for interactive exploration of multi-dimensional scientific datasets.

## Overview

The Interactive Data Exploration Framework (IDEF) enables researchers to interactively explore, visualize, and analyze complex multi-dimensional scientific datasets. The framework provides both programmatic and graphical interfaces for maximum flexibility.

## Features

- **Multi-format Data Support**: Import data from common scientific formats (CSV, HDF5, NetCDF, etc.)
- **Multi-dimensional Visualization**: Support for visualizing data with 3+ dimensions through various techniques
- **Interactive Elements**: Zooming, panning, brushing, dynamic filtering, and linked views
- **Statistical Analysis**: Basic statistical functions integrated directly into the visualization workflow
- **Extensible Architecture**: Plugin system for custom visualizations, data connectors, and analysis methods
- **Dual Interfaces**: Web-based GUI and programmatic Python API

## Installation

```bash
# Clone the repository
git clone https://github.com/username/idef.git
cd idef

# Install the package
pip install -e .
```

## Quick Start

### Programmatic API

```python
from idef.ui.api import load_data, visualize, show

# Load a dataset
dataset = load_data("path/to/data.nc")

# Create a visualization
viz = visualize(dataset, "scatter", x="temperature", y="pressure")

# Show the visualization
show(viz)
```

### Web Dashboard

```python
from idef.ui.api import dashboard

# Launch the interactive dashboard
dashboard()
```

## Documentation

- [User Guide](docs/user_guide.md)
- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
