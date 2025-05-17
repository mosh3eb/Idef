"""
Dashboard module for the Interactive Data Exploration Framework.
Provides a web-based interface for interactive data exploration.
"""

from typing import Dict, List, Union, Optional, Any
import panel as pn
import param
import holoviews as hv
from bokeh.models import HoverTool

from ..app.session import Explorer
from ..visualization.components import Visualization

# Initialize Panel and HoloViews extensions
pn.extension('plotly')
hv.extension('bokeh')


class Dashboard(param.Parameterized):
    """
    Web-based dashboard for interactive data exploration.
    """
    
    def __init__(self, explorer: Explorer):
        """
        Initialize a dashboard.
        
        Args:
            explorer: Explorer instance to use for data and visualizations
        """
        super().__init__()
        self.explorer = explorer
        self.panels = {}
        self.layout = None
        self._initialize_dashboard()
    
    def _initialize_dashboard(self):
        """Initialize the dashboard layout and components."""
        # Create main layout
        self.layout = pn.template.MaterialTemplate(
            title="Interactive Data Exploration Framework",
            sidebar_width=300
        )
        
        # Add dataset selector to sidebar
        self.dataset_selector = pn.widgets.Select(
            name='Dataset',
            options=self.explorer.list_datasets() or ['No datasets loaded']
        )
        self.layout.sidebar.append(self.dataset_selector)
        
        # Add visualization type selector to sidebar
        self.viz_type_selector = pn.widgets.Select(
            name='Visualization Type',
            options=['scatter', 'line', 'heatmap', 'scatter3d', 'parallel']
        )
        self.layout.sidebar.append(self.viz_type_selector)
        
        # Add visualization parameters section
        self.viz_params = pn.Column(
            pn.pane.Markdown("## Visualization Parameters"),
            sizing_mode='stretch_width'
        )
        self.layout.sidebar.append(self.viz_params)
        
        # Add create visualization button
        self.create_viz_button = pn.widgets.Button(
            name='Create Visualization',
            button_type='primary'
        )
        self.create_viz_button.on_click(self._create_visualization)
        self.layout.sidebar.append(self.create_viz_button)
        
        # Add main content area
        self.content = pn.Column(
            pn.pane.Markdown("# Welcome to the Interactive Data Exploration Framework"),
            pn.pane.Markdown("Load a dataset and create visualizations to begin exploring your data."),
            sizing_mode='stretch_both'
        )
        self.layout.main.append(self.content)
        
        # Add data loader section
        self.data_loader = self._create_data_loader()
        self.layout.sidebar.insert(0, self.data_loader)
        
        # Add visualization list
        self.viz_list = pn.widgets.Select(
            name='Visualizations',
            options=['No visualizations created']
        )
        self.layout.sidebar.append(pn.pane.Markdown("## Saved Visualizations"))
        self.layout.sidebar.append(self.viz_list)
        
        # Add show visualization button
        self.show_viz_button = pn.widgets.Button(
            name='Show Selected Visualization',
            button_type='success'
        )
        self.show_viz_button.on_click(self._show_visualization)
        self.layout.sidebar.append(self.show_viz_button)
    
    def _create_data_loader(self):
        """Create the data loading interface."""
        file_input = pn.widgets.FileInput(accept='.csv,.nc,.xlsx,.json,.h5')
        load_button = pn.widgets.Button(name='Load Data', button_type='primary')
        dataset_name = pn.widgets.TextInput(name='Dataset Name', placeholder='Enter dataset name (optional)')
        
        def load_data(event):
            if file_input.value is None:
                return
            
            # Get filename and extension
            filename = file_input.filename
            _, ext = filename.rsplit('.', 1)
            
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
                tmp.write(file_input.value)
                tmp_path = tmp.name
            
            # Load the dataset
            name = dataset_name.value or None
            try:
                dataset = self.explorer.load_dataset(tmp_path, name=name)
                
                # Update dataset selector
                self.dataset_selector.options = self.explorer.list_datasets()
                self.dataset_selector.value = dataset.name
                
                # Update visualization parameters
                self._update_viz_params()
                
                # Show success message
                self.content.clear()
                self.content.append(pn.pane.Markdown(f"# Dataset '{dataset.name}' loaded successfully"))
                self.content.append(pn.pane.Markdown(f"Dimensions: {dataset.dimensions}"))
                self.content.append(pn.pane.Markdown(f"Variables: {dataset.variables}"))
                
                # Suggest visualizations
                suggestions = self.explorer.suggest_visualizations(dataset)
                if suggestions:
                    self.content.append(pn.pane.Markdown("## Suggested Visualizations"))
                    for suggestion in suggestions:
                        suggestion_button = pn.widgets.Button(
                            name=f"Create {suggestion['name']}",
                            button_type='success'
                        )
                        
                        # Create a closure to capture the suggestion
                        def create_suggested_viz(event, suggestion=suggestion):
                            viz = self.explorer.visualize(
                                dataset,
                                suggestion['type'],
                                **suggestion['params'],
                                name=suggestion['name']
                            )
                            self._display_visualization(viz)
                            self._update_viz_list()
                        
                        suggestion_button.on_click(create_suggested_viz)
                        self.content.append(suggestion_button)
            
            except Exception as e:
                self.content.clear()
                self.content.append(pn.pane.Markdown(f"# Error loading dataset"))
                self.content.append(pn.pane.Markdown(f"Error: {str(e)}"))
            
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        load_button.on_click(load_data)
        
        return pn.Column(
            pn.pane.Markdown("## Load Dataset"),
            file_input,
            dataset_name,
            load_button
        )
    
    def _update_viz_params(self):
        """Update visualization parameters based on selected dataset and visualization type."""
        dataset_name = self.dataset_selector.value
        viz_type = self.viz_type_selector.value
        
        # Clear current parameters
        self.viz_params.clear()
        self.viz_params.append(pn.pane.Markdown("## Visualization Parameters"))
        
        # If no dataset is selected, return
        if not dataset_name or dataset_name == 'No datasets loaded':
            self.viz_params.append(pn.pane.Markdown("Please select a dataset first."))
            return
        
        try:
            # Get dataset
            dataset = self.explorer.get_dataset(dataset_name)
            
            # Add parameters based on visualization type
            if viz_type == 'scatter':
                # X and Y selectors
                x_selector = pn.widgets.Select(name='X Axis', options=dataset.variables)
                y_selector = pn.widgets.Select(name='Y Axis', options=dataset.variables)
                color_selector = pn.widgets.Select(name='Color By', options=['None'] + dataset.variables)
                
                self.viz_params.extend([x_selector, y_selector, color_selector])
                
            elif viz_type == 'line':
                # X and Y selectors
                x_selector = pn.widgets.Select(name='X Axis', options=dataset.variables + dataset.dimensions)
                y_selector = pn.widgets.Select(name='Y Axis', options=dataset.variables)
                
                self.viz_params.extend([x_selector, y_selector])
                
            elif viz_type == 'heatmap':
                # X, Y, and Z selectors
                x_selector = pn.widgets.Select(name='X Axis', options=dataset.dimensions)
                y_selector = pn.widgets.Select(name='Y Axis', options=dataset.dimensions)
                z_selector = pn.widgets.Select(name='Z Value', options=dataset.variables)
                
                self.viz_params.extend([x_selector, y_selector, z_selector])
                
            elif viz_type == 'scatter3d':
                # X, Y, and Z selectors
                x_selector = pn.widgets.Select(name='X Axis', options=dataset.variables)
                y_selector = pn.widgets.Select(name='Y Axis', options=dataset.variables)
                z_selector = pn.widgets.Select(name='Z Axis', options=dataset.variables)
                color_selector = pn.widgets.Select(name='Color By', options=['None'] + dataset.variables)
                
                self.viz_params.extend([x_selector, y_selector, z_selector, color_selector])
                
            elif viz_type == 'parallel':
                # Dimensions selector
                dimensions_selector = pn.widgets.MultiSelect(
                    name='Dimensions',
                    options=dataset.variables,
                    size=min(5, len(dataset.variables))
                )
                color_selector = pn.widgets.Select(name='Color By', options=['None'] + dataset.variables)
                
                self.viz_params.extend([dimensions_selector, color_selector])
        
        except Exception as e:
            self.viz_params.append(pn.pane.Markdown(f"Error: {str(e)}"))
    
    def _create_visualization(self, event):
        """Create a visualization based on selected parameters."""
        dataset_name = self.dataset_selector.value
        viz_type = self.viz_type_selector.value
        
        # If no dataset is selected, return
        if not dataset_name or dataset_name == 'No datasets loaded':
            self.content.clear()
            self.content.append(pn.pane.Markdown("# Error creating visualization"))
            self.content.append(pn.pane.Markdown("Please select a dataset first."))
            return
        
        try:
            # Get dataset
            dataset = self.explorer.get_dataset(dataset_name)
            
            # Get parameters based on visualization type
            params = {}
            
            if viz_type == 'scatter':
                params['x'] = self.viz_params[1].value
                params['y'] = self.viz_params[2].value
                if self.viz_params[3].value != 'None':
                    params['color'] = self.viz_params[3].value
                
            elif viz_type == 'line':
                params['x'] = self.viz_params[1].value
                params['y'] = self.viz_params[2].value
                
            elif viz_type == 'heatmap':
                params['x'] = self.viz_params[1].value
                params['y'] = self.viz_params[2].value
                params['z'] = self.viz_params[3].value
                
            elif viz_type == 'scatter3d':
                params['x'] = self.viz_params[1].value
                params['y'] = self.viz_params[2].value
                params['z'] = self.viz_params[3].value
                if self.viz_params[4].value != 'None':
                    params['color'] = self.viz_params[4].value
                
            elif viz_type == 'parallel':
                params['dimensions'] = self.viz_params[1].value
                if self.viz_params[2].value != 'None':
                    params['color'] = self.viz_params[2].value
            
            # Create visualization
            viz = self.explorer.visualize(dataset, viz_type, **params)
            
            # Display visualization
            self._display_visualization(viz)
            
            # Update visualization list
            self._update_viz_list()
            
        except Exception as e:
            self.content.clear()
            self.content.append(pn.pane.Markdown("# Error creating visualization"))
            self.content.append(pn.pane.Markdown(f"Error: {str(e)}"))
    
    def _display_visualization(self, viz: Visualization):
        """Display a visualization in the content area."""
        self.content.clear()
        
        # Add title
        self.content.append(pn.pane.Markdown(f"# {viz.name}"))
        
        # Render visualization
        fig = viz.render()
        
        # Display based on type
        if hasattr(fig, 'to_bokeh'):
            # HoloViews object
            self.content.append(pn.pane.HoloViews(fig))
        else:
            # Assume Plotly figure
            self.content.append(pn.pane.Plotly(fig))
        
        # Add export button
        export_button = pn.widgets.Button(name='Export Visualization', button_type='success')
        
        def export_viz(event):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                tmp_path = tmp.name
            
            self.explorer.export(tmp_path, viz)
            
            # Create download link
            download_link = pn.widgets.FileDownload(
                file=tmp_path,
                filename=f"{viz.name.replace(' ', '_')}.html",
                button_type='success',
                label='Download Visualization'
            )
            
            # Replace export button with download link
            self.content[-1] = download_link
        
        export_button.on_click(export_viz)
        self.content.append(export_button)
    
    def _show_visualization(self, event):
        """Show the selected visualization."""
        viz_name = self.viz_list.value
        
        # If no visualization is selected, return
        if not viz_name or viz_name == 'No visualizations created':
            return
        
        try:
            # Get visualization
            viz = self.explorer.get_visualization(viz_name)
            
            # Display visualization
            self._display_visualization(viz)
            
        except Exception as e:
            self.content.clear()
            self.content.append(pn.pane.Markdown("# Error showing visualization"))
            self.content.append(pn.pane.Markdown(f"Error: {str(e)}"))
    
    def _update_viz_list(self):
        """Update the visualization list."""
        viz_list = self.explorer.list_visualizations()
        if viz_list:
            self.viz_list.options = viz_list
            self.viz_list.value = viz_list[0]
        else:
            self.viz_list.options = ['No visualizations created']
    
    def show(self):
        """
        Show the dashboard.
        
        Returns:
            panel.Template: The dashboard layout
        """
        # Update visualization parameters based on current selections
        self.viz_type_selector.param.watch(
            lambda event: self._update_viz_params(),
            'value'
        )
        self.dataset_selector.param.watch(
            lambda event: self._update_viz_params(),
            'value'
        )
        
        return self.layout
    
    def export
(Content truncated due to size limit. Use line ranges to read in chunks)