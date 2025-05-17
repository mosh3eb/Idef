"""
Export Module for IDEF.

This module handles exporting and sharing of analysis results and visualizations.
"""

import os
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import zipfile
import base64

@dataclass
class ExportMetadata:
    """Metadata for exported content."""
    created_at: str
    content_type: str
    format: str
    version: str = "1.0"
    description: Optional[str] = None
    tags: List[str] = None

class ExportManager:
    """Manages export operations for analysis results and visualizations."""
    
    SUPPORTED_FORMATS = {
        'data': ['csv', 'json', 'pickle', 'parquet'],
        'visualization': ['html', 'png', 'svg', 'pdf'],
        'analysis': ['json', 'pickle']
    }
    
    def __init__(self, export_dir: Optional[str] = None):
        self.export_dir = export_dir or self._get_default_export_dir()
        os.makedirs(self.export_dir, exist_ok=True)
        
    def _get_default_export_dir(self) -> str:
        """Get the default export directory."""
        return os.path.join(os.path.expanduser('~'), '.idef', 'exports')
        
    def _create_metadata(self, content_type: str, 
                        format: str, description: str = None,
                        tags: List[str] = None) -> ExportMetadata:
        """Create metadata for export."""
        return ExportMetadata(
            created_at=datetime.now().isoformat(),
            content_type=content_type,
            format=format,
            description=description,
            tags=tags or []
        )
        
    def export_data(self, data: Any, filename: str,
                   format: str = 'csv', **kwargs) -> str:
        """Export data to file."""
        if format not in self.SUPPORTED_FORMATS['data']:
            raise ValueError(f"Unsupported data format: {format}")
            
        filepath = os.path.join(self.export_dir, filename)
        
        if format == 'csv':
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_csv(filepath, **kwargs)
            else:
                pd.DataFrame(data).to_csv(filepath, **kwargs)
        elif format == 'json':
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_json(filepath, **kwargs)
            else:
                with open(filepath, 'w') as f:
                    json.dump(data, f, **kwargs)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        elif format == 'parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, **kwargs)
            else:
                pd.DataFrame(data).to_parquet(filepath, **kwargs)
                
        return filepath
        
    def export_visualization(self, figure: Any, filename: str,
                           format: str = 'html', **kwargs) -> str:
        """Export visualization to file."""
        if format not in self.SUPPORTED_FORMATS['visualization']:
            raise ValueError(f"Unsupported visualization format: {format}")
            
        filepath = os.path.join(self.export_dir, filename)
        
        if hasattr(figure, 'write_html') and format == 'html':
            figure.write_html(filepath, **kwargs)
        elif hasattr(figure, 'write_image'):
            figure.write_image(filepath, **kwargs)
        elif hasattr(figure, 'savefig'):
            figure.savefig(filepath, format=format, **kwargs)
        else:
            raise ValueError("Unsupported figure type for export")
            
        return filepath
        
    def export_analysis_result(self, result: Any, filename: str,
                             format: str = 'json', **kwargs) -> str:
        """Export analysis result to file."""
        if format not in self.SUPPORTED_FORMATS['analysis']:
            raise ValueError(f"Unsupported analysis format: {format}")
            
        filepath = os.path.join(self.export_dir, filename)
        
        if format == 'json':
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif hasattr(result, '__dict__'):
                result_dict = asdict(result)
            else:
                result_dict = result
                
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, **kwargs)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(result, f, **kwargs)
                
        return filepath
        
    def create_export_bundle(self, items: List[Dict[str, Any]], 
                           bundle_name: str) -> str:
        """Create a bundle of exported items."""
        bundle_path = os.path.join(self.export_dir, bundle_name)
        
        with zipfile.ZipFile(bundle_path, 'w') as bundle:
            for item in items:
                content = item['content']
                filename = item['filename']
                content_type = item['type']
                format = item['format']
                description = item.get('description')
                tags = item.get('tags', [])
                
                # Export the content
                if content_type == 'data':
                    filepath = self.export_data(content, filename, format)
                elif content_type == 'visualization':
                    filepath = self.export_visualization(content, filename, format)
                elif content_type == 'analysis':
                    filepath = self.export_analysis_result(content, filename, format)
                else:
                    raise ValueError(f"Unsupported content type: {content_type}")
                    
                # Add to bundle
                bundle.write(filepath, filename)
                
                # Add metadata
                metadata = self._create_metadata(content_type, format, 
                                              description, tags)
                metadata_filename = f"{filename}.metadata.json"
                metadata_content = json.dumps(asdict(metadata))
                bundle.writestr(metadata_filename, metadata_content)
                
                # Clean up temporary file
                os.remove(filepath)
                
        return bundle_path
        
    def load_export_bundle(self, bundle_path: str, 
                         extract_dir: Optional[str] = None) -> Dict[str, Any]:
        """Load contents from an export bundle."""
        extract_dir = extract_dir or os.path.join(self.export_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        results = {}
        with zipfile.ZipFile(bundle_path, 'r') as bundle:
            # Extract all files
            bundle.extractall(extract_dir)
            
            # Process each file
            for filename in bundle.namelist():
                if filename.endswith('.metadata.json'):
                    continue
                    
                metadata_filename = f"{filename}.metadata.json"
                if metadata_filename not in bundle.namelist():
                    continue
                    
                # Load metadata
                with open(os.path.join(extract_dir, metadata_filename)) as f:
                    metadata = json.load(f)
                    
                # Load content based on type and format
                filepath = os.path.join(extract_dir, filename)
                content = self._load_content(filepath, metadata)
                
                results[filename] = {
                    'content': content,
                    'metadata': metadata
                }
                
        return results
        
    def _load_content(self, filepath: str, metadata: Dict) -> Any:
        """Load content based on metadata."""
        content_type = metadata['content_type']
        format = metadata['format']
        
        if content_type == 'data':
            if format == 'csv':
                return pd.read_csv(filepath)
            elif format == 'json':
                return pd.read_json(filepath)
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif format == 'parquet':
                return pd.read_parquet(filepath)
        elif content_type == 'analysis':
            if format == 'json':
                with open(filepath) as f:
                    return json.load(f)
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
                    
        # For visualizations, return the file path since they're static exports
        return filepath

# Global export manager instance
export_manager = ExportManager()
