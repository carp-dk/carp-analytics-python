import pandas as pd
import folium
from folium.plugins import HeatMap
from typing import Optional, List, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..reader import SleepinessData

from rich.console import Console

console = Console()


class ParticipantVisualizer:
    """
    Fluent API for visualizing participant data.
    Usage: sd.participant("email").visualize.location()
    """
    
    def __init__(self, sleepiness_data: 'SleepinessData', deployment_ids: Set[str], email: str):
        self._sd = sleepiness_data
        self._deployment_ids = deployment_ids
        self._email = email
    
    def location(
        self,
        output_file: Optional[str] = None,
        location_type: str = "dk.cachet.carp.geolocation",
        step_type: str = "dk.cachet.carp.stepcount",
        include_steps: bool = True,
        parquet_dir: Optional[str] = "output_parquet"
    ) -> Optional[str]:
        """
        Generate a location heatmap for this participant.
        
        Args:
            output_file: Output HTML file path. Defaults to "{email}_location.html"
            location_type: Data type for location data
            step_type: Data type for step count data
            include_steps: Whether to overlay step count markers
            parquet_dir: Directory with parquet files for faster loading
            
        Returns:
            Path to the generated HTML file, or None if no data found
        """
        if output_file is None:
            # Sanitize email for filename
            safe_email = self._email.replace("@", "_at_").replace(".", "_")
            output_file = f"{safe_email}_location.html"
        
        console.print(f"[bold blue]Generating location heatmap for {self._email}...[/bold blue]")
        
        # Load location data
        df_loc = self._sd.get_dataframe(location_type, parquet_dir)
        
        if df_loc is None or df_loc.empty:
            console.print(f"[bold red]No location data found for type {location_type}[/bold red]")
            return None
        
        # Filter by deployment IDs
        user_series_loc = self._get_field(df_loc, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
        if user_series_loc is not None:
            df_loc = df_loc[user_series_loc.isin(self._deployment_ids)]
        
        if df_loc.empty:
            console.print(f"[bold red]No location data found for {self._email}[/bold red]")
            return None
        
        # Load step data if requested
        df_steps = pd.DataFrame()
        if include_steps:
            df_steps_raw = self._sd.get_dataframe(step_type, parquet_dir)
            if df_steps_raw is not None and not df_steps_raw.empty:
                user_series_steps = self._get_field(df_steps_raw, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
                if user_series_steps is not None:
                    df_steps = df_steps_raw[user_series_steps.isin(self._deployment_ids)]
        
        # Extract coordinates
        df_loc['_lat'] = self._get_field(df_loc, ['measurement.data.latitude', 'latitude'])
        df_loc['_lon'] = self._get_field(df_loc, ['measurement.data.longitude', 'longitude'])
        df_loc['_time'] = self._get_field(df_loc, ['measurement.sensorStartTime', 'sensorStartTime'])
        
        if df_loc['_lat'].isnull().all() or df_loc['_lon'].isnull().all():
            console.print("[bold red]Could not find latitude/longitude columns in location data[/bold red]")
            return None
        
        # Extract step data
        if not df_steps.empty:
            df_steps['_steps'] = self._get_field(df_steps, ['measurement.data.steps', 'steps'])
            df_steps['_time'] = self._get_field(df_steps, ['measurement.sensorStartTime', 'sensorStartTime'])
        
        # Render the map
        self._render_map(df_loc, df_steps, output_file)
        return output_file
    
    def _get_field(self, df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        """Extract a series from dataframe using candidate field paths."""
        for path in candidates:
            if path in df.columns:
                return df[path]
            
            parts = path.split('.')
            if parts[0] in df.columns:
                try:
                    series = df[parts[0]]
                    for part in parts[1:]:
                        series = series.apply(lambda x: x.get(part) if isinstance(x, dict) else None)
                    return series
                except Exception:
                    pass
        return None
    
    def _render_map(self, df_loc: pd.DataFrame, df_steps: pd.DataFrame, output_file: str):
        """Render the heatmap to an HTML file."""
        df_loc = df_loc.dropna(subset=['_lat', '_lon'])
        
        if df_loc.empty:
            console.print("[bold red]No valid coordinates found after filtering[/bold red]")
            return
        
        center_lat = df_loc['_lat'].mean()
        center_lon = df_loc['_lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add heatmap layer
        heat_data = df_loc[['_lat', '_lon']].values.tolist()
        HeatMap(heat_data).add_to(m)
        
        # Add step markers
        if not df_steps.empty and '_steps' in df_steps.columns and '_time' in df_steps.columns:
            if '_time' in df_loc.columns:
                df_loc_sorted = df_loc.sort_values('_time')
                df_steps_sorted = df_steps.sort_values('_time')
                
                df_loc_sorted['_time'] = df_loc_sorted['_time'].astype('int64')
                df_steps_sorted['_time'] = df_steps_sorted['_time'].astype('int64')
                
                merged = pd.merge_asof(
                    df_steps_sorted,
                    df_loc_sorted[['_time', '_lat', '_lon']],
                    on='_time',
                    direction='nearest',
                    tolerance=300_000_000  # 5 minutes in microseconds
                )
                
                for _, row in merged.iterrows():
                    if pd.notnull(row['_lat']) and pd.notnull(row['_lon']) and pd.notnull(row['_steps']):
                        steps = row['_steps']
                        if steps > 0:
                            folium.CircleMarker(
                                location=[row['_lat'], row['_lon']],
                                radius=min(max(steps / 10, 3), 20),
                                popup=f"Steps: {steps}<br>Time: {row['_time']}",
                                color="blue",
                                fill=True,
                                fill_color="blue"
                            ).add_to(m)
        
        m.save(output_file)
        console.print(f"[bold green]Heatmap saved to {output_file}[/bold green]")


class LocationVisualizer:
    def __init__(self, sd: 'SleepinessData'):
        self.sd = sd

    def _get_field(self, df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        """
        Tries to extract a series from the dataframe using a list of candidate field paths.
        Supports dot-notation for nested dict columns.
        """
        for path in candidates:
            if path in df.columns:
                return df[path]
            
            # Try nested
            parts = path.split('.')
            if parts[0] in df.columns:
                try:
                    series = df[parts[0]]
                    for part in parts[1:]:
                        # Handle None/NaN
                        series = series.apply(lambda x: x.get(part) if isinstance(x, dict) else None)
                    return series
                except Exception:
                    pass
        return None

    def _render_map(self, df_loc: pd.DataFrame, df_steps: pd.DataFrame, output_file: str):
        """
        Internal method to render the map from prepared dataframes.
        Expects df_loc to have _lat, _lon, _time columns.
        Expects df_steps to have _steps, _time columns.
        """
        # Drop NaNs in location
        df_loc = df_loc.dropna(subset=['_lat', '_lon'])
        
        if df_loc.empty:
            console.print("[bold red]No valid coordinates found after filtering[/bold red]")
            return

        # Create Map
        center_lat = df_loc['_lat'].mean()
        center_lon = df_loc['_lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add Heatmap
        heat_data = df_loc[['_lat', '_lon']].values.tolist()
        HeatMap(heat_data).add_to(m)
        
        # Add Step Markers
        if not df_steps.empty:
            if '_steps' in df_steps.columns and '_time' in df_steps.columns and '_time' in df_loc.columns:
                # Sort by time
                df_loc = df_loc.sort_values('_time')
                df_steps = df_steps.sort_values('_time')
                
                # Ensure types match
                df_loc['_time'] = df_loc['_time'].astype('int64')
                df_steps['_time'] = df_steps['_time'].astype('int64')
                
                merged = pd.merge_asof(
                    df_steps, 
                    df_loc[['_time', '_lat', '_lon']], 
                    on='_time', 
                    direction='nearest',
                    tolerance=300_000_000 # 5 minutes in microseconds
                )
                
                for idx, row in merged.iterrows():
                    if pd.notnull(row['_lat']) and pd.notnull(row['_lon']) and pd.notnull(row['_steps']):
                        steps = row['_steps']
                        if steps > 0:
                            folium.CircleMarker(
                                location=[row['_lat'], row['_lon']],
                                radius=min(max(steps / 10, 3), 20),
                                popup=f"Steps: {steps}<br>Time: {row['_time']}",
                                color="blue",
                                fill=True,
                                fill_color="blue"
                            ).add_to(m)
                            
        # Save
        m.save(output_file)
        console.print(f"[bold green]Heatmap saved to {output_file}[/bold green]")

    def plot_heatmap_from_items(
        self,
        location_items: List[Any],
        step_items: Optional[List[Any]] = None,
        output_file: str = "user_heatmap.html"
    ):
        """
        Generates a heatmap from a list of type-safe objects (e.g. generated SleepinessItem).
        """
        console.print(f"[bold blue]Generating heatmap from {len(location_items)} location items...[/bold blue]")
        
        # Helper to safely get attributes
        def get_attr(obj, path):
            parts = path.split('.')
            curr = obj
            for p in parts:
                if curr is None:
                    return None
                curr = getattr(curr, p, None)
            return curr

        # Extract Location Data
        loc_data = []
        for item in location_items:
            lat = get_attr(item, 'measurement.data.latitude')
            lon = get_attr(item, 'measurement.data.longitude')
            time = get_attr(item, 'measurement.sensorStartTime')
            
            if lat is not None and lon is not None:
                loc_data.append({'_lat': lat, '_lon': lon, '_time': time})
        
        df_loc = pd.DataFrame(loc_data)
        
        if df_loc.empty:
            console.print("[bold red]No valid coordinates found in location items[/bold red]")
            return

        # Extract Step Data
        df_steps = pd.DataFrame()
        if step_items:
            step_data = []
            for item in step_items:
                steps = get_attr(item, 'measurement.data.steps')
                time = get_attr(item, 'measurement.sensorStartTime')
                if steps is not None:
                    step_data.append({'_steps': steps, '_time': time})
            df_steps = pd.DataFrame(step_data)
            
        self._render_map(df_loc, df_steps, output_file)

    def plot_user_heatmap(
        self, 
        study_deployment_id: str, 
        location_type: str = "dk.cachet.carp.geolocation",
        step_type: str = "dk.cachet.carp.stepcount",
        parquet_dir: Optional[str] = "output_parquet",
        output_file: str = "user_heatmap.html"
    ):
        """
        Generates a heatmap of user locations and overlays step count data.
        """
        console.print(f"[bold blue]Generating heatmap for user {study_deployment_id}...[/bold blue]")
        
        # 1. Load Data
        df_loc = self.sd.get_dataframe(location_type, parquet_dir)
        df_steps = self.sd.get_dataframe(step_type, parquet_dir)
        
        if df_loc is None or df_loc.empty:
            console.print(f"[bold red]No location data found for type {location_type}[/bold red]")
            return
            
        if df_steps is None:
            console.print(f"[yellow]No step data found for type {step_type}. Plotting location only.[/yellow]")
            df_steps = pd.DataFrame()

        # 2. Filter by User
        user_series_loc = self._get_field(df_loc, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
        if user_series_loc is not None:
            df_loc = df_loc[user_series_loc == study_deployment_id]
        
        if df_loc.empty:
            console.print(f"[bold red]No location data found for user {study_deployment_id}[/bold red]")
            return

        if not df_steps.empty:
            user_series_steps = self._get_field(df_steps, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
            if user_series_steps is not None:
                df_steps = df_steps[user_series_steps == study_deployment_id]

        # 3. Extract Coordinates and Time
        df_loc['_lat'] = self._get_field(df_loc, ['measurement.data.latitude', 'latitude'])
        df_loc['_lon'] = self._get_field(df_loc, ['measurement.data.longitude', 'longitude'])
        df_loc['_time'] = self._get_field(df_loc, ['measurement.sensorStartTime', 'sensorStartTime'])
        
        if df_loc['_lat'].isnull().all() or df_loc['_lon'].isnull().all():
            console.print("[bold red]Could not find latitude/longitude columns in location data[/bold red]")
            return
            
        # 6. Add Step Markers
        if not df_steps.empty:
            df_steps['_steps'] = self._get_field(df_steps, ['measurement.data.steps', 'steps'])
            df_steps['_time'] = self._get_field(df_steps, ['measurement.sensorStartTime', 'sensorStartTime'])
            
        self._render_map(df_loc, df_steps, output_file)

    def plot_participant_heatmap(
        self, 
        unified_participant_id: str, 
        location_type: str = "dk.cachet.carp.geolocation",
        step_type: str = "dk.cachet.carp.stepcount",
        parquet_dir: Optional[str] = "output_parquet",
        output_file: str = "participant_heatmap.html"
    ):
        """
        Generates a heatmap for a specific unified participant across all their deployments.
        This aggregates data from all phases/folders for the same participant.
        """
        # Get all deployment IDs for this participant
        participants = self.sd.participant_manager.get_unified_participant(unified_participant_id)
        if not participants:
            console.print(f"[bold red]No participant found with ID {unified_participant_id}[/bold red]")
            return
        
        deployment_ids = [p.study_deployment_id for p in participants]
        console.print(f"[bold blue]Generating heatmap for participant {unified_participant_id} "
                     f"({len(deployment_ids)} deployments)...[/bold blue]")
        
        # 1. Load Data
        df_loc = self.sd.get_dataframe(location_type, parquet_dir)
        df_steps = self.sd.get_dataframe(step_type, parquet_dir)
        
        if df_loc is None or df_loc.empty:
            console.print(f"[bold red]No location data found for type {location_type}[/bold red]")
            return
            
        if df_steps is None:
            console.print(f"[yellow]No step data found for type {step_type}. Plotting location only.[/yellow]")
            df_steps = pd.DataFrame()

        # 2. Filter by all User deployments
        user_series_loc = self._get_field(df_loc, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
        if user_series_loc is not None:
            df_loc = df_loc[user_series_loc.isin(deployment_ids)]
        
        if df_loc.empty:
            console.print(f"[bold red]No location data found for participant {unified_participant_id}[/bold red]")
            return

        if not df_steps.empty:
            user_series_steps = self._get_field(df_steps, ['studyDeploymentId', 'dataStream.studyDeploymentId'])
            if user_series_steps is not None:
                df_steps = df_steps[user_series_steps.isin(deployment_ids)]

        # 3. Extract Coordinates and Time
        df_loc['_lat'] = self._get_field(df_loc, ['measurement.data.latitude', 'latitude'])
        df_loc['_lon'] = self._get_field(df_loc, ['measurement.data.longitude', 'longitude'])
        df_loc['_time'] = self._get_field(df_loc, ['measurement.sensorStartTime', 'sensorStartTime'])
        
        if df_loc['_lat'].isnull().all() or df_loc['_lon'].isnull().all():
            console.print("[bold red]Could not find latitude/longitude columns in location data[/bold red]")
            return
            
        # 4. Add Step Markers
        if not df_steps.empty:
            df_steps['_steps'] = self._get_field(df_steps, ['measurement.data.steps', 'steps'])
            df_steps['_time'] = self._get_field(df_steps, ['measurement.sensorStartTime', 'sensorStartTime'])
            
        self._render_map(df_loc, df_steps, output_file)
