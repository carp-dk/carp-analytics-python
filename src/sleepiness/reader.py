import ijson
from pathlib import Path
from typing import Generator, Any, Dict, Optional, List, Set
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import json
from dataclasses import dataclass, field

console = Console()


@dataclass
class ParticipantInfo:
    """Represents participant information from participant-data.json"""
    study_deployment_id: str
    role_name: str = "Participant"
    full_name: Optional[str] = None
    sex: Optional[str] = None
    ssn: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None
    consent_signed: bool = False
    consent_timestamp: Optional[str] = None
    source_folder: Optional[str] = None
    # Unified participant ID assigned when same participant is detected across folders
    unified_participant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_deployment_id": self.study_deployment_id,
            "role_name": self.role_name,
            "full_name": self.full_name,
            "sex": self.sex,
            "ssn": self.ssn,
            "user_id": self.user_id,
            "email": self.email,
            "consent_signed": self.consent_signed,
            "consent_timestamp": self.consent_timestamp,
            "source_folder": self.source_folder,
            "unified_participant_id": self.unified_participant_id,
        }


class ParticipantManager:
    """
    Manages participant data across multiple data folders.
    Links participants across folders using SSN or user ID as identifiers.
    """
    
    def __init__(self):
        # studyDeploymentId -> ParticipantInfo
        self.participants_by_deployment: Dict[str, ParticipantInfo] = {}
        # unified_participant_id -> list of ParticipantInfo (same person across folders)
        self.unified_participants: Dict[str, List[ParticipantInfo]] = {}
        # For generating unified IDs
        self._unified_id_counter = 0
    
    def load_participant_data(self, data_folders: List[Path]):
        """
        Loads participant data from participant-data.json files in each data folder.
        """
        console.print(f"[bold blue]Loading participant data from {len(data_folders)} folders...[/bold blue]")
        
        for folder in data_folders:
            participant_file = folder / "participant-data.json"
            if participant_file.exists():
                self._load_single_file(participant_file, folder.name)
            else:
                console.print(f"[yellow]No participant-data.json found in {folder}[/yellow]")
        
        # After loading all, unify participants
        self._unify_participants()
        
        console.print(f"[bold green]Loaded {len(self.participants_by_deployment)} participant deployments, "
                     f"{len(self.unified_participants)} unique participants[/bold green]")
    
    def _load_single_file(self, file_path: Path, folder_name: str):
        """Load participant data from a single file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")
            return
        
        for entry in data:
            deployment_id = entry.get("studyDeploymentId")
            if not deployment_id:
                continue
            
            roles = entry.get("roles", [])
            for role in roles:
                role_name = role.get("roleName", "Unknown")
                role_data = role.get("data", {})
                
                # Extract participant info from various fields
                participant = ParticipantInfo(
                    study_deployment_id=deployment_id,
                    role_name=role_name,
                    source_folder=folder_name,
                )
                
                # Extract full name (can be a dict with firstName/lastName or a string)
                full_name_data = role_data.get("dk.carp.webservices.input.full_name")
                if full_name_data:
                    if isinstance(full_name_data, dict):
                        # Combine firstName, middleName, lastName
                        parts = []
                        if full_name_data.get("firstName"):
                            parts.append(full_name_data["firstName"])
                        if full_name_data.get("middleName"):
                            parts.append(full_name_data["middleName"])
                        if full_name_data.get("lastName"):
                            parts.append(full_name_data["lastName"])
                        if parts:
                            participant.full_name = " ".join(parts)
                    elif isinstance(full_name_data, str):
                        participant.full_name = full_name_data
                
                # Extract sex
                sex_data = role_data.get("dk.cachet.carp.input.sex")
                if sex_data:
                    participant.sex = sex_data
                
                # Extract SSN (can be a dict with socialSecurityNumber or a string)
                ssn_data = role_data.get("dk.carp.webservices.input.ssn")
                if ssn_data:
                    if isinstance(ssn_data, dict):
                        ssn_value = ssn_data.get("socialSecurityNumber")
                        if ssn_value:
                            participant.ssn = str(ssn_value)
                    elif isinstance(ssn_data, str):
                        participant.ssn = ssn_data
                
                # Extract consent info
                consent_data = role_data.get("dk.carp.webservices.input.informed_consent")
                if consent_data:
                    participant.consent_signed = True
                    if isinstance(consent_data, dict):
                        participant.consent_timestamp = consent_data.get("signedTimestamp")
                        participant.user_id = consent_data.get("userId")
                        participant.email = consent_data.get("name")  # email is stored in "name" field
                        
                        # Extract name from consent signature if not already set
                        if not participant.full_name:
                            consent_json_str = consent_data.get("consent")
                            if consent_json_str and isinstance(consent_json_str, str):
                                try:
                                    consent_doc = json.loads(consent_json_str)
                                    signature = consent_doc.get("signature", {})
                                    if isinstance(signature, dict):
                                        first_name = (signature.get("firstName") or "").strip()
                                        last_name = (signature.get("lastName") or "").strip()
                                        if first_name or last_name:
                                            participant.full_name = f"{first_name} {last_name}".strip()
                                except json.JSONDecodeError:
                                    pass
                
                self.participants_by_deployment[deployment_id] = participant
    
    def _unify_participants(self):
        """
        Identify the same participant across different folders/deployments.
        Uses email as primary identifier (most accurate), falls back to SSN, then full name.
        """
        # Group by identifier
        by_email: Dict[str, List[ParticipantInfo]] = defaultdict(list)
        by_ssn: Dict[str, List[ParticipantInfo]] = defaultdict(list)
        by_name: Dict[str, List[ParticipantInfo]] = defaultdict(list)
        
        for p in self.participants_by_deployment.values():
            # Email, SSN, name must be strings for use as dict keys
            if p.email and isinstance(p.email, str):
                by_email[p.email.lower()].append(p)  # normalize email to lowercase
            if p.ssn and isinstance(p.ssn, str):
                by_ssn[p.ssn].append(p)
            if p.full_name and isinstance(p.full_name, str):
                by_name[p.full_name.strip().lower()].append(p)  # normalize name
        
        # Assign unified IDs, preferring email grouping (most accurate)
        assigned: Set[str] = set()  # deployment IDs already assigned
        
        # First pass: use email (most accurate identifier)
        for email, participants in by_email.items():
            unified_id = f"P{self._unified_id_counter:04d}"
            self._unified_id_counter += 1
            
            for p in participants:
                if p.study_deployment_id not in assigned:
                    p.unified_participant_id = unified_id
                    assigned.add(p.study_deployment_id)
            
            self.unified_participants[unified_id] = participants
        
        # Second pass: use SSN for remaining
        for ssn, participants in by_ssn.items():
            unassigned = [p for p in participants if p.study_deployment_id not in assigned]
            if not unassigned:
                continue
            
            unified_id = f"P{self._unified_id_counter:04d}"
            self._unified_id_counter += 1
            
            for p in unassigned:
                p.unified_participant_id = unified_id
                assigned.add(p.study_deployment_id)
            
            self.unified_participants[unified_id] = unassigned
        
        # Third pass: use full name for remaining
        for name, participants in by_name.items():
            unassigned = [p for p in participants if p.study_deployment_id not in assigned]
            if not unassigned:
                continue
            
            unified_id = f"P{self._unified_id_counter:04d}"
            self._unified_id_counter += 1
            
            for p in unassigned:
                p.unified_participant_id = unified_id
                assigned.add(p.study_deployment_id)
            
            self.unified_participants[unified_id] = unassigned
        
        # Remaining participants get their own unified ID
        for p in self.participants_by_deployment.values():
            if p.study_deployment_id not in assigned:
                unified_id = f"P{self._unified_id_counter:04d}"
                self._unified_id_counter += 1
                p.unified_participant_id = unified_id
                self.unified_participants[unified_id] = [p]
        
        # Propagate name/SSN data across unified participants
        # If any deployment has name/SSN, share it with all deployments of same participant
        self._propagate_participant_data()
    
    def _propagate_participant_data(self):
        """
        Propagate name, SSN, and other data to all records of the same unified participant.
        If one deployment has data that others don't, copy it to all.
        """
        for unified_id, participants in self.unified_participants.items():
            # Collect best available data from all records
            best_full_name = None
            best_ssn = None
            best_sex = None
            
            for p in participants:
                if p.full_name and isinstance(p.full_name, str) and not best_full_name:
                    best_full_name = p.full_name
                if p.ssn and isinstance(p.ssn, str) and not best_ssn:
                    best_ssn = p.ssn
                if p.sex and not best_sex:
                    best_sex = p.sex
            
            # Apply to all records
            for p in participants:
                if best_full_name and not p.full_name:
                    p.full_name = best_full_name
                if best_ssn and not p.ssn:
                    p.ssn = best_ssn
                if best_sex and not p.sex:
                    p.sex = best_sex
    
    def get_participant(self, study_deployment_id: str) -> Optional[ParticipantInfo]:
        """Get participant info by study deployment ID."""
        return self.participants_by_deployment.get(study_deployment_id)
    
    def get_unified_participant(self, unified_id: str) -> List[ParticipantInfo]:
        """Get all deployments for a unified participant."""
        return self.unified_participants.get(unified_id, [])
    
    def find_by_email(self, email: str) -> List[ParticipantInfo]:
        """Find all participant records matching an email address."""
        email_lower = email.lower()
        return [p for p in self.participants_by_deployment.values() 
                if p.email and p.email.lower() == email_lower]
    
    def find_by_ssn(self, ssn: str) -> List[ParticipantInfo]:
        """Find all participant records matching an SSN."""
        return [p for p in self.participants_by_deployment.values() 
                if p.ssn and p.ssn == ssn]
    
    def find_by_name(self, name: str) -> List[ParticipantInfo]:
        """Find all participant records matching a full name (case-insensitive)."""
        name_lower = name.strip().lower()
        return [p for p in self.participants_by_deployment.values() 
                if p.full_name and p.full_name.strip().lower() == name_lower]
    
    def get_deployment_ids_by_email(self, email: str) -> List[str]:
        """Get all deployment IDs for a participant by email."""
        return [p.study_deployment_id for p in self.find_by_email(email)]
    
    def get_deployment_ids_by_ssn(self, ssn: str) -> List[str]:
        """Get all deployment IDs for a participant by SSN."""
        return [p.study_deployment_id for p in self.find_by_ssn(ssn)]
    
    def get_deployment_ids_by_name(self, name: str) -> List[str]:
        """Get all deployment IDs for a participant by name."""
        return [p.study_deployment_id for p in self.find_by_name(name)]
    
    def print_summary(self):
        """Print a summary table of participants."""
        table = Table(title="Participants Summary")
        table.add_column("Unified ID", style="cyan")
        table.add_column("Deployments", style="magenta")
        table.add_column("Folders", style="green")
        table.add_column("Email", style="yellow")
        table.add_column("SSN", style="red")
        table.add_column("Full Name", style="white")
        
        for unified_id, participants in self.unified_participants.items():
            folders = set(p.source_folder for p in participants if p.source_folder)
            emails = set(p.email for p in participants if p.email and isinstance(p.email, str))
            ssns = set(p.ssn for p in participants if p.ssn and isinstance(p.ssn, str))
            names = set(p.full_name for p in participants if p.full_name and isinstance(p.full_name, str))
            table.add_row(
                unified_id,
                str(len(participants)),
                ", ".join(sorted(folders)),
                ", ".join(emails) if emails else "N/A",
                ", ".join(ssns) if ssns else "N/A",
                ", ".join(names) if names else "N/A",
            )
        
        console.print(table)


class ParticipantAccessor:
    """
    Fluent API for accessing participant data.
    Usage: sd.participant("email@example.com").info(), .all_data(), .available_fields()
           sd.participant("email@example.com").visualize.location()
    """
    
    def __init__(self, sleepiness_data: 'SleepinessData', email: str):
        self._sd = sleepiness_data
        self._email = email
        self._participants = sleepiness_data.participant_manager.find_by_email(email)
        self._deployment_ids = set(
            sleepiness_data.participant_manager.get_deployment_ids_by_email(email)
        )
        self._visualizer = None
    
    @property
    def exists(self) -> bool:
        """Check if participant exists."""
        return len(self._participants) > 0
    
    @property
    def visualize(self):
        """
        Access visualization methods for this participant.
        Usage: sd.participant("email").visualize.location()
        """
        if self._visualizer is None:
            from .plotting import ParticipantVisualizer
            self._visualizer = ParticipantVisualizer(self._sd, self._deployment_ids, self._email)
        return self._visualizer
    
    def info(self) -> Optional[Dict[str, Any]]:
        """
        Get participant information as a dictionary.
        Returns combined info from all deployments for this participant.
        """
        if not self._participants:
            return None
        
        # Get first participant as base
        base = self._participants[0]
        
        # Combine info from all records
        all_folders = set()
        all_deployment_ids = set()
        
        for p in self._participants:
            if p.source_folder:
                all_folders.add(p.source_folder)
            all_deployment_ids.add(p.study_deployment_id)
        
        return {
            "email": self._email,
            "unified_id": base.unified_participant_id,
            "full_name": base.full_name,
            "ssn": base.ssn,
            "sex": base.sex,
            "user_id": base.user_id,
            "consent_signed": base.consent_signed,
            "consent_timestamp": base.consent_timestamp,
            "folders": sorted(all_folders),
            "deployment_ids": sorted(all_deployment_ids),
            "num_deployments": len(all_deployment_ids),
        }
    
    def print_info(self):
        """Print participant information in a formatted table."""
        info = self.info()
        if not info:
            console.print(f"[red]No participant found with email: {self._email}[/red]")
            return
        
        table = Table(title=f"Participant: {self._email}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in info.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            table.add_row(key, str(value) if value is not None else "N/A")
        
        console.print(table)
    
    def all_data(self, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Get all data items for this participant.
        Optionally filter by data type (e.g., "dk.cachet.carp.stepcount").
        """
        yield from self._sd._get_data_by_deployment_ids(self._deployment_ids, data_type)
    
    def available_fields(self, sample_size: int = 100) -> Set[str]:
        """
        Discover all available fields in this participant's data.
        Scans a sample of records and returns field paths in dot-notation.
        """
        fields = set()
        count = 0
        
        for item in self.all_data():
            if count >= sample_size:
                break
            self._collect_fields(item, "", fields)
            count += 1
        
        return fields
    
    def _collect_fields(self, obj: Any, prefix: str, fields: Set[str]):
        """Recursively collect field paths."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                fields.add(path)
                self._collect_fields(value, path, fields)
        elif isinstance(obj, list) and obj:
            # Sample first item in list
            self._collect_fields(obj[0], f"{prefix}[]", fields)
    
    def print_available_fields(self, sample_size: int = 100):
        """Print all available fields in a formatted list."""
        fields = self.available_fields(sample_size)
        console.print(f"[bold]Available fields for {self._email}:[/bold]")
        for f in sorted(fields):
            console.print(f"  - {f}")
    
    def data_types(self) -> Set[str]:
        """Get all unique data types for this participant."""
        types = set()
        for item in self.all_data():
            data_stream = item.get("dataStream", {})
            data_type = data_stream.get("dataType", {})
            type_name = data_type.get("name")
            if type_name:
                types.add(type_name)
        return types
    
    def print_data_types(self):
        """Print all data types available for this participant."""
        types = self.data_types()
        console.print(f"[bold]Data types for {self._email}:[/bold]")
        for t in sorted(types):
            console.print(f"  - {t}")
    
    def count(self, data_type: Optional[str] = None) -> int:
        """Count total data items for this participant."""
        return sum(1 for _ in self.all_data(data_type))
    
    def dataframe(self, data_type: str, parquet_dir: Optional[str] = None):
        """
        Get a pandas DataFrame of this participant's data for a specific type.
        Uses parquet files if available and parquet_dir is specified.
        """
        try:
            import pandas as pd
        except ImportError:
            console.print("[red]pandas is required for dataframe(). Install with: pip install pandas[/red]")
            return None
        
        if parquet_dir:
            # Try to load from parquet and filter
            df = self._sd.get_dataframe(data_type, parquet_dir)
            if df is not None and not df.empty:
                return df[df['studyDeploymentId'].isin(self._deployment_ids)]
        
        # Fall back to streaming
        items = list(self.all_data(data_type))
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items)


class SleepinessData:
    def __init__(self, file_paths: str | Path | List[str | Path], load_participants: bool = True):
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        self.file_paths = [Path(p) for p in file_paths]
        for p in self.file_paths:
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
        
        self.schema_cache = {}
        self.participant_manager = ParticipantManager()
        
        # Auto-detect and load participant data from parent folders
        if load_participants:
            self._auto_load_participants()
    
    def _auto_load_participants(self):
        """
        Automatically detect and load participant data from the data folders
        containing the input files.
        """
        data_folders = set()
        for file_path in self.file_paths:
            # Each file is typically in a phase folder like data/phase-1-1/data-streams.json
            parent = file_path.parent
            if (parent / "participant-data.json").exists():
                data_folders.add(parent)
        
        if data_folders:
            self.participant_manager.load_participant_data(list(data_folders))
    
    def load_participants_from_folders(self, folders: List[str | Path]):
        """
        Manually load participant data from specific folders.
        Useful when files are in a different location than the input data.
        """
        folder_paths = [Path(f) for f in folders]
        self.participant_manager.load_participant_data(folder_paths)
    
    def participant(self, email: str) -> ParticipantAccessor:
        """
        Access participant data via email using a fluent API.
        
        Usage:
            sd.participant("email@example.com").info()
            sd.participant("email@example.com").all_data()
            sd.participant("email@example.com").available_fields()
            sd.participant("email@example.com").data_types()
            sd.participant("email@example.com").dataframe("dk.cachet.carp.stepcount")
        """
        return ParticipantAccessor(self, email)
    
    def get_participant(self, study_deployment_id: str) -> Optional[ParticipantInfo]:
        """Get participant info by study deployment ID."""
        return self.participant_manager.get_participant(study_deployment_id)
    
    def find_participant_by_email(self, email: str) -> List[ParticipantInfo]:
        """Find all participant records matching an email address."""
        return self.participant_manager.find_by_email(email)
    
    def find_participant_by_ssn(self, ssn: str) -> List[ParticipantInfo]:
        """Find all participant records matching an SSN."""
        return self.participant_manager.find_by_ssn(ssn)
    
    def find_participant_by_name(self, name: str) -> List[ParticipantInfo]:
        """Find all participant records matching a full name."""
        return self.participant_manager.find_by_name(name)
    
    def get_data_by_email(self, email: str, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Get all data items for a participant identified by email.
        Optionally filter by data type.
        """
        deployment_ids = set(self.participant_manager.get_deployment_ids_by_email(email))
        yield from self._get_data_by_deployment_ids(deployment_ids, data_type)
    
    def get_data_by_ssn(self, ssn: str, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Get all data items for a participant identified by SSN.
        Optionally filter by data type.
        """
        deployment_ids = set(self.participant_manager.get_deployment_ids_by_ssn(ssn))
        yield from self._get_data_by_deployment_ids(deployment_ids, data_type)
    
    def get_data_by_name(self, name: str, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Get all data items for a participant identified by full name.
        Optionally filter by data type.
        """
        deployment_ids = set(self.participant_manager.get_deployment_ids_by_name(name))
        yield from self._get_data_by_deployment_ids(deployment_ids, data_type)
    
    def _get_data_by_deployment_ids(self, deployment_ids: set, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Internal helper to filter data by deployment IDs and optionally by type."""
        if not deployment_ids:
            return
        
        for item in self._get_item_generator():
            item_deployment_id = item.get('studyDeploymentId')
            if not item_deployment_id:
                item_deployment_id = item.get('dataStream', {}).get('studyDeploymentId')
            
            if item_deployment_id not in deployment_ids:
                continue
            
            if data_type:
                dt = item.get('dataStream', {}).get('dataType', {})
                target_namespace, target_name = data_type.rsplit('.', 1)
                if dt.get('name') != target_name or dt.get('namespace') != target_namespace:
                    continue
            
            yield item
    
    def print_participants(self):
        """Print a summary of all participants."""
        self.participant_manager.print_summary()

    def _get_item_generator(self) -> Generator[Dict[str, Any], None, None]:
        """
        Returns a generator that yields items from the JSON files.
        Uses ijson for memory-efficient streaming.
        """
        for file_path in self.file_paths:
            with open(file_path, 'rb') as f:
                # Assuming the file is a list of objects. 
                # 'item' matches objects in a list.
                # use_float=True ensures numbers are floats, avoiding Decimal schema mismatches in PyArrow
                yield from ijson.items(f, 'item', use_float=True)

    def _get_item_generator_with_participant(self, include_participant: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Returns a generator that yields items from the JSON files,
        optionally enriched with participant info.
        """
        for item in self._get_item_generator():
            if include_participant:
                deployment_id = item.get('studyDeploymentId')
                if not deployment_id:
                    deployment_id = item.get('dataStream', {}).get('studyDeploymentId')
                
                if deployment_id:
                    participant = self.participant_manager.get_participant(deployment_id)
                    if participant:
                        item = item.copy()  # Don't mutate original
                        item['_participant'] = participant.to_dict()
            
            yield item

    def get_data_with_participants(self, data_type: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Yields items enriched with participant information.
        If data_type is specified, filters to that type.
        """
        gen = self._get_item_generator_with_participant(include_participant=True)
        
        if data_type:
            target_namespace, target_name = data_type.rsplit('.', 1)
            for item in gen:
                dt = item.get('dataStream', {}).get('dataType', {})
                if dt.get('name') == target_name and dt.get('namespace') == target_namespace:
                    yield item
        else:
            yield from gen

    def group_by_participant(self, output_dir: str | Path, data_type: Optional[str] = None):
        """
        Groups data by unified participant ID and exports each group to a separate JSON file.
        Useful for analyzing individual participant data across all phases.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Grouping data by participant into {output_dir}...[/bold blue]")
        
        files = {}
        gen = self.get_data_with_participants(data_type)
        
        try:
            for item in tqdm(gen, desc="Grouping by participant"):
                participant_info = item.get('_participant', {})
                unified_id = participant_info.get('unified_participant_id', 'unknown')
                
                if unified_id not in files:
                    f = open(output_dir / f"{unified_id}.json", 'w')
                    f.write('[')
                    files[unified_id] = {'handle': f, 'first': True}
                
                f_info = files[unified_id]
                if not f_info['first']:
                    f_info['handle'].write(',')
                json.dump(item, f_info['handle'])
                f_info['first'] = False
                
        finally:
            for f_info in files.values():
                f_info['handle'].write(']')
                f_info['handle'].close()
                
        console.print(f"[bold green]Grouping complete! Created {len(files)} participant files.[/bold green]")

    def group_by_email(self, output_dir: str | Path, data_type: Optional[str] = None):
        """
        Groups data by participant email and exports each group to a separate JSON file.
        """
        self._group_by_field_value(output_dir, 'email', data_type)
    
    def group_by_ssn(self, output_dir: str | Path, data_type: Optional[str] = None):
        """
        Groups data by participant SSN and exports each group to a separate JSON file.
        """
        self._group_by_field_value(output_dir, 'ssn', data_type)
    
    def group_by_name(self, output_dir: str | Path, data_type: Optional[str] = None):
        """
        Groups data by participant full name and exports each group to a separate JSON file.
        """
        self._group_by_field_value(output_dir, 'full_name', data_type)
    
    def _group_by_field_value(self, output_dir: str | Path, field: str, data_type: Optional[str] = None):
        """Internal helper to group data by a participant field (email, ssn, or full_name)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Grouping data by {field} into {output_dir}...[/bold blue]")
        
        files = {}
        gen = self.get_data_with_participants(data_type)
        
        try:
            for item in tqdm(gen, desc=f"Grouping by {field}"):
                participant_info = item.get('_participant', {})
                value = participant_info.get(field, 'unknown')
                
                if not value or not isinstance(value, str):
                    value = 'unknown'
                
                # Sanitize filename
                safe_value = "".join(c for c in value if c.isalnum() or c in ('-', '_', '@', '.')).strip()
                if not safe_value:
                    safe_value = "unknown"
                    
                if safe_value not in files:
                    f = open(output_dir / f"{safe_value}.json", 'w')
                    f.write('[')
                    files[safe_value] = {'handle': f, 'first': True}
                
                f_info = files[safe_value]
                if not f_info['first']:
                    f_info['handle'].write(',')
                json.dump(item, f_info['handle'])
                f_info['first'] = False
                
        finally:
            for f_info in files.values():
                f_info['handle'].write(']')
                f_info['handle'].close()
                
        console.print(f"[bold green]Grouping complete! Created {len(files)} files.[/bold green]")

    def get_dataframe_with_participants(self, data_type: str, parquet_dir: Optional[str | Path] = None):
        """
        Returns a pandas DataFrame for the specified data type, enriched with participant info.
        Adds columns: participant_id, participant_email, participant_folder
        """
        try:
            import pandas as pd
        except ImportError:
            console.print("[bold red]pandas is required for DataFrame conversion.[/bold red]")
            return None

        # Get base dataframe
        df = self.get_dataframe(data_type, parquet_dir)
        if df is None or df.empty:
            return df
        
        # Add participant columns
        def get_participant_info(deployment_id):
            p = self.participant_manager.get_participant(deployment_id)
            if p:
                return pd.Series({
                    'participant_id': p.unified_participant_id,
                    'participant_email': p.email,
                    'participant_folder': p.source_folder
                })
            return pd.Series({
                'participant_id': None,
                'participant_email': None,
                'participant_folder': None
            })
        
        # Extract studyDeploymentId from dataStream column if it exists
        if 'dataStream' in df.columns:
            deployment_ids = df['dataStream'].apply(
                lambda x: x.get('studyDeploymentId') if isinstance(x, dict) else None
            )
        elif 'studyDeploymentId' in df.columns:
            deployment_ids = df['studyDeploymentId']
        else:
            console.print("[yellow]Could not find studyDeploymentId column[/yellow]")
            return df
        
        participant_info = deployment_ids.apply(get_participant_info)
        return pd.concat([df, participant_info], axis=1)

    def scan_schema(self) -> Dict[str, Any]:
        """
        Scans the entire file to infer the schema of the data.
        Returns a dictionary mapping data types to their field structures.
        """
        schemas = defaultdict(set)
        
        # We need to count items for tqdm, but counting requires a pass. 
        # For very large files, we might just use file size or unknown length.
        # Let's try to estimate or just use a simple progress bar.
        
        console.print(f"[bold blue]Scanning schema for {len(self.file_paths)} files...[/bold blue]")
        
        # We can use tqdm wrapping the generator, but we don't know total length easily without reading.
        # We can use file size as a proxy if we read raw bytes, but ijson handles the reading.
        # Let's just use a counter.
        
        count = 0
        with tqdm(desc="Processing items", unit=" items") as pbar:
            for item in self._get_item_generator():
                data_type = item.get('dataStream', {}).get('dataType', {}).get('name', 'unknown')
                namespace = item.get('dataStream', {}).get('dataType', {}).get('namespace', 'unknown')
                full_type = f"{namespace}.{data_type}"
                
                measurement_data = item.get('measurement', {}).get('data', {})
                
                # Collect keys
                for key in measurement_data.keys():
                    schemas[full_type].add(key)
                
                count += 1
                if count % 1000 == 0:
                    pbar.update(1000)
            pbar.update(count % 1000)

        # Convert sets to lists for JSON serialization/display
        self.schema_cache = {k: list(v) for k, v in schemas.items()}
        return self.schema_cache

    def print_schema(self):
        if not self.schema_cache:
            self.scan_schema()
            
        table = Table(title="Inferred Schema")
        table.add_column("Data Type", style="cyan")
        table.add_column("Fields", style="magenta")
        
        for dtype, fields in self.schema_cache.items():
            table.add_row(dtype, ", ".join(sorted(fields)))
            
        console.print(table)

    def get_data_by_type(self, target_type: str) -> Generator[Dict[str, Any], None, None]:
        """
        Yields items of a specific data type.
        """
        target_namespace, target_name = target_type.rsplit('.', 1)
        
        for item in self._get_item_generator():
            dt = item.get('dataStream', {}).get('dataType', {})
            if dt.get('name') == target_name and dt.get('namespace') == target_namespace:
                yield item

    def export_to_json(self, output_path: str, data_type: Optional[str] = None):
        """
        Exports data to a JSON file. Can filter by data type.
        """
        console.print(f"[bold green]Exporting data to {output_path}...[/bold green]")
        
        generator = self.get_data_by_type(data_type) if data_type else self._get_item_generator()
        
        with open(output_path, 'w') as f:
            f.write('[')
            first = True
            for item in tqdm(generator, desc="Exporting"):
                if not first:
                    f.write(',')
                json.dump(item, f)
                first = False
            f.write(']')
        
        console.print("[bold green]Export complete![/bold green]")
    
    def group_by_field(self, field_path: str, output_dir: str | Path):
        """
        Groups data by a specific field and exports each group to a separate JSON file.
        field_path is a dot-separated string, e.g., 'dataStream.dataType.name'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Grouping data by {field_path} into {output_dir}...[/bold blue]")
        
        # We can't keep all files open if there are too many groups.
        # But for things like dataType, there are usually < 20 groups.
        # A safe approach for low memory is to read the file once and append to files, 
        # but opening/closing files for every line is slow.
        # A middle ground is to keep a cache of open file handles, closing LRU if too many.
        
        # For simplicity and speed assuming reasonable number of groups (<100):
        files = {}
        
        try:
            for item in tqdm(self._get_item_generator(), desc="Grouping"):
                # Extract value
                value = item
                for part in field_path.split('.'):
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                
                if value is None:
                    value = "unknown"
                
                value = str(value)
                # Sanitize filename
                safe_value = "".join(c for c in value if c.isalnum() or c in ('-', '_')).strip()
                if not safe_value:
                    safe_value = "unknown"
                    
                if safe_value not in files:
                    f = open(output_dir / f"{safe_value}.json", 'w')
                    f.write('[')
                    files[safe_value] = {'handle': f, 'first': True}
                
                f_info = files[safe_value]
                if not f_info['first']:
                    f_info['handle'].write(',')
                json.dump(item, f_info['handle'])
                f_info['first'] = False
                
        finally:
            for f_info in files.values():
                f_info['handle'].write(']')
                f_info['handle'].close()
                
        console.print(f"[bold green]Grouping complete! Created {len(files)} files.[/bold green]")
    
    def count_items(self) -> int:
        """
        Counts the total number of items in the JSON file.
        """
        console.print(f"[bold blue]Counting items in {len(self.file_paths)} files...[/bold blue]")
        count = 0
        for _ in tqdm(self._get_item_generator(), desc="Counting"):
            count += 1
        return count
    
    def convert_to_parquet(self, output_dir: str | Path, batch_size: int = 10000):
        """
        Converts the JSON data to Parquet files, grouped by data type.
        Requires pyarrow and pandas.
        """
        import importlib.util
        if not importlib.util.find_spec("pyarrow") or not importlib.util.find_spec("pandas"):
             console.print("[bold red]pyarrow and pandas are required for Parquet conversion. Please install them.[/bold red]")
             return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Converting to Parquet in {output_dir}...[/bold blue]")
        
        writers = {}
        buffers = defaultdict(list)
        
        try:
            for item in tqdm(self._get_item_generator(), desc="Converting"):
                # Determine type
                try:
                    dtype = item.get('dataStream', {}).get('dataType', {}).get('name', 'unknown')
                    # Sanitize
                    safe_name = "".join(c for c in dtype if c.isalnum() or c in ('-', '_')).strip()
                    if not safe_name:
                        safe_name = "unknown"
                except (AttributeError, TypeError):
                    safe_name = "unknown"
                
                buffers[safe_name].append(item)
                
                if len(buffers[safe_name]) >= batch_size:
                    self._flush_buffer_to_parquet(safe_name, buffers[safe_name], writers, output_dir)
                    buffers[safe_name].clear()
                    
        finally:
            # Flush remaining
            for name, buf in buffers.items():
                if buf:
                    self._flush_buffer_to_parquet(name, buf, writers, output_dir)
            
            # Close writers
            for writer in writers.values():
                writer.close()
                
        console.print(f"[bold green]Conversion complete! Created {len(writers)} Parquet files.[/bold green]")

    def _flush_buffer_to_parquet(self, name, buffer, writers, output_dir):
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        if not buffer:
            return

        try:
            # PyArrow's from_pylist is robust but might need explicit schema if types vary.
            # We let it infer for now.
            table = pa.Table.from_pylist(buffer)
        except Exception as e:
            console.print(f"[red]Error converting batch for {name}: {e}[/red]")
            return

        if name not in writers:
            file_path = output_dir / f"{name}.parquet"
            # Use the schema from the first batch
            writers[name] = pq.ParquetWriter(file_path, table.schema)
            
        try:
            # If the new batch has a different schema (e.g. missing fields or new fields),
            # write_table might fail or produce a file with multiple schemas (which is bad).
            # Ideally we should unify schemas, but that requires reading all data first.
            # For now, we assume schema consistency or that PyArrow handles minor diffs.
            # If strict schema validation fails, we might need to cast.
            
            # Check if schema matches writer's schema
            if not table.schema.equals(writers[name].schema):
                # Try to cast to the writer's schema
                # This handles cases where a field is missing (null) or type promotion is needed
                try:
                    table = table.cast(writers[name].schema)
                except Exception:
                    # If casting fails, we might have a problem.
                    # For now, log and skip or try to write anyway (which might fail)
                    # console.print(f"[yellow]Schema mismatch for {name}. Attempting cast... {cast_error}[/yellow]")
                    pass

            writers[name].write_table(table)
        except Exception as e:
            console.print(f"[red]Error writing batch for {name}: {e}[/red]")
    
    def get_dataframe(self, data_type: str, parquet_dir: Optional[str | Path] = None):
        """
        Returns a pandas DataFrame for the specified data type.
        If parquet_dir is provided and contains the corresponding parquet file, it loads from there.
        Otherwise, it scans the JSON file (which is slower).
        """
        try:
            import pandas as pd
        except ImportError:
            console.print("[bold red]pandas is required for DataFrame conversion. Please install it.[/bold red]")
            return None

        # Check Parquet first
        if parquet_dir:
            parquet_dir = Path(parquet_dir)
            # data_type might be full namespace "dk.cachet.carp.heartbeat"
            # or just "heartbeat" if we simplified names in conversion.
            # Our conversion uses simplified names.
            
            simple_name = data_type.split('.')[-1]
            parquet_path = parquet_dir / f"{simple_name}.parquet"
            
            if parquet_path.exists():
                console.print(f"[bold blue]Loading {data_type} from {parquet_path}...[/bold blue]")
                return pd.read_parquet(parquet_path)
            
            # Try full name just in case
            safe_full_name = "".join(c for c in data_type if c.isalnum() or c in ('-', '_')).strip()
            parquet_path_full = parquet_dir / f"{safe_full_name}.parquet"
            if parquet_path_full.exists():
                console.print(f"[bold blue]Loading {data_type} from {parquet_path_full}...[/bold blue]")
                return pd.read_parquet(parquet_path_full)

        # Fallback to JSON scan
        console.print(f"[bold yellow]Parquet file not found. Scanning JSON for {data_type}...[/bold yellow]")
        data = list(tqdm(self.get_data_by_type(data_type), desc="Loading to DataFrame"))
        return pd.DataFrame(data)

    def list_all_fields(self, sample_size: int = 100) -> List[str]:
        """
        Scans a sample of items to find all available dot-separated field paths.
        Useful for determining what fields can be used in group_by_field.
        """
        console.print(f"[bold blue]Scanning first {sample_size} items to find field paths...[/bold blue]")
        paths = set()
        
        def _recurse(obj, current_path):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{current_path}.{k}" if current_path else k
                    paths.add(new_path)
                    _recurse(v, new_path)
        
        count = 0
        for item in self._get_item_generator():
            _recurse(item, "")
            count += 1
            if count >= sample_size:
                break
        
        return sorted(list(paths))

    def generate_type_definitions(self, output_file: str = "generated_types.py", sample_size: int = 1000):
        """
        Generates a Python module with dataclasses representing the data schema.
        Detects nested JSON strings and generates types for them as well.
        """
        console.print(f"[bold blue]Inferring schema from first {sample_size} items...[/bold blue]")
        schema = self._infer_full_schema(sample_size)
        
        console.print("[bold blue]Generating code...[/bold blue]")
        code = self._generate_code_from_schema(schema)
        
        with open(output_file, 'w') as f:
            f.write(code)
        console.print(f"[bold green]Generated type definitions in {output_file}[/bold green]")

    def _infer_full_schema(self, sample_size: int) -> Dict[str, Any]:
        root_schema = {"type": "object", "fields": {}}
        
        def merge(schema, value):
            if value is None:
                schema["nullable"] = True
                return

            if isinstance(value, dict):
                if schema.get("type") and schema["type"] != "object":
                    schema["type"] = "Any" # Conflict
                    return
                schema["type"] = "object"
                if "fields" not in schema:
                    schema["fields"] = {}
                
                for k, v in value.items():
                    if k not in schema["fields"]:
                        schema["fields"][k] = {}
                    merge(schema["fields"][k], v)
            
            elif isinstance(value, list):
                if schema.get("type") and schema["type"] != "list":
                    schema["type"] = "Any"
                    return
                schema["type"] = "list"
                if "item_type" not in schema:
                    schema["item_type"] = {}
                
                for item in value:
                    merge(schema["item_type"], item)
            
            else:
                # Primitive
                # Check if string is JSON
                is_json = False
                if isinstance(value, str):
                    try:
                        if (value.strip().startswith('{') and value.strip().endswith('}')) or \
                           (value.strip().startswith('[') and value.strip().endswith(']')):
                            parsed = json.loads(value)
                            if isinstance(parsed, (dict, list)):
                                is_json = True
                                schema["is_json_string"] = True
                                merge(schema, parsed)
                                return
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                if not is_json:
                    py_type = type(value).__name__
                    # Map python types to type hints
                    if py_type == "float":
                        py_type = "float"
                    elif py_type == "int":
                        py_type = "int"
                    elif py_type == "str":
                        py_type = "str"
                    elif py_type == "bool":
                        py_type = "bool"
                    
                    if schema.get("type") == "primitive" and schema.get("python_type") != py_type:
                        # If mixing int and float, upgrade to float
                        if {schema.get("python_type"), py_type} == {"int", "float"}:
                            schema["python_type"] = "float"
                        else:
                            schema["python_type"] = "Any"
                    else:
                        schema["type"] = "primitive"
                        schema["python_type"] = py_type

        count = 0
        for item in self._get_item_generator():
            merge(root_schema, item)
            count += 1
            if count >= sample_size:
                break
        
        return root_schema

    def _generate_code_from_schema(self, schema: Dict[str, Any]) -> str:
        classes = {} # name -> definition
        
        def get_type_name(schema, context_name):
            if schema.get("type") == "object":
                class_name = "".join(x[:1].upper() + x[1:] for x in context_name.split('_'))
                if not class_name:
                    class_name = "Root"
                
                # Handle collision
                base_name = class_name
                counter = 1
                while class_name in classes and classes[class_name] is not None and classes[class_name] != schema.get("fields"):
                    # Note: comparing fields is a weak check for equality, but sufficient for now
                    class_name = f"{base_name}{counter}"
                    counter += 1
                
                if class_name not in classes:
                    classes[class_name] = None # Placeholder
                    fields = []
                    for k, v in schema.get("fields", {}).items():
                        field_type = get_type_name(v, k)
                        fields.append((k, field_type, v.get("nullable", False), v.get("is_json_string", False)))
                    classes[class_name] = fields
                
                return class_name
            
            elif schema.get("type") == "list":
                item_type = get_type_name(schema.get("item_type", {}), context_name + "_item")
                return f"List[{item_type}]"
            
            elif schema.get("type") == "primitive":
                t = schema.get("python_type", "Any")
                return "Any" if t == "Any" else t
            
            return "Any"

        get_type_name(schema, "SleepinessItem")
        
        # Generate Code
        lines = [
            "# Auto-generated type definitions",
            "",
            "from __future__ import annotations",
            "from dataclasses import dataclass",
            "from typing import List, Optional, Any, Dict",
            "import json",
            "",
            "def parse_json_field(value):",
            "    if isinstance(value, str):",
            "        try:",
            "            return json.loads(value)",
            "        except:",
            "            return value",
            "    return value",
            ""
        ]
        
        for name, fields in classes.items():
            if fields is None:
                continue # Should not happen if recursion finished
            
            lines.append("@dataclass")
            lines.append(f"class {name}:")
            if not fields:
                lines.append("    pass")
            
            for fname, ftype, nullable, is_json in fields:
                safe_fname = fname
                if safe_fname in ('from', 'class', 'def', 'return', 'import', 'type', 'global', 'for', 'if', 'else', 'while'):
                    safe_fname = f"{fname}_"
                
                type_hint = ftype
                if nullable:
                    type_hint = f"Optional[{type_hint}]"
                
                lines.append(f"    {safe_fname}: {type_hint} = None")
            
            # Add from_dict method
            lines.append("")
            lines.append("    @classmethod")
            lines.append("    def from_dict(cls, obj: Any) -> Any:")
            lines.append("        if not isinstance(obj, dict): return obj")
            lines.append("        instance = cls()")
            for fname, ftype, nullable, is_json in fields:
                safe_fname = fname
                if safe_fname in ('from', 'class', 'def', 'return', 'import', 'type', 'global', 'for', 'if', 'else', 'while'):
                    safe_fname = f"{fname}_"
                
                base_type = ftype
                is_list = False
                if ftype.startswith("List[") and ftype.endswith("]"):
                    base_type = ftype[5:-1]
                    is_list = True
                
                is_custom_class = base_type in classes
                
                lines.append(f"        val = obj.get('{fname}')")
                if is_json:
                    lines.append("        if isinstance(val, str): val = parse_json_field(val)")
                
                if is_custom_class:
                    if is_list:
                        lines.append("        if val is not None and isinstance(val, list):")
                        lines.append(f"            instance.{safe_fname} = [{base_type}.from_dict(x) for x in val]")
                    else:
                        lines.append("        if val is not None:")
                        lines.append(f"            instance.{safe_fname} = {base_type}.from_dict(val)")
                else:
                    lines.append(f"        instance.{safe_fname} = val")
            
            lines.append("        return instance")
            lines.append("")

        return "\n".join(lines)

