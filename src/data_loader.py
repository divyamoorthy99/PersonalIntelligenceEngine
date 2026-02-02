import json
from datetime import datetime
from typing import List, Dict


def load_data(file_path: str) -> List[Dict]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file containing entries
        
    Returns:
        List of entry dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        entry['date_obj'] = datetime.strptime(entry['date'], '%Y-%m-%d')
    
    data.sort(key=lambda x: x['date_obj'])
    
    return data
