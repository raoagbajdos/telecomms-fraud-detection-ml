"""
Date utilities for telecoms fraud detection.
"""

from datetime import datetime, timedelta, date
from typing import Optional, Union, List
import re

class DateUtils:
    """Utility class for date and time operations."""
    
    @staticmethod
    def get_current_timestamp() -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        return datetime.now().isoformat()
    
    @staticmethod
    def get_date_string(format_str: str = '%Y%m%d') -> str:
        """
        Get current date as formatted string.
        
        Args:
            format_str: Date format string
            
        Returns:
            Formatted date string
        """
        return datetime.now().strftime(format_str)
    
    @staticmethod
    def get_datetime_string(format_str: str = '%Y%m%d_%H%M%S') -> str:
        """
        Get current datetime as formatted string.
        
        Args:
            format_str: Datetime format string
            
        Returns:
            Formatted datetime string
        """
        return datetime.now().strftime(format_str)
    
    @staticmethod
    def parse_date(date_string: str, format_str: Optional[str] = None) -> Optional[datetime]:
        """
        Parse date string to datetime object.
        
        Args:
            date_string: Date string to parse
            format_str: Expected format (if None, tries common formats)
            
        Returns:
            Datetime object or None if parsing fails
        """
        if format_str:
            try:
                return datetime.strptime(date_string, format_str)
            except ValueError:
                return None
        
        # Try common date formats
        common_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f'
        ]
        
        for fmt in common_formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def calculate_age_from_date(birth_date: Union[str, datetime], 
                               reference_date: Optional[Union[str, datetime]] = None) -> Optional[int]:
        """
        Calculate age from birth date.
        
        Args:
            birth_date: Birth date string or datetime object
            reference_date: Reference date (default: current date)
            
        Returns:
            Age in years or None if calculation fails
        """
        # Parse birth date if string
        if isinstance(birth_date, str):
            birth_date = DateUtils.parse_date(birth_date)
            if birth_date is None:
                return None
        
        # Parse reference date if string
        if reference_date is None:
            reference_date = datetime.now()
        elif isinstance(reference_date, str):
            reference_date = DateUtils.parse_date(reference_date)
            if reference_date is None:
                return None
        
        # Calculate age
        age = reference_date.year - birth_date.year
        
        # Adjust for birthday not yet occurred this year
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return age
    
    @staticmethod
    def days_between_dates(date1: Union[str, datetime], 
                          date2: Union[str, datetime]) -> Optional[int]:
        """
        Calculate days between two dates.
        
        Args:
            date1: First date
            date2: Second date
            
        Returns:
            Number of days between dates or None if parsing fails
        """
        # Parse dates if strings
        if isinstance(date1, str):
            date1 = DateUtils.parse_date(date1)
            if date1 is None:
                return None
        
        if isinstance(date2, str):
            date2 = DateUtils.parse_date(date2)
            if date2 is None:
                return None
        
        return abs((date2 - date1).days)
    
    @staticmethod
    def add_business_days(start_date: Union[str, datetime], 
                         business_days: int) -> Optional[datetime]:
        """
        Add business days to a date (excludes weekends).
        
        Args:
            start_date: Starting date
            business_days: Number of business days to add
            
        Returns:
            New date or None if parsing fails
        """
        # Parse start date if string
        if isinstance(start_date, str):
            start_date = DateUtils.parse_date(start_date)
            if start_date is None:
                return None
        
        current_date = start_date
        days_added = 0
        
        while days_added < business_days:
            current_date += timedelta(days=1)
            
            # Skip weekends (Saturday = 5, Sunday = 6)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date
    
    @staticmethod
    def is_weekend(date_obj: Union[str, datetime]) -> Optional[bool]:
        """
        Check if a date falls on weekend.
        
        Args:
            date_obj: Date to check
            
        Returns:
            True if weekend, False if weekday, None if parsing fails
        """
        # Parse date if string
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)
            if date_obj is None:
                return None
        
        # Saturday = 5, Sunday = 6
        return date_obj.weekday() >= 5
    
    @staticmethod
    def get_quarter(date_obj: Union[str, datetime]) -> Optional[int]:
        """
        Get quarter of the year for a date.
        
        Args:
            date_obj: Date to get quarter for
            
        Returns:
            Quarter (1-4) or None if parsing fails
        """
        # Parse date if string
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)
            if date_obj is None:
                return None
        
        return (date_obj.month - 1) // 3 + 1
    
    @staticmethod
    def get_week_of_year(date_obj: Union[str, datetime]) -> Optional[int]:
        """
        Get week number of the year for a date.
        
        Args:
            date_obj: Date to get week for
            
        Returns:
            Week number (1-53) or None if parsing fails
        """
        # Parse date if string
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)
            if date_obj is None:
                return None
        
        return date_obj.isocalendar()[1]
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
        else:
            days = seconds / 86400
            return f"{days:.2f} days"
    
    @staticmethod
    def create_model_filename(base_name: str, 
                             model_type: str, 
                             include_date: bool = True,
                             extension: str = '.pkl') -> str:
        """
        Create a model filename with optional date stamp.
        
        Args:
            base_name: Base name for the model
            model_type: Type of the model
            include_date: Whether to include date in filename
            extension: File extension
            
        Returns:
            Generated filename
        """
        if include_date:
            date_str = DateUtils.get_datetime_string()
            return f"{base_name}_{model_type}_{date_str}{extension}"
        else:
            return f"{base_name}_{model_type}{extension}"
    
    @staticmethod
    def parse_model_filename(filename: str) -> dict:
        """
        Parse model filename to extract components.
        
        Args:
            filename: Model filename to parse
            
        Returns:
            Dictionary with parsed components
        """
        # Remove extension
        name_without_ext = filename.rsplit('.', 1)[0]
        extension = filename.rsplit('.', 1)[1] if '.' in filename else ''
        
        # Try to extract date pattern (YYYYMMDD_HHMMSS)
        date_pattern = r'(\d{8}_\d{6})$'
        match = re.search(date_pattern, name_without_ext)
        
        if match:
            date_str = match.group(1)
            name_without_date = name_without_ext[:-len(date_str)-1]  # -1 for underscore
            
            # Parse the date
            try:
                model_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            except ValueError:
                model_date = None
        else:
            date_str = None
            model_date = None
            name_without_date = name_without_ext
        
        # Split remaining name to get base name and model type
        parts = name_without_date.split('_')
        if len(parts) >= 2:
            model_type = parts[-1]
            base_name = '_'.join(parts[:-1])
        else:
            model_type = 'unknown'
            base_name = name_without_date
        
        return {
            'filename': filename,
            'base_name': base_name,
            'model_type': model_type,
            'date_string': date_str,
            'model_date': model_date,
            'extension': extension
        }
    
    @staticmethod
    def validate_date_range(start_date: Union[str, datetime], 
                           end_date: Union[str, datetime]) -> dict:
        """
        Validate a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with validation results
        """
        # Parse dates if strings
        if isinstance(start_date, str):
            start_date = DateUtils.parse_date(start_date)
        
        if isinstance(end_date, str):
            end_date = DateUtils.parse_date(end_date)
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if dates were parsed successfully
        if start_date is None:
            validation_result['errors'].append("Invalid start date")
            validation_result['valid'] = False
        
        if end_date is None:
            validation_result['errors'].append("Invalid end date")
            validation_result['valid'] = False
        
        # Check date order
        if start_date and end_date:
            if start_date > end_date:
                validation_result['errors'].append("Start date must be before end date")
                validation_result['valid'] = False
            
            # Check if range is too large
            days_diff = (end_date - start_date).days
            if days_diff > 365 * 2:  # More than 2 years
                validation_result['warnings'].append("Date range spans more than 2 years")
            
            validation_result['days_difference'] = days_diff
            validation_result['start_date'] = start_date
            validation_result['end_date'] = end_date
        
        return validation_result
