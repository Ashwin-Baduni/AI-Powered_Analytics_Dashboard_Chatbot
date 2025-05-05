import re
import json
import os
from pathlib import Path
from typing import Union, Dict, Callable, Tuple, Optional, Any
import operator

class Calculations:
    """
    Handles both basic math expressions and complex vehicle data calculations.
    """
    # Basic math operations
    OPERATIONS: Dict[str, Callable[[float, float], float]] = {
        'plus': operator.add,
        'minus': operator.sub,
        'times': operator.mul,
        'dividedby': operator.truediv,
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    # Regex patterns for math parsing
    DIRECT_ARITHMETIC_PATTERN = re.compile(r'^\-?\d+[\+\-\*\/]\-?\d+$')
    PHRASE_PATTERN = re.compile(r'^whatis(\d+)(plus|minus|times|dividedby)(\d+)$')
    
    # Regex patterns for vehicle data calculations
    YEAR_RANGE_PATTERN = re.compile(r'(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and|through|until)\s+((?:19|20)\d{2})')
    SINGLE_YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    
    # Keywords for vehicle calculation queries
    CALC_KEYWORDS = {
        'total', 'sum', 'average', 'mean', 'median', 'calculate',
        'how many', 'count', 'difference between'
    }
    
    DATA_KEYWORDS = {'registration', 'vehicle', 'car', 'data'}

    @staticmethod
    def evaluate_expression(expression: str) -> Union[float, str]:
        """Evaluate basic mathematical expressions safely."""
        # Normalize the input
        expr = expression.lower().replace(' ', '').replace('?', '').strip()
        
        # Try direct arithmetic evaluation first
        if Calculations.DIRECT_ARITHMETIC_PATTERN.match(expr):
            try:
                # Parse manually instead of using eval
                for op_symbol in ['+', '-', '*', '/']:
                    if op_symbol in expr:
                        parts = expr.split(op_symbol)
                        if len(parts) == 2:
                            try:
                                num1 = float(parts[0])
                                num2 = float(parts[1])
                                return Calculations.OPERATIONS[op_symbol](num1, num2)
                            except (ValueError, ZeroDivisionError):
                                return "Error: Invalid numbers or division by zero"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Handle natural language phrases
        match = Calculations.PHRASE_PATTERN.match(expr)
        if match:
            try:
                num1 = float(match.group(1))
                op_name = match.group(2)
                num2 = float(match.group(3))
                
                if op_name not in Calculations.OPERATIONS:
                    return "Error: Unknown operation"
                    
                op_func = Calculations.OPERATIONS[op_name]
                try:
                    return op_func(num1, num2)
                except ZeroDivisionError:
                    return "Error: Division by zero"
            except ValueError:
                return "Error: Invalid numbers"
            except Exception as e:
                return f"Error: {str(e)}"
                
        return "Error: Could not parse expression"

    @staticmethod
    def get_data(data_path=None):
        """Load vehicle registration data from JSON file."""
        if data_path is None:
            base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Define specific paths to check
            potential_paths = [
                base_path / "newdata.json",
                base_path / "data" / "newdata.json"
            ]
            
            for path in potential_paths:
                if path.exists():
                    try:
                        with open(path) as f:
                            return json.load(f)
                    except Exception:
                        continue
                        
            return None
        else:
            with open(data_path) as f:
                return json.load(f)

    @classmethod
    def is_calculation_query(cls, query):
        """Determine if query asks for calculations on vehicle registration data."""
        query_lower = query.lower()
        has_year_range = cls.YEAR_RANGE_PATTERN.search(query_lower) is not None
        has_single_year = cls.SINGLE_YEAR_PATTERN.search(query_lower) is not None
        has_data_reference = any(word in query_lower for word in cls.DATA_KEYWORDS)
        has_calc_keyword = any(keyword in query_lower for keyword in cls.CALC_KEYWORDS)
        
        return ((has_year_range or (has_single_year and has_calc_keyword)) and has_data_reference)

    @classmethod
    def extract_years(cls, query):
        """Extract year ranges from query."""
        query_lower = query.lower()
        
        # Check for year range pattern (from 2000 to 2020)
        range_match = cls.YEAR_RANGE_PATTERN.search(query_lower)
        if range_match:
            start_year = int(range_match.group(1))
            end_year = int(range_match.group(2))
            return min(start_year, end_year), max(start_year, end_year)
        
        # Extract all year mentions
        years = [int(y) for y in cls.SINGLE_YEAR_PATTERN.findall(query_lower)]
        if len(years) >= 2:
            return min(years), max(years)
        elif len(years) == 1:
            return years[0], years[0]
            
        return None

    @classmethod
    def calculate_total(cls, start_year, end_year, data=None):
        """Calculate total vehicle registrations between start_year and end_year."""
        if data is None:
            data = cls.get_data()
            
        if not data:
            return None, "Could not find vehicle registration data."
            
        # Combine and filter data in one step
        years_data = {}
        total_vehicles = 0
        
        for entry in data.get('Original Data', []) + data.get('Predicted Data', []):
            year = int(entry['year'])
            if start_year <= year <= end_year:
                total_vehicles += entry['vehicles']
                years_data[year] = entry['vehicles']
                
        if not years_data:
            return None, f"No data available for the years {start_year} to {end_year}."
            
        available_years = sorted(years_data.keys())
        
        # Build response
        detail_text = [f"Total vehicle registrations from {start_year} to {end_year}: {total_vehicles:,}\n"]
        detail_text.append("Yearly breakdown:")
        
        for year in range(start_year, end_year + 1):
            if year in years_data:
                detail_text.append(f"- {year}: {years_data[year]:,} vehicles")
            else:
                detail_text.append(f"- {year}: No data available")
                
        if len(available_years) >= 2:
            first_year = min(available_years)
            last_year = max(available_years)
            first_value = years_data[first_year]
            last_value = years_data[last_year]
            change = last_value - first_value
            percent_change = (change / first_value) * 100
            
            detail_text.append(f"\nFrom {first_year} to {last_year}, there was a " +
                              (f"growth of {change:,} vehicles (+{percent_change:.1f}%)."
                               if change > 0 else
                               f"decrease of {abs(change):,} vehicles ({percent_change:.1f}%)."))
                               
        return total_vehicles, "\n".join(detail_text)

    @classmethod
    def process_query(cls, query):
        """Process a calculation query and return the appropriate response."""
        if not cls.is_calculation_query(query):
            return None
            
        years = cls.extract_years(query)
        if not years:
            return None
            
        start_year, end_year = years
        return cls.calculate_total(start_year, end_year)[1]

    @classmethod
    def calculate_ratio(cls, query):
        """Calculate ratio between two vehicle types or fuel types."""
        query_lower = query.lower()
        
        # Check if this is a ratio query
        if not ('ratio' in query_lower or 'proportion' in query_lower):
            return None
            
        # Extract year
        years = cls.extract_years(query)
        if not years:
            return None
            
        year = years[0] if isinstance(years, tuple) else years[0]
        
        # Determine if this is a category or fuel ratio query
        if 'fuel' in query_lower or any(fuel in query_lower for fuel in ['petrol', 'diesel', 'electric', 'ev', 'cng', 'lpg']):
            return cls._calculate_fuel_ratio(query_lower, year)
        else:
            return cls._calculate_category_ratio(query_lower, year)

    @classmethod
    def _calculate_fuel_ratio(cls, query_lower, year):
        """Calculate ratio between two fuel types."""
        from data_handler import AdaptiveDataParser
        
        # Extract fuel types
        fuel_types = AdaptiveDataParser._extract_fuels_from_query(query_lower)
        if len(fuel_types) < 2:
            return None
            
        # Get data
        fuel_data = AdaptiveDataParser.get_data('fuels')
        if not fuel_data or str(year) not in fuel_data:
            return None
            
        year_data = fuel_data[str(year)]
        
        # Calculate ratio
        fuel1_count = year_data.get(fuel_types[0], 0)
        fuel2_count = year_data.get(fuel_types[1], 0)
        
        if fuel2_count == 0:
            ratio_text = f"Cannot calculate ratio as {fuel_types[1]} has zero vehicles"
        else:
            ratio = fuel1_count / fuel2_count
            ratio_text = f"The ratio of {fuel_types[0]} to {fuel_types[1]} in {year} is {ratio:.2f}:1"
            
        # Add percentages
        total = fuel1_count + fuel2_count
        if total > 0:
            ratio_text += f"\n{fuel_types[0]}: {fuel1_count:,} vehicles ({(fuel1_count/total)*100:.1f}%)"
            ratio_text += f"\n{fuel_types[1]}: {fuel2_count:,} vehicles ({(fuel2_count/total)*100:.1f}%)"
            
        return ratio_text

    @classmethod
    def _calculate_category_ratio(cls, query_lower, year):
        """Calculate ratio between two vehicle categories."""
        from data_handler import AdaptiveDataParser
        
        # Extract categories
        categories = AdaptiveDataParser._extract_categories_from_query(query_lower)
        if len(categories) < 2:
            return None
            
        # Get data
        cat_data = AdaptiveDataParser.get_data('categories')
        if not cat_data or str(year) not in cat_data:
            return None
            
        year_data = cat_data[str(year)]
        
        # Calculate ratio
        cat1_count = year_data.get(categories[0], 0)
        cat2_count = year_data.get(categories[1], 0)
        
        if cat2_count == 0:
            ratio_text = f"Cannot calculate ratio as {categories[1]} has zero vehicles"
        else:
            ratio = cat1_count / cat2_count
            ratio_text = f"The ratio of {categories[0]} to {categories[1]} in {year} is {ratio:.2f}:1"
            
        # Add percentages
        total = cat1_count + cat2_count
        if total > 0:
            ratio_text += f"\n{categories[0]}: {cat1_count:,} vehicles ({(cat1_count/total)*100:.1f}%)"
            ratio_text += f"\n{categories[1]}: {cat2_count:,} vehicles ({(cat2_count/total)*100:.1f}%)"
            
        return ratio_text
