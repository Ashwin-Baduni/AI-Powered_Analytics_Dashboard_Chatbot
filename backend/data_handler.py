import json
import re
from pathlib import Path
from functools import lru_cache
import datetime
from typing import Dict, Any, List, Optional
import difflib

class AdaptiveDataParser:
    DATA_PATH = Path(__file__).parent.parent
    
    DATA_FILES = {
        'registrations': 'newdata.json',
        'categories': 'getvehiclecountsbycategory.json',
        'fuels': 'getvehiclecountsbyfuel.json',
        'categories_and_fuels': 'getvehiclecountsbycategoryandfuel.json'
    }
    
    # Enhanced query signatures with more comprehensive keywords
    QUERY_SIGNATURES = {
        'registrations': {
            'keywords': ['total', 'overall', 'registration', 'vehicle', 'count', 
                         'how many', 'number of vehicles', 'trend', 'increase', 'decrease',
                         'growth', 'decline', 'statistics', 'data', 'projection'],
            'excludes': ['category', 'fuel', 'petrol', 'diesel', 'type', 'truck', 'car',
                         'electric', 'ev', 'cng', 'lpg', 'two wheeler', 'bike']
        },
        'categories': {
            'keywords': ['category', 'categories', 'vehicle type', 'classification', 'class',
                        '2wn', 'two wheeler', 'lmv', 'car', 'lgv', 'truck', 'bus', 'hpv', 
                        'auto', '3wn', 'three wheeler', 'motorcycle', 'scooter', 'jeep'],
            'excludes': ['fuel', 'petrol', 'diesel', 'electric', 'ev', 'cng', 'lpg', 'hybrid']
        },
        'fuels': {
            'keywords': ['fuel', 'petrol', 'diesel', 'cng', 'lpg', 'electric', 'ev', 
                        'hybrid', 'solar', 'ethanol', 'gas', 'power source'],
            'excludes': ['category', 'type', 'truck', 'car', 'two wheeler', 'bike', 'motorcycle']
        },
        'categories_and_fuels': {
            'keywords': ['both', 'breakdown', 'distribution', 'by fuel and category',
                        'by category and fuel', 'petrol cars', 'diesel trucks', 
                        'electric two wheelers', 'fuel wise categories', 'petrol bike',
                        'electric car', 'diesel bus', 'ev car', 'ev bike'],
            'excludes': []
        }
    }
    
    # Common misspellings dictionary
    COMMON_MISSPELLINGS = {
        'petrl': 'petrol',
        'disel': 'diesel',
        'deisel': 'diesel',
        'vehicl': 'vehicle',
        'regestration': 'registration',
        'registeration': 'registration',
        'motercycle': 'motorcycle',
        'motorbike': 'motorcycle',
        'electic': 'electric',
        'eletric': 'electric',
        'hybrd': 'hybrid',
        'truk': 'truck',
        'trucks': 'truck',
        'vehicals': 'vehicles',
        'vehecle': 'vehicle',
        'twowheeler': 'two wheeler',
        'fourwheeler': 'four wheeler',
        'passanger': 'passenger',
        'catagory': 'category',
        'catagories': 'categories'
    }
    
    # Consolidated mappings
    CATEGORY_MAPPING = {
        '2wn': '2WN', 'two wheeler': '2WN', 'bike': '2WN', 'motorcycle': '2WN', 'scooter': '2WN',
        'lmv': 'LMV', 'car': 'LMV', 'jeep': 'LMV', 'four wheeler': 'LMV',
        'lpv': 'LPV', 'passenger': 'LPV',
        'hgv': 'HGV', 'heavy goods': 'HGV', 'truck': 'HGV',
        'lgv': 'LGV', 'light goods': 'LGV',
        'mgv': 'MGV', 'medium goods': 'MGV',
        '3wn': '3WN', 'three wheeler': '3WN', 'auto': '3WN', 'rickshaw': '3WN',
        'hpv': 'HPV', 'heavy passenger': 'HPV', 'bus': 'HPV'
    }
    
    FUEL_MAPPING = {
        'petrol': 'PETROL',
        'diesel': 'DIESEL',
        'cng': 'CNG ONLY',
        'lpg': 'LPG ONLY',
        'electric': 'ELECTRIC BOV',
        'ev': 'ELECTRIC BOV',
        'pure ev': 'PURE EV',
        'hybrid': ['DIESEL/HYBRID', 'PETROL/HYBRID', 'STRONG HYBRID EV'],
        'solar': 'SOLAR',
        'ethanol': 'PETROL/ETHANOL'
    }
    
    @classmethod
    def preprocess_query(cls, query: str) -> str:
        """
        Preprocess the query to correct common misspellings.
        """
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            # Check for exact matches in the misspellings dictionary
            if word in cls.COMMON_MISSPELLINGS:
                corrected_words.append(cls.COMMON_MISSPELLINGS[word])
                continue
            
            # Check for fuzzy matches in the misspellings dictionary
            best_match = None
            best_score = 0
            for misspelling, correction in cls.COMMON_MISSPELLINGS.items():
                score = difflib.SequenceMatcher(None, word, misspelling).ratio()
                if score > 0.85 and score > best_score:
                    best_match = correction
                    best_score = score
            
            if best_match:
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    @classmethod
    def _fuzzy_match_keyword(cls, query: str, keyword: str, threshold=0.8) -> bool:
        """
        Check if a keyword or a close variation exists in the query.
        """
        words = query.lower().split()
        for word in words:
            similarity = difflib.SequenceMatcher(None, word, keyword).ratio()
            if similarity >= threshold:
                return True
        
        # Also check for the keyword as a phrase in the query
        similarity = difflib.SequenceMatcher(None, query, keyword).ratio()
        if similarity >= threshold:
            return True
        
        return False
    
    @classmethod
    def recognize_query_pattern(cls, query: str) -> Optional[str]:
        """Recognize specific query patterns to determine the appropriate data file."""
        query_lower = query.lower()
        
        # Pattern for "How many [fuel] [category] in [year]"
        fuel_category_year_pattern = re.search(r'how many\s+(\w+)\s+(\w+).*?\b(19|20)\d{2}\b', query_lower)
        if fuel_category_year_pattern:
            fuel_term = fuel_category_year_pattern.group(1)
            category_term = fuel_category_year_pattern.group(2)
            
            is_fuel = any(cls._fuzzy_match_keyword(fuel_term, key, 0.8) or 
                          (fuel_term == key or fuel_term in key)
                          for key in cls.FUEL_MAPPING.keys())
            is_category = any(cls._fuzzy_match_keyword(category_term, key, 0.8) or 
                             (category_term == key or category_term in key)
                             for key in cls.CATEGORY_MAPPING.keys())
            
            if is_fuel and is_category:
                return 'categories_and_fuels'
        
        # Pattern for "How many [category] in [year]"
        category_year_pattern = re.search(r'how many\s+(\w+).*?\b(19|20)\d{2}\b', query_lower)
        if category_year_pattern:
            category_term = category_year_pattern.group(1)
            is_category = any(cls._fuzzy_match_keyword(category_term, key, 0.8) or 
                             (category_term == key or category_term in key)
                             for key in cls.CATEGORY_MAPPING.keys())
            if is_category:
                return 'categories'
        
        # Pattern for "How many [fuel] vehicles in [year]"
        fuel_year_pattern = re.search(r'how many\s+(\w+)(?:\s+vehicles|\s+cars)?.*?\b(19|20)\d{2}\b', query_lower)
        if fuel_year_pattern:
            fuel_term = fuel_year_pattern.group(1)
            is_fuel = any(cls._fuzzy_match_keyword(fuel_term, key, 0.8) or 
                         (fuel_term == key or fuel_term in key)
                         for key in cls.FUEL_MAPPING.keys())
            if is_fuel:
                return 'fuels'
        
        return None
    
    @classmethod
    def determine_data_file_by_decision_tree(cls, query: str) -> Optional[str]:
        """Use a decision tree approach to determine the appropriate data file."""
        query_lower = query.lower()
        
        # Extract components
        years = cls._extract_years_from_query(query_lower)
        categories = cls._extract_categories_from_query(query_lower)
        fuels = cls._extract_fuels_from_query(query_lower)
        
        # Decision tree
        if categories and fuels:
            return 'categories_and_fuels'
        elif categories:
            return 'categories'
        elif fuels:
            return 'fuels'
        elif years or any(term in query_lower for term in ['total', 'overall', 'trend']):
            return 'registrations'
        
        # Default
        return None
    
    @classmethod
    def determine_data_files_needed(cls, query: str) -> List[str]:
        """
        Analyze the query to determine which data files are most relevant.
        Returns a list of data file keys in order of relevance.
        """
        # Preprocess the query to correct common misspellings
        processed_query = cls.preprocess_query(query)
        query_lower = processed_query.lower()
        
        # Try pattern recognition first
        pattern_result = cls.recognize_query_pattern(query_lower)
        if pattern_result:
            return [pattern_result]
        
        # Try decision tree approach
        decision_result = cls.determine_data_file_by_decision_tree(query_lower)
        if decision_result:
            return [decision_result]
        
        # Fall back to scoring method
        file_scores = {}
        
        # Score each file based on keyword matches and exclusions
        for file_key, signature in cls.QUERY_SIGNATURES.items():
            score = 0
            for keyword in signature['keywords']:
                if keyword in query_lower or cls._fuzzy_match_keyword(query_lower, keyword):
                    score += 2
            for exclusion in signature['excludes']:
                if exclusion in query_lower or cls._fuzzy_match_keyword(query_lower, exclusion, 0.9):
                    score -= 3
            file_scores[file_key] = score
        
        # Handle special cases
        if any(term in query_lower for term in ['compare', 'comparison', 'difference between', 'trend']):
            file_scores['registrations'] += 2
        if any(term in query_lower for term in ['predict', 'prediction', 'forecast', 'future']):
            file_scores['registrations'] += 3
        
        # Sort by score (descending) and filter to only positive scores
        sorted_files = [file_key for file_key, score in
                       sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
                       if score > 0]
        
        # Default to registrations if no file is identified as relevant
        return sorted_files if sorted_files else ['registrations']
    
    @classmethod
    @lru_cache(maxsize=16)
    def _load_data_file(cls, file_path):
        """Load JSON data from file with caching."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    @classmethod
    def get_data_file_path(cls, file_key):
        """Get the full path for a data file by key."""
        if file_key not in cls.DATA_FILES:
            return None
        
        # Define specific paths to check
        potential_paths = [
            cls.DATA_PATH / cls.DATA_FILES[file_key],
            cls.DATA_PATH / "data" / cls.DATA_FILES[file_key]
        ]
        
        for path in potential_paths:
            if path.exists():
                return path
        
        return None
    
    @classmethod
    def get_data(cls, file_key):
        """Load a specific data file by key."""
        path = cls.get_data_file_path(file_key)
        if path:
            return cls._load_data_file(path)
        return None
    
    @classmethod
    def get_relevant_context(cls, query: str) -> dict:
        """Get relevant context data for the query."""
        context_texts = []
        
        # Preprocess the query to correct common misspellings
        processed_query = cls.preprocess_query(query)
        query_lower = processed_query.lower()
        
        # Determine which files are most relevant to this query
        relevant_files = cls.determine_data_files_needed(query_lower)
        
        # Process data based on relevance
        cls._process_data_by_relevance(query_lower, relevant_files, context_texts)
        
        # Add default context if nothing was added
        if not context_texts:
            cls._add_default_context(context_texts)
        
        # Add a note if the query was corrected
        if processed_query.lower() != query.lower():
            context_texts.append(f"Note: I interpreted '{query}' as '{processed_query}'")
        
        return {
            'texts': context_texts,
            'images': [],
            'datasets': []
        }
    
    @classmethod
    def _process_data_by_relevance(cls, query_lower, relevant_files, context_texts):
        """Process data files based on relevance to the query."""
        # Generic processor for different data types
        for file_key in relevant_files:
            data = cls.get_data(file_key)
            if not data:
                continue
            
            # Process based on file type
            if file_key == 'registrations':
                cls._process_registration_data(query_lower, context_texts)
            elif file_key == 'categories':
                cls._process_category_data(query_lower, context_texts)
            elif file_key == 'fuels':
                cls._process_fuel_data(query_lower, context_texts)
            elif file_key == 'categories_and_fuels':
                cls._process_category_and_fuel_data(query_lower, context_texts)
    
    @classmethod
    def _add_default_context(cls, context_texts):
        """Add default context information when no specific data is found."""
        reg_data = cls.get_data('registrations')
        if reg_data:
            combined_data = reg_data.get('Original Data', []) + reg_data.get('Predicted Data', [])
            years = sorted([int(entry['year']) for entry in combined_data])
            if years:
                min_year, max_year = min(years), max(years)
                context_texts.append(f"Vehicle registration data is available from {min_year} to {max_year}.")
                current_year = datetime.datetime.now().year
                future_years = [y for y in years if y > current_year]
                if future_years:
                    context_texts.append(f"This includes predictions up to {max(future_years)}.")
    
    @classmethod
    def _has_keyword_type(cls, query_lower, keyword_type):
        """Check if query contains keywords of a specific type."""
        return any(k in query_lower for k in cls.QUERY_SIGNATURES[keyword_type]['keywords'])
    
    @classmethod
    def _process_registration_data(cls, query_lower, context_texts):
        """Process registration data for the query."""
        data = cls.get_data('registrations')
        if not data:
            context_texts.append("No vehicle registration data is available.")
            return
        
        combined_data = data.get('Original Data', []) + data.get('Predicted Data', [])
        year_to_entry = {int(entry['year']): entry for entry in combined_data}
        
        if not year_to_entry:
            context_texts.append("No vehicle registration data entries found.")
            return
        
        years = sorted(year_to_entry.keys())
        min_year, max_year = min(years), max(years)
        
        # Process query patterns
        cls._process_total_query(query_lower, year_to_entry, context_texts)
        cls._process_year_mentions(query_lower, year_to_entry, context_texts)
        cls._process_trend_query(query_lower, year_to_entry, min_year, max_year, context_texts)
        cls._process_prediction_query(query_lower, year_to_entry, context_texts)
        cls._process_time_periods(query_lower, year_to_entry, context_texts)
    
    @classmethod
    def _process_category_data(cls, query_lower, context_texts):
        """Process category data for the query."""
        data = cls.get_data('categories')
        if not data:
            return
        
        # Extract years and categories from query
        years = cls._extract_years_from_query(query_lower)
        categories = cls._extract_categories_from_query(query_lower)
        
        # Process year-specific data
        if years:
            for year in years:
                year_str = str(year)
                if year_str in data:
                    if categories:
                        # Show specific categories for the year
                        context_texts.append(f"Vehicle categories for {year}:")
                        for category in categories:
                            if category in data[year_str]:
                                context_texts.append(f"- {category}: {data[year_str][category]:,} vehicles")
                    else:
                        # Show top categories for the year
                        sorted_categories = sorted(data[year_str].items(), key=lambda x: x[1], reverse=True)
                        context_texts.append(f"Top vehicle categories in {year}:")
                        for category, count in sorted_categories[:5]:
                            context_texts.append(f"- {category}: {count:,} vehicles")
        else:
            # Use latest year if no specific year is mentioned
            available_years = sorted([int(y) for y in data.keys()])
            if available_years:
                latest_year = str(max(available_years))
                sorted_categories = sorted(data[latest_year].items(), key=lambda x: x[1], reverse=True)
                context_texts.append(f"Top vehicle categories in {latest_year}:")
                for category, count in sorted_categories[:5]:
                    context_texts.append(f"- {category}: {count:,} vehicles")
    
    @classmethod
    def _process_fuel_data(cls, query_lower, context_texts):
        """Process fuel data for the query."""
        data = cls.get_data('fuels')
        if not data:
            return
        
        # Extract years and fuels from query
        years = cls._extract_years_from_query(query_lower)
        fuels = cls._extract_fuels_from_query(query_lower)
        
        # Process year-specific data
        if years:
            for year in years:
                year_str = str(year)
                if year_str in data:
                    if fuels:
                        # Show specific fuels for the year
                        context_texts.append(f"Fuel types for {year}:")
                        for fuel in fuels:
                            if fuel in data[year_str]:
                                context_texts.append(f"- {fuel}: {data[year_str][fuel]:,} vehicles")
                    else:
                        # Show top fuels for the year
                        sorted_fuels = sorted(data[year_str].items(), key=lambda x: x[1], reverse=True)
                        context_texts.append(f"Top fuel types in {year}:")
                        for fuel, count in sorted_fuels[:5]:
                            if count > 0: # Only show non-zero counts
                                context_texts.append(f"- {fuel}: {count:,} vehicles")
        else:
            # Use latest year if no specific year is mentioned
            available_years = sorted([int(y) for y in data.keys()])
            if available_years:
                latest_year = str(max(available_years))
                sorted_fuels = sorted(data[latest_year].items(), key=lambda x: x[1], reverse=True)
                context_texts.append(f"Top fuel types in {latest_year}:")
                for fuel, count in sorted_fuels[:5]:
                    if count > 0: # Only show non-zero counts
                        context_texts.append(f"- {fuel}: {count:,} vehicles")
    
    @classmethod
    def _process_category_and_fuel_data(cls, query_lower, context_texts):
        """Process combined category and fuel data for the query."""
        data = cls.get_data('categories_and_fuels')
        if not data:
            return
        
        # Extract years, categories and fuels from query
        years = cls._extract_years_from_query(query_lower)
        categories = cls._extract_categories_from_query(query_lower)
        fuels = cls._extract_fuels_from_query(query_lower)
        
        # Process based on what's specified in the query
        cls._process_category_fuel_combinations(data, years, categories, fuels, context_texts)
    
    @classmethod
    def _process_category_fuel_combinations(cls, data, years, categories, fuels, context_texts):
        """Process different combinations of categories and fuels."""
        # If we have specific categories and fuels
        if categories and fuels:
            cls._process_specific_categories_and_fuels(data, years, categories, fuels, context_texts)
        # If we have categories but no specific fuels
        elif categories and not fuels:
            cls._process_categories_with_top_fuels(data, years, categories, context_texts)
        # If we have fuels but no specific categories
        elif fuels and not categories:
            cls._process_fuels_with_top_categories(data, years, fuels, context_texts)
    
    @classmethod
    def _process_specific_categories_and_fuels(cls, data, years, categories, fuels, context_texts):
        """Process specific categories and fuels."""
        if years:
            # Show specific data for the years
            for year in years:
                year_str = str(year)
                if year_str in data:
                    for category in categories:
                        if category in data[year_str]:
                            context_texts.append(f"{category} vehicles by fuel type in {year}:")
                            category_data = data[year_str][category]['fuel_counts']
                            for fuel in fuels:
                                if fuel in category_data:
                                    context_texts.append(f"- {fuel}: {category_data[fuel]:,} vehicles")
        else:
            # Latest year
            available_years = sorted([int(y) for y in data.keys()])
            if available_years:
                latest_year = str(max(available_years))
                for category in categories:
                    if category in data[latest_year]:
                        context_texts.append(f"{category} vehicles by fuel type in {latest_year}:")
                        category_data = data[latest_year][category]['fuel_counts']
                        for fuel in fuels:
                            if fuel in category_data:
                                context_texts.append(f"- {fuel}: {category_data[fuel]:,} vehicles")
    
    @classmethod
    def _process_categories_with_top_fuels(cls, data, years, categories, context_texts):
        """Process categories and show top fuels for each."""
        if years:
            for year in years:
                year_str = str(year)
                if year_str in data:
                    for category in categories:
                        if category in data[year_str]:
                            fuel_data = data[year_str][category]['fuel_counts']
                            sorted_fuels = sorted(fuel_data.items(), key=lambda x: x[1], reverse=True)
                            context_texts.append(f"Top fuel types for {category} in {year}:")
                            for fuel, count in sorted_fuels[:3]:
                                if count > 0:
                                    context_texts.append(f"- {fuel}: {count:,} vehicles")
        else:
            # Latest year
            available_years = sorted([int(y) for y in data.keys()])
            if available_years:
                latest_year = str(max(available_years))
                for category in categories:
                    if category in data[latest_year]:
                        fuel_data = data[latest_year][category]['fuel_counts']
                        sorted_fuels = sorted(fuel_data.items(), key=lambda x: x[1], reverse=True)
                        context_texts.append(f"Top fuel types for {category} in {latest_year}:")
                        for fuel, count in sorted_fuels[:3]:
                            if count > 0:
                                context_texts.append(f"- {fuel}: {count:,} vehicles")
    
    @classmethod
    def _process_fuels_with_top_categories(cls, data, years, fuels, context_texts):
        """Process fuels and show top categories for each."""
        if years:
            for year in years:
                year_str = str(year)
                if year_str in data:
                    for fuel in fuels:
                        fuel_by_category = {}
                        for category, cat_data in data[year_str].items():
                            if fuel in cat_data['fuel_counts']:
                                fuel_by_category[category] = cat_data['fuel_counts'][fuel]
                        if fuel_by_category:
                            sorted_categories = sorted(fuel_by_category.items(), key=lambda x: x[1], reverse=True)
                            context_texts.append(f"Top categories for {fuel} vehicles in {year}:")
                            for category, count in sorted_categories[:3]:
                                if count > 0:
                                    context_texts.append(f"- {category}: {count:,} vehicles")
        else:
            # Latest year
            available_years = sorted([int(y) for y in data.keys()])
            if available_years:
                latest_year = str(max(available_years))
                for fuel in fuels:
                    fuel_by_category = {}
                    for category, cat_data in data[latest_year].items():
                        if fuel in cat_data['fuel_counts']:
                            fuel_by_category[category] = cat_data['fuel_counts'][fuel]
                    if fuel_by_category:
                        sorted_categories = sorted(fuel_by_category.items(), key=lambda x: x[1], reverse=True)
                        context_texts.append(f"Top categories for {fuel} vehicles in {latest_year}:")
                        for category, count in sorted_categories[:3]:
                            if count > 0:
                                context_texts.append(f"- {category}: {count:,} vehicles")
    
    @staticmethod
    def _extract_years_from_query(query):
        """Extract year mentions from a query string."""
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', query)]
        return sorted(set(years))
    
    @classmethod
    def _extract_categories_from_query(cls, query):
        """Extract vehicle categories from a query string with fuzzy matching."""
        found_categories = set()
        query_lower = query.lower()
        
        # First try exact matches
        for term, code in cls.CATEGORY_MAPPING.items():
            if term in query_lower:
                found_categories.add(code)
        
        # If no exact matches, try fuzzy matching
        if not found_categories:
            words = query_lower.split()
            for word in words:
                for term, code in cls.CATEGORY_MAPPING.items():
                    if difflib.SequenceMatcher(None, word, term).ratio() > 0.8:
                        found_categories.add(code)
        
        return list(found_categories)
    
    @classmethod
    def _extract_fuels_from_query(cls, query):
        """Extract fuel types from a query string with fuzzy matching."""
        found_fuels = set()
        query_lower = query.lower()
        
        # First try exact matches
        for term, code in cls.FUEL_MAPPING.items():
            if term in query_lower:
                if isinstance(code, list):
                    found_fuels.update(code)
                else:
                    found_fuels.add(code)
        
        # If no exact matches, try fuzzy matching
        if not found_fuels:
            words = query_lower.split()
            for word in words:
                for term, code in cls.FUEL_MAPPING.items():
                    if difflib.SequenceMatcher(None, word, term).ratio() > 0.8:
                        if isinstance(code, list):
                            found_fuels.update(code)
                        else:
                            found_fuels.add(code)
        
        return list(found_fuels)
    
    @staticmethod
    def _process_total_query(query_lower, year_to_entry, context_texts):
        """Process total/sum queries."""
        if any(word in query_lower for word in ['total', 'sum']):
            match = re.search(r'(19\d{2}|20\d{2}).*(19\d{2}|20\d{2})', query_lower)
            if match:
                y1, y2 = sorted([int(match.group(1)), int(match.group(2))])
                relevant_years = [y for y in year_to_entry.keys() if y1 <= y <= y2]
                if relevant_years:
                    total = sum(year_to_entry[y]['vehicles'] for y in relevant_years)
                    context_texts.append(f"The total number of registered vehicles from {y1} to {y2} is {total:,}.")
                    for year in range(y1, y2 + 1):
                        if year in year_to_entry:
                            context_texts.append(f"In {year}, there were {year_to_entry[year]['vehicles']:,} registered vehicles.")
    
    @staticmethod
    def _process_year_mentions(query, year_to_entry, context_texts):
        """Process specific year mentions in queries."""
        year_matches = set(map(int, re.findall(r'\b(19\d{2}|20\d{2})\b', query)))
        for year in year_matches:
            if year in year_to_entry:
                context_texts.append(f"In {year}, there were {year_to_entry[year]['vehicles']:,} registered vehicles.")
    
    @staticmethod
    def _process_trend_query(query_lower, year_to_entry, min_year, max_year, context_texts):
        """Process trend/growth queries."""
        trend_keywords = ['trend', 'growth', 'decline', 'increase', 'decrease']
        if any(k in query_lower for k in trend_keywords):
            if min_year in year_to_entry and max_year in year_to_entry:
                start, end = year_to_entry[min_year], year_to_entry[max_year]
                change = end['vehicles'] - start['vehicles']
                pct = (change / start['vehicles']) * 100
                trend = "increased" if change > 0 else "decreased"
                context_texts.append(f"From {min_year} to {max_year}, vehicle registrations {trend} by {abs(change):,} ({abs(pct):.1f}%).")
    
    @staticmethod
    def _process_prediction_query(query_lower, year_to_entry, context_texts):
        """Process prediction/forecast queries."""
        prediction_keywords = ['predict', 'forecast', 'future', 'projection', 'next']
        years_keywords = ['year', 'years']
        
        has_prediction_term = any(k in query_lower for k in prediction_keywords)
        has_year_term = any(y in query_lower for y in years_keywords)
        
        if has_prediction_term or (has_year_term and 'next' in query_lower):
            current_year = datetime.datetime.now().year
            
            # Get how many future years were requested
            num_years = 5 # Default to 5 years
            match = re.search(r'next\s+(\d+)\s+years?', query_lower)
            if match:
                try:
                    num_years = int(match.group(1))
                except ValueError:
                    pass
            
            # Get prediction years
            prediction_years = sorted([y for y in year_to_entry.keys() if y > current_year])
            if prediction_years:
                # Limit to requested number of years or available predictions
                prediction_years = prediction_years[:min(num_years, len(prediction_years))]
                context_texts.append(f"Predicted vehicle registrations for the next {len(prediction_years)} years:")
                for year in prediction_years:
                    context_texts.append(f"In {year}, the predicted number is {year_to_entry[year]['vehicles']:,} registered vehicles.")
    
    @classmethod
    def _process_time_periods(cls, query_lower, year_to_entry, context_texts):
        """Compare vehicle data between two time periods."""
        # Extract time periods using regex
        period_pattern = re.compile(r'(\d{4})[^\d]+(\d{4})[^\d]+(\d{4})[^\d]+(\d{4})')
        match = period_pattern.search(query_lower)
        if not match:
            return False
        
        period1_start, period1_end, period2_start, period2_end = map(int, match.groups())
        
        # Get data for both periods
        combined_data = year_to_entry
        if not combined_data:
            return False
        
        # Calculate period totals
        period1_years = [y for y in range(period1_start, period1_end + 1) if y in combined_data]
        period2_years = [y for y in range(period2_start, period2_end + 1) if y in combined_data]
        
        if not period1_years or not period2_years:
            context_texts.append("Insufficient data for one or both time periods.")
            return True
        
        period1_total = sum(combined_data[y]['vehicles'] for y in period1_years)
        period2_total = sum(combined_data[y]['vehicles'] for y in period2_years)
        
        # Add comparison to context
        context_texts.append(f"Period {period1_start}-{period1_end}: {period1_total:,} vehicles")
        context_texts.append(f"Period {period2_start}-{period2_end}: {period2_total:,} vehicles")
        
        change = period2_total - period1_total
        pct_change = cls.calculate_percentage_change(period1_total, period2_total)
        
        if change > 0:
            context_texts.append(f"The second period had {change:,} more vehicles (+{pct_change:.1f}%)")
        else:
            context_texts.append(f"The second period had {abs(change):,} fewer vehicles ({pct_change:.1f}%)")
        
        return True
    
    @classmethod
    def calculate_percentage_change(cls, start_value, end_value):
        """Calculate percentage change between two values."""
        if start_value == 0:
            return float('inf') # Handle division by zero
        return ((end_value - start_value) / start_value) * 100
    
    @classmethod
    def calculate_compound_growth_rate(cls, start_value, end_value, years):
        """Calculate compound annual growth rate."""
        if start_value == 0 or years == 0:
            return 0
        return (pow(end_value / start_value, 1 / years) - 1) * 100
