#!/usr/bin/env python3
"""
World Manager for Spot Robot - Provides 10 different world configurations
"""

import json
import os

class WorldManager:
    """Manages different world configurations for the Spot robot."""
    
    def __init__(self):
        self.worlds = self._create_world_configurations()
    
    def _create_world_configurations(self):
        """Create 10 different world configurations."""
        return {
            "1": {
                "name": "Simple Pick & Drop",
                "description": "Basic two-location scenario for testing",
                "config": {
                    "waypoints": {
                        "PICKUP_BEVERAGES": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "DeliveryArea": {
                            "centroid": {"x": 2.0, "y": 0.5, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["delivery", "drop-off", "destination"]
                        }
                    },
                    "synonyms": {
                        "beverages": "PICKUP_BEVERAGES",
                        "drinks": "PICKUP_BEVERAGES",
                        "drink station": "PICKUP_BEVERAGES",
                        "beverage station": "PICKUP_BEVERAGES",
                        "delivery area": "DeliveryArea",
                        "drop-off area": "DeliveryArea",
                        "destination": "DeliveryArea"
                    }
                }
            },
            "2": {
                "name": "Grocery Store",
                "description": "Supermarket environment with produce, dairy, and checkout",
                "config": {
                    "waypoints": {
                        "PICKUP_BEVERAGES": {"x": 3.5, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_PRODUCE": {"x": 1.8, "y": 2.2, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "PICKUP_DAIRY": {"x": -1.5, "y": 1.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None}
                    },
                    "zones": {
                        "Checkout": {
                            "centroid": {"x": 1.0, "y": 2.0, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["checkout", "cashier", "payment"]
                        }
                    },
                    "synonyms": {
                        "drinks": "PICKUP_BEVERAGES",
                        "beverages": "PICKUP_BEVERAGES",
                        "soda": "PICKUP_BEVERAGES",
                        "water bottle": "PICKUP_BEVERAGES",
                        "drink aisle": "PICKUP_BEVERAGES",

                        "fruits": "PICKUP_PRODUCE",
                        "produce": "PICKUP_PRODUCE",
                        "apples": "PICKUP_PRODUCE",
                        "bananas": "PICKUP_PRODUCE",
                        "oranges": "PICKUP_PRODUCE",

                        "dairy": "PICKUP_DAIRY",
                        "milk": "PICKUP_DAIRY",
                        "cheese": "PICKUP_DAIRY",
                        "yogurt": "PICKUP_DAIRY",
                        "butter": "PICKUP_DAIRY",

                        "checkout": "Checkout",
                        "cashier": "Checkout",
                        "payment": "Checkout"
                    }
                }
            },
            "3": {
                "name": "Warehouse",
                "description": "Industrial warehouse with loading dock and storage areas",
                "config": {
                    "waypoints": {
                        "PICKUP_INCOMING": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_ELECTRONICS": {"x": 5.0, "y": 2.0, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "PICKUP_TEXTILES": {"x": 5.0, "y": -2.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None},
                        "PICKUP_TOOLS": {"x": 2.0, "y": -4.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "ShippingArea": {
                            "centroid": {"x": -3.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["shipping", "outbound", "dispatch"]
                        },
                        "HighValueZone": {
                            "centroid": {"x": 3.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["secure", "valuable", "electronics", "precious"]
                        }
                    },
                    "synonyms": {
                        "incoming goods": "PICKUP_INCOMING",
                        "loading dock": "PICKUP_INCOMING",
                        "dock": "PICKUP_INCOMING",
                        "electronics": "PICKUP_ELECTRONICS",
                        "electronic items": "PICKUP_ELECTRONICS",
                        "gadgets": "PICKUP_ELECTRONICS",
                        "textiles": "PICKUP_TEXTILES",
                        "fabric": "PICKUP_TEXTILES",
                        "clothing": "PICKUP_TEXTILES",
                        "tools": "PICKUP_TOOLS",
                        "hardware": "PICKUP_TOOLS",
                        "shipping area": "ShippingArea",
                        "shipping": "ShippingArea",
                        "outbound": "ShippingArea",
                        "secure zone": "HighValueZone",
                        "valuable items": "HighValueZone"
                    }
                }
            },
            "4": {
                "name": "Office Building",
                "description": "Corporate office with reception, meeting rooms, and workstations",
                "config": {
                    "waypoints": {
                        "PICKUP_STATIONERY": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_OFFICE_SUPPLIES": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "PICKUP_COMPUTERS": {"x": 3.0, "y": -2.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None},
                        "PICKUP_COFFEE": {"x": -4.0, "y": 3.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Desks": {
                            "centroid": {"x": -2.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["workstations", "desks", "employees"]
                        },
                        "BossOffice": {
                            "centroid": {"x": 1.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["executive", "management", "private"]
                        }
                    },
                    "synonyms": {
                        "pens": "PICKUP_STATIONERY",
                        "pencils": "PICKUP_STATIONERY",
                        "markers": "PICKUP_STATIONERY",
                        "writing": "PICKUP_STATIONERY",
                        "supplies": "PICKUP_OFFICE_SUPPLIES",
                        "office supplies": "PICKUP_OFFICE_SUPPLIES",
                        "staplers": "PICKUP_OFFICE_SUPPLIES",
                        "paperclips": "PICKUP_OFFICE_SUPPLIES",
                        "computers": "PICKUP_COMPUTERS",
                        "laptops": "PICKUP_COMPUTERS",
                        "tech": "PICKUP_COMPUTERS",
                        "coffee": "PICKUP_COFFEE",
                        "espresso": "PICKUP_COFFEE",
                        "cappuccino": "PICKUP_COFFEE",
                        "desks": "Desks",
                        "workstations": "Desks",
                        "boss office": "BossOffice",
                        "executive": "BossOffice"
                    }
                }
            },
            "5": {
                "name": "Hospital",
                "description": "Medical facility with patient rooms, pharmacy, and emergency areas",
                "config": {
                    "waypoints": {
                        "PICKUP_MEDICATIONS": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_PPE": {"x": 4.0, "y": 1.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None},
                        "PICKUP_EMERGENCY_SUPPLIES": {"x": -3.0, "y": 2.0, "z": 0.0, "pick_yaw": 3.14, "drop_yaw": None}
                    },
                    "zones": {
                        "Patients": {
                            "centroid": {"x": 1.0, "y": 1.5, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["patients", "rooms", "beds"]
                        },
                        "ICU": {
                            "centroid": {"x": -1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["critical", "icu", "intensive"]
                        },
                        "Surgery": {
                            "centroid": {"x": 2.0, "y": -3.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["surgery", "prep", "operating"]
                        }
                    },
                    "synonyms": {
                        "medicine": "PICKUP_MEDICATIONS",
                        "pills": "PICKUP_MEDICATIONS",
                        "medications": "PICKUP_MEDICATIONS",
                        "pharmacy": "PICKUP_MEDICATIONS",
                        "gloves": "PICKUP_PPE",
                        "masks": "PICKUP_PPE",
                        "ppe": "PICKUP_PPE",
                        "emergency kit": "PICKUP_EMERGENCY_SUPPLIES",
                        "first aid": "PICKUP_EMERGENCY_SUPPLIES",
                        "defibrillator": "PICKUP_EMERGENCY_SUPPLIES",
                        "patients": "Patients",
                        "ward": "Patients",
                        "icu": "ICU",
                        "critical care": "ICU",
                        "surgery": "Surgery",
                        "operating room": "Surgery"
                    }
                }
            },
            "6": {
                "name": "Restaurant Kitchen",
                "description": "Commercial kitchen with prep stations, cooking areas, and service",
                "config": {
                    "waypoints": {
                        "PICKUP_INGREDIENTS": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_KNIVES": {"x": 3.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_UTENSILS": {"x": -2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Counter": {
                            "centroid": {"x": 0.0, "y": 2.0, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["service", "pass", "orders"]
                        },
                        "Fridge": {
                            "centroid": {"x": -1.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["cold", "refrigerated", "fresh"]
                        },
                        "Stove": {
                            "centroid": {"x": 2.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["hot", "cooking", "grill"]
                        }
                    },
                    "synonyms": {
                        "ingredients": "PICKUP_INGREDIENTS",
                        "vegetables": "PICKUP_INGREDIENTS",
                        "meat": "PICKUP_INGREDIENTS",
                        "sauce": "PICKUP_INGREDIENTS",
                        "knives": "PICKUP_KNIVES",
                        "chef knife": "PICKUP_KNIVES",
                        "forks": "PICKUP_UTENSILS",
                        "utensils": "PICKUP_UTENSILS",
                        "cutlery": "PICKUP_UTENSILS",
                        "counter": "Counter",
                        "service": "Counter",
                        "fridge": "Fridge",
                        "refrigerator": "Fridge",
                        "stove": "Stove",
                        "cooking": "Stove"
                    }
                }
            },
            "7": {
                "name": "Laboratory",
                "description": "Research lab with equipment stations and specimen areas",
                "config": {
                    "waypoints": {
                        "PICKUP_GLASSWARE": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_MICROSCOPES": {"x": 2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_SAMPLES": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Trash": {
                            "centroid": {"x": 0.0, "y": -2.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["waste", "disposal", "hazardous"]
                        },
                        "CleanRoom": {
                            "centroid": {"x": 1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["sterile", "clean", "contamination-free"]
                        },
                        "Chemicals": {
                            "centroid": {"x": -1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["hazardous", "chemicals", "dangerous"]
                        }
                    },
                    "synonyms": {
                        "test tubes": "PICKUP_GLASSWARE",
                        "glass tubes": "PICKUP_GLASSWARE",
                        "microscopes": "PICKUP_MICROSCOPES",
                        "microscope": "PICKUP_MICROSCOPES",
                        "samples": "PICKUP_SAMPLES",
                        "specimens": "PICKUP_SAMPLES",
                        "vials": "PICKUP_SAMPLES",
                        "trash": "Trash",
                        "waste": "Trash",
                        "clean room": "CleanRoom",
                        "sterile": "CleanRoom",
                        "chemicals": "Chemicals",
                        "hazardous": "Chemicals"
                    }
                }
            },
            "8": {
                "name": "Retail Store",
                "description": "Department store with clothing, electronics, and customer service",
                "config": {
                    "waypoints": {
                        "PICKUP_ELECTRONICS_PHONES": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_APPAREL_TOPS": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_TOYS": {"x": 1.0, "y": -2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Help": {
                            "centroid": {"x": 1.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["service", "help", "information"]
                        },
                        "Fitting": {
                            "centroid": {"x": -3.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["fitting", "changing", "rooms"]
                        },
                        "Furniture": {
                            "centroid": {"x": 2.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["home", "furniture", "decor"]
                        }
                    },
                    "synonyms": {
                        "phones": "PICKUP_ELECTRONICS_PHONES",
                        "smartphones": "PICKUP_ELECTRONICS_PHONES",
                        "electronics": "PICKUP_ELECTRONICS_PHONES",
                        "shirts": "PICKUP_APPAREL_TOPS",
                        "t shirts": "PICKUP_APPAREL_TOPS",
                        "clothing": "PICKUP_APPAREL_TOPS",
                        "toys": "PICKUP_TOYS",
                        "board games": "PICKUP_TOYS",
                        "help": "Help",
                        "service": "Help",
                        "fitting": "Fitting",
                        "changing": "Fitting",
                        "furniture": "Furniture",
                        "home": "Furniture"
                    }
                }
            },
            "9": {
                "name": "Airport Terminal",
                "description": "Airport with gates, security, and baggage areas",
                "config": {
                    "waypoints": {
                        "PICKUP_SECURITY_ITEMS": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_TICKETING": {"x": 4.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_MAPS_INFO": {"x": 1.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Gate": {
                            "centroid": {"x": 4.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["gate", "boarding", "departure"]
                        },
                        "Baggage": {
                            "centroid": {"x": -3.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["baggage", "arrival", "luggage"]
                        },
                        "Shopping": {
                            "centroid": {"x": 2.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["shopping", "duty-free", "luxury"]
                        },
                        "Food": {
                            "centroid": {"x": 0.0, "y": -2.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["food", "restaurants", "dining"]
                        }
                    },
                    "synonyms": {
                        "security tray": "PICKUP_SECURITY_ITEMS",
                        "bins": "PICKUP_SECURITY_ITEMS",
                        "belt": "PICKUP_SECURITY_ITEMS",
                        "tickets": "PICKUP_TICKETING",
                        "boarding passes": "PICKUP_TICKETING",
                        "map": "PICKUP_MAPS_INFO",
                        "information map": "PICKUP_MAPS_INFO",
                        "gate": "Gate",
                        "boarding": "Gate",
                        "baggage": "Baggage",
                        "luggage": "Baggage",
                        "shopping": "Shopping",
                        "duty free": "Shopping",
                        "food": "Food",
                        "restaurants": "Food"
                    }
                }
            },
            "10": {
                "name": "Construction Site",
                "description": "Building site with materials, tools, and work areas",
                "config": {
                    "waypoints": {
                        "PICKUP_HANDTOOLS": {"x": 1.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_LUMBER": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PICKUP_SAFETY_HELMETS": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "Building": {
                            "centroid": {"x": 5.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["construction", "work", "building"]
                        },
                        "Danger": {
                            "centroid": {"x": 4.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["dangerous", "hazardous", "caution"]
                        },
                        "Safe": {
                            "centroid": {"x": 1.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["safe", "clean", "finished"]
                        }
                    },
                    "synonyms": {
                        "hammer": "PICKUP_HANDTOOLS",
                        "hand tools": "PICKUP_HANDTOOLS",
                        "hammers": "PICKUP_HANDTOOLS",
                        "wood": "PICKUP_LUMBER",
                        "timber": "PICKUP_LUMBER",
                        "lumber": "PICKUP_LUMBER",
                        "helmet": "PICKUP_SAFETY_HELMETS",
                        "hard hat": "PICKUP_SAFETY_HELMETS",
                        "helmets": "PICKUP_SAFETY_HELMETS",
                        "building": "Building",
                        "construction": "Building",
                        "danger": "Danger",
                        "hazardous": "Danger",
                        "safe": "Safe",
                        "clean": "Safe"
                    }
                }
            }
        }
    
    def get_world_list(self):
        """Get list of available worlds."""
        return [(key, world["name"], world["description"]) for key, world in self.worlds.items()]
    
    def get_world_config(self, world_id):
        """Get configuration for a specific world."""
        if world_id in self.worlds:
            return self.worlds[world_id]["config"]
        return None
    
    def get_world_info(self, world_id):
        """Get name and description for a specific world."""
        if world_id in self.worlds:
            return self.worlds[world_id]["name"], self.worlds[world_id]["description"]
        return None, None
    
    def save_world_to_file(self, world_id, filepath):
        """Save a world configuration to a JSON file."""
        if world_id in self.worlds:
            with open(filepath, 'w') as f:
                json.dump(self.worlds[world_id]["config"], f, indent=2)
            return True
        return False
    
    def load_world_from_file(self, filepath):
        """Load a world configuration from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading world from {filepath}: {e}")
            return None

    @staticmethod
    def resolve_pickup_area(item_query: str, config: dict):
        """Return the PICKUP_* key for a natural-language item request."""
        q = (item_query or "").strip().lower()
        syn = config.get("synonyms", {}) or {}
        # Exact hit
        if q in syn:
            return syn[q]
        # Token-wise fallback
        tokens = q.replace("-", " ").replace("_", " ").split()
        for i in range(len(tokens), 0, -1):
            k = " ".join(tokens[:i])
            if k in syn:
                return syn[k]
        return None

def main():
    """Demo the world manager."""
    wm = WorldManager()
    
    print("Available Worlds:")
    print("=" * 50)
    
    for world_id, name, description in wm.get_world_list():
        print(f"{world_id}. {name}")
        print(f"   {description}")
        print()
    
    # Example usage
    print("Example: Loading Grocery Store world")
    config = wm.get_world_config("2")
    if config:
        print(f"Waypoints: {list(config['waypoints'].keys())}")
        print(f"Zones: {list(config['zones'].keys())}")
        print(f"Synonyms: {len(config['synonyms'])}")

if __name__ == "__main__":
    main()
