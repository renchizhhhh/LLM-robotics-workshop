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
                        "PickupOfDrinks": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "DropOffArea": {"x": 2.0, "y": 0.5, "z": 0.0, "pick_yaw": None, "drop_yaw": 1.57}
                    },
                    "zones": {},
                    "synonyms": {
                        "pickup of drinks": "PickupOfDrinks",
                        "drink pickup": "PickupOfDrinks",
                        "pickup location": "PickupOfDrinks",
                        "drop-off location": "DropOffArea",
                        "drop off location": "DropOffArea",
                        "delivery area": "DropOffArea"
                    }
                }
            },
            "2": {
                "name": "Grocery Store",
                "description": "Supermarket environment with produce, dairy, and checkout",
                "config": {
                    "waypoints": {
                        "PickupOfDrinks": {"x": 3.5, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "CheckoutCounter": {"x": 1.0, "y": 2.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 1.57},
                        "PickupOfDrinks2": {"x": -2.0, "y": 1.5, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None}
                    },
                    "zones": {
                        "VegZoneA": {
                            "centroid": {"x": 1.8, "y": 2.2, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["vegetables", "produce", "fresh"]
                        },
                        "DairySection": {
                            "centroid": {"x": -1.5, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["dairy", "milk", "cheese", "yogurt"]
                        }
                    },
                    "synonyms": {
                        "pickup of drinks": "PickupOfDrinks",
                        "drink pickup": "PickupOfDrinks",
                        "pickup location": "PickupOfDrinks",
                        "pickup of drinks 2": "PickupOfDrinks2",
                        "drink pickup 2": "PickupOfDrinks2",
                        "second pickup": "PickupOfDrinks2",
                        "where the vegetables are": "VegZoneA",
                        "vegetable area": "VegZoneA",
                        "veggies": "VegZoneA",
                        "produce section": "VegZoneA",
                        "checkout": "CheckoutCounter",
                        "cashier": "CheckoutCounter",
                        "dairy section": "DairySection",
                        "milk area": "DairySection"
                    }
                }
            },
            "3": {
                "name": "Warehouse",
                "description": "Industrial warehouse with loading dock and storage areas",
                "config": {
                    "waypoints": {
                        "LoadingDock": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "StorageA": {"x": 5.0, "y": 2.0, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "StorageB": {"x": 5.0, "y": -2.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None},
                        "ShippingArea": {"x": -3.0, "y": 0.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0},
                        "QualityControl": {"x": 2.0, "y": -4.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "HighValueZone": {
                            "centroid": {"x": 3.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["secure", "valuable", "electronics", "precious"]
                        }
                    },
                    "synonyms": {
                        "loading dock": "LoadingDock",
                        "dock": "LoadingDock",
                        "storage area a": "StorageA",
                        "storage a": "StorageA",
                        "storage area b": "StorageB",
                        "storage b": "StorageB",
                        "shipping": "ShippingArea",
                        "shipping area": "ShippingArea",
                        "quality control": "QualityControl",
                        "qc": "QualityControl",
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
                        "Reception": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "MeetingRoomA": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "MeetingRoomB": {"x": 3.0, "y": -2.0, "z": 0.0, "pick_yaw": -1.57, "drop_yaw": None},
                        "WorkstationArea": {"x": -2.0, "y": 0.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0},
                        "Kitchen": {"x": -4.0, "y": 3.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "ExecutiveZone": {
                            "centroid": {"x": 1.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["executive", "management", "private"]
                        }
                    },
                    "synonyms": {
                        "reception": "Reception",
                        "front desk": "Reception",
                        "meeting room a": "MeetingRoomA",
                        "conference room a": "MeetingRoomA",
                        "meeting room b": "MeetingRoomB",
                        "conference room b": "MeetingRoomB",
                        "workstation area": "WorkstationArea",
                        "desk area": "WorkstationArea",
                        "kitchen": "Kitchen",
                        "break room": "Kitchen",
                        "executive area": "ExecutiveZone",
                        "management zone": "ExecutiveZone"
                    }
                }
            },
            "5": {
                "name": "Hospital",
                "description": "Medical facility with patient rooms, pharmacy, and emergency areas",
                "config": {
                    "waypoints": {
                        "PickupOfDrinks": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "PickupOfDrinks2": {"x": 4.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "EmergencyRoom": {"x": -3.0, "y": 2.0, "z": 0.0, "pick_yaw": 1.57, "drop_yaw": None},
                        "SurgeryPrep": {"x": 2.0, "y": -3.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0}
                    },
                    "zones": {
                        "PatientWard": {
                            "centroid": {"x": 1.0, "y": 1.5, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["patients", "rooms", "beds"]
                        },
                        "CriticalCare": {
                            "centroid": {"x": -1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["critical", "icu", "intensive"]
                        }
                    },
                    "synonyms": {
                        "pickup of drinks": "PickupOfDrinks",
                        "drink pickup": "PickupOfDrinks",
                        "pickup location": "PickupOfDrinks",
                        "pickup of drinks 2": "PickupOfDrinks2",
                        "drink pickup 2": "PickupOfDrinks2",
                        "second pickup": "PickupOfDrinks2",
                        "coffee pickup": "PickupOfDrinks",
                        "coffee location": "PickupOfDrinks",
                        "drink station": "PickupOfDrinks",
                        "beverage pickup": "PickupOfDrinks",
                        "emergency room": "EmergencyRoom",
                        "er": "EmergencyRoom",
                        "surgery prep": "SurgeryPrep",
                        "prep room": "SurgeryPrep",
                        "patient ward": "PatientWard",
                        "ward": "PatientWard",
                        "critical care": "CriticalCare",
                        "icu": "CriticalCare"
                    }
                }
            },
            "6": {
                "name": "Restaurant Kitchen",
                "description": "Commercial kitchen with prep stations, cooking areas, and service",
                "config": {
                    "waypoints": {
                        "PrepStation": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "CookingArea": {"x": 3.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "ServiceWindow": {"x": 0.0, "y": 2.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 1.57},
                        "StorageRoom": {"x": -2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "ColdStorage": {
                            "centroid": {"x": -1.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["cold", "refrigerated", "fresh"]
                        },
                        "HotLine": {
                            "centroid": {"x": 2.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["hot", "cooking", "grill"]
                        }
                    },
                    "synonyms": {
                        "prep station": "PrepStation",
                        "prep area": "PrepStation",
                        "cooking area": "CookingArea",
                        "cooking station": "CookingArea",
                        "service window": "ServiceWindow",
                        "pass": "ServiceWindow",
                        "storage room": "StorageRoom",
                        "pantry": "StorageRoom",
                        "cold storage": "ColdStorage",
                        "refrigerator": "ColdStorage",
                        "hot line": "HotLine",
                        "grill area": "HotLine"
                    }
                }
            },
            "7": {
                "name": "Laboratory",
                "description": "Research lab with equipment stations and specimen areas",
                "config": {
                    "waypoints": {
                        "MainBench": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "MicroscopeStation": {"x": 2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "SpecimenStorage": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "WasteDisposal": {"x": 0.0, "y": -2.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0}
                    },
                    "zones": {
                        "CleanRoom": {
                            "centroid": {"x": 1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["sterile", "clean", "contamination-free"]
                        },
                        "HazardousArea": {
                            "centroid": {"x": -1.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["hazardous", "chemicals", "dangerous"]
                        }
                    },
                    "synonyms": {
                        "main bench": "MainBench",
                        "workbench": "MainBench",
                        "microscope station": "MicroscopeStation",
                        "microscope": "MicroscopeStation",
                        "specimen storage": "SpecimenStorage",
                        "sample storage": "SpecimenStorage",
                        "waste disposal": "WasteDisposal",
                        "waste area": "WasteDisposal",
                        "clean room": "CleanRoom",
                        "sterile area": "CleanRoom",
                        "hazardous area": "HazardousArea",
                        "chemical storage": "HazardousArea"
                    }
                }
            },
            "8": {
                "name": "Retail Store",
                "description": "Department store with clothing, electronics, and customer service",
                "config": {
                    "waypoints": {
                        "CustomerService": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "ElectronicsSection": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "ClothingSection": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "FittingRooms": {"x": -3.0, "y": -1.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0}
                    },
                    "zones": {
                        "KidsSection": {
                            "centroid": {"x": 1.0, "y": -2.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["children", "toys", "kids"]
                        },
                        "HomeGoods": {
                            "centroid": {"x": 2.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["home", "furniture", "decor"]
                        }
                    },
                    "synonyms": {
                        "customer service": "CustomerService",
                        "service desk": "CustomerService",
                        "electronics section": "ElectronicsSection",
                        "electronics": "ElectronicsSection",
                        "clothing section": "ClothingSection",
                        "apparel": "ClothingSection",
                        "fitting rooms": "FittingRooms",
                        "changing rooms": "FittingRooms",
                        "kids section": "KidsSection",
                        "children's area": "KidsSection",
                        "home goods": "HomeGoods",
                        "furniture section": "HomeGoods"
                    }
                }
            },
            "9": {
                "name": "Airport Terminal",
                "description": "Airport with gates, security, and baggage areas",
                "config": {
                    "waypoints": {
                        "SecurityCheckpoint": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "GateA1": {"x": 4.0, "y": 1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "GateA2": {"x": 4.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "BaggageClaim": {"x": -3.0, "y": 0.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0},
                        "InformationDesk": {"x": 1.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "DutyFree": {
                            "centroid": {"x": 2.0, "y": 0.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["shopping", "duty-free", "luxury"]
                        },
                        "FoodCourt": {
                            "centroid": {"x": 0.0, "y": -2.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["food", "restaurants", "dining"]
                        }
                    },
                    "synonyms": {
                        "security checkpoint": "SecurityCheckpoint",
                        "security": "SecurityCheckpoint",
                        "gate a1": "GateA1",
                        "gate 1": "GateA1",
                        "gate a2": "GateA2",
                        "gate 2": "GateA2",
                        "baggage claim": "BaggageClaim",
                        "baggage": "BaggageClaim",
                        "information desk": "InformationDesk",
                        "info desk": "InformationDesk",
                        "duty free": "DutyFree",
                        "shopping area": "DutyFree",
                        "food court": "FoodCourt",
                        "restaurants": "FoodCourt"
                    }
                }
            },
            "10": {
                "name": "Construction Site",
                "description": "Building site with materials, tools, and work areas",
                "config": {
                    "waypoints": {
                        "ToolShed": {"x": 0.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "MaterialStorage": {"x": 3.0, "y": 2.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "WorkSite": {"x": 5.0, "y": 0.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None},
                        "SafetyStation": {"x": -2.0, "y": 1.0, "z": 0.0, "pick_yaw": None, "drop_yaw": 0.0}
                    },
                    "zones": {
                        "HazardZone": {
                            "centroid": {"x": 4.0, "y": 1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["dangerous", "hazardous", "caution"]
                        },
                        "CleanArea": {
                            "centroid": {"x": 1.0, "y": -1.0, "z": 0.0},
                            "yaw_hint": 0.0,
                            "tags": ["safe", "clean", "finished"]
                        }
                    },
                    "synonyms": {
                        "tool shed": "ToolShed",
                        "tool storage": "ToolShed",
                        "material storage": "MaterialStorage",
                        "materials": "MaterialStorage",
                        "work site": "WorkSite",
                        "construction area": "WorkSite",
                        "safety station": "SafetyStation",
                        "safety area": "SafetyStation",
                        "hazard zone": "HazardZone",
                        "dangerous area": "HazardZone",
                        "clean area": "CleanArea",
                        "safe zone": "CleanArea"
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
