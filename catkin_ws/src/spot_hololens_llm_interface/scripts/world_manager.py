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
        """Create 11 different world configurations with grid-based coordinates."""
        return {
            "1": {
                "name": "Simple Pick & Drop",
                "description": "Basic two-location scenario for testing",
                "config": {
                    "waypoints": {
                        "PICKUP_BEVERAGES": {"row": 2, "col": 3, "pick_direction": "N"}
                    },
                    "zones": {
                        "DeliveryArea": {
                            "row": 7, "col": 8,
                            "direction": "S",
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
                        "PICKUP_BEVERAGES": {"row": 3, "col": 4, "pick_direction": "N"},
                        "PICKUP_PRODUCE": {"row": 1, "col": 2, "pick_direction": "E"},
                        "PICKUP_DAIRY": {"row": 1, "col": 6, "pick_direction": "W"}
                    },
                    "zones": {
                        "Checkout": {
                            "row": 7, "col": 5,
                            "direction": "E",
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
                        "PICKUP_INCOMING": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_ELECTRONICS": {"row": 2, "col": 7, "pick_direction": "E"},
                        "PICKUP_TEXTILES": {"row": 6, "col": 7, "pick_direction": "W"},
                        "PICKUP_TOOLS": {"row": 4, "col": 2, "pick_direction": "N"}
                    },
                    "zones": {
                        "ShippingArea": {
                            "row": 0, "col": 5,
                            "direction": "N",
                            "tags": ["shipping", "outbound", "dispatch"]
                        },
                        "HighValueZone": {
                            "row": 8, "col": 5,
                            "direction": "S",
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
                        "PICKUP_STATIONERY": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_OFFICE_SUPPLIES": {"row": 2, "col": 3, "pick_direction": "E"},
                        "PICKUP_COMPUTERS": {"row": 6, "col": 3, "pick_direction": "W"},
                        "PICKUP_COFFEE": {"row": 0, "col": 6, "pick_direction": "N"}
                    },
                    "zones": {
                        "Desks": {
                            "row": 4, "col": 4,
                            "direction": "N",
                            "tags": ["workstations", "desks", "employees"]
                        },
                        "BossOffice": {
                            "row": 8, "col": 8,
                            "direction": "S",
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
                        "PICKUP_MEDICATIONS": {"row": 2, "col": 3, "pick_direction": "N"},
                        "PICKUP_PPE": {"row": 1, "col": 7, "pick_direction": "W"},
                        "PICKUP_EMERGENCY_SUPPLIES": {"row": 0, "col": 2, "pick_direction": "S"}
                    },
                    "zones": {
                        "Patients": {
                            "row": 3, "col": 4,
                            "direction": "E",
                            "tags": ["patients", "rooms", "beds"]
                        },
                        "ICU": {
                            "row": 5, "col": 5,
                            "direction": "W",
                            "tags": ["critical", "icu", "intensive"]
                        },
                        "Surgery": {
                            "row": 8, "col": 3,
                            "direction": "N",
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
                        "PICKUP_INGREDIENTS": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_KNIVES": {"row": 3, "col": 5, "pick_direction": "N"},
                        "PICKUP_UTENSILS": {"row": 6, "col": 2, "pick_direction": "N"}
                    },
                    "zones": {
                        "Counter": {
                            "row": 2, "col": 5,
                            "direction": "E",
                            "tags": ["service", "pass", "orders"]
                        },
                        "Fridge": {
                            "row": 1, "col": 3,
                            "direction": "W",
                            "tags": ["cold", "refrigerated", "fresh"]
                        },
                        "Stove": {
                            "row": 3, "col": 3,
                            "direction": "E",
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
                        "PICKUP_GLASSWARE": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_MICROSCOPES": {"row": 2, "col": 1, "pick_direction": "N"},
                        "PICKUP_SAMPLES": {"row": 1, "col": 8, "pick_direction": "N"}
                    },
                    "zones": {
                        "Trash": {
                            "row": 8, "col": 5,
                            "direction": "S",
                            "tags": ["waste", "disposal", "hazardous"]
                        },
                        "CleanRoom": {
                            "row": 4, "col": 4,
                            "direction": "E",
                            "tags": ["sterile", "clean", "contamination-free"]
                        },
                        "Chemicals": {
                            "row": 4, "col": 6,
                            "direction": "W",
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
                        "PICKUP_ELECTRONICS_PHONES": {"row": 2, "col": 7, "pick_direction": "N"},
                        "PICKUP_APPAREL_TOPS": {"row": 1, "col": 2, "pick_direction": "N"},
                        "PICKUP_TOYS": {"row": 6, "col": 4, "pick_direction": "N"}
                    },
                    "zones": {
                        "Help": {
                            "row": 4, "col": 4,
                            "direction": "E",
                            "tags": ["service", "help", "information"]
                        },
                        "Fitting": {
                            "row": 7, "col": 1,
                            "direction": "W",
                            "tags": ["fitting", "changing", "rooms"]
                        },
                        "Furniture": {
                            "row": 6, "col": 6,
                            "direction": "N",
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
                        "PICKUP_SECURITY_ITEMS": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_TICKETING": {"row": 1, "col": 7, "pick_direction": "N"},
                        "PICKUP_MAPS_INFO": {"row": 2, "col": 1, "pick_direction": "N"}
                    },
                    "zones": {
                        "Gate": {
                            "row": 7, "col": 7,
                            "direction": "E",
                            "tags": ["gate", "boarding", "departure"]
                        },
                        "Baggage": {
                            "row": 5, "col": 0,
                            "direction": "W",
                            "tags": ["baggage", "arrival", "luggage"]
                        },
                        "Shopping": {
                            "row": 4, "col": 4,
                            "direction": "N",
                            "tags": ["shopping", "duty-free", "luxury"]
                        },
                        "Food": {
                            "row": 8, "col": 4,
                            "direction": "S",
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
                        "PICKUP_HANDTOOLS": {"row": 1, "col": 1, "pick_direction": "N"},
                        "PICKUP_LUMBER": {"row": 2, "col": 3, "pick_direction": "N"},
                        "PICKUP_SAFETY_HELMETS": {"row": 1, "col": 8, "pick_direction": "N"}
                    },
                    "zones": {
                        "Building": {
                            "row": 7, "col": 5,
                            "direction": "E",
                            "tags": ["construction", "work", "building"]
                        },
                        "Danger": {
                            "row": 4, "col": 7,
                            "direction": "W",
                            "tags": ["dangerous", "hazardous", "caution"]
                        },
                        "Safe": {
                            "row": 6, "col": 2,
                            "direction": "S",
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
            },
            "11": {
                "name": "Maze Challenge",
                "description": "Complex maze with walls requiring careful navigation to reach pickup and drop-off locations",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 3, "col": 7, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                        "wall_cells": [
                            # Cells that are walls - robot cannot enter these
                            # Format: [row, col] - this cell is a wall
                            
                            # Outer walls (border)
                            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                            [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                            [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                            [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],
                            
                            # Inner maze walls - create accessible paths
                            # Top section
                            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                            
                            # Middle section - create maze pattern
                            [2, 1], [2, 3], [2, 5], [2, 7],
                            [3, 1], [3, 3], [3, 5], [3, 6],  # Leave (3,7) accessible for pickup
                            [4, 1], [4, 5], [4, 6], [4, 7],  # Leave (4,3) accessible for robot start
                            [5, 1], [5, 3], [5, 5], [5, 6], [5, 7],
                            [6, 1], [6, 3], [6, 5], [6, 6], [6, 7],
                            [7, 1], [7, 3], [7, 5], [7, 6], [7, 7],
                            
                            # Bottom section - leave (8,1) accessible for dropoff
                            # Note: (8,1) is NOT in wall_cells, so it's accessible
                            # Also leaving (8,2) to (8,8) accessible for easier navigation
                        ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
                    }
                }
            },
            "12": {
                "name": "Advanced Maze Challenge",
                "description": "Complex maze with multiple dead ends and longer paths requiring strategic navigation",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 1, "col": 8, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                    "wall_cells": [
                        # Cells that are walls - robot cannot enter these
                        # Format: [row, col] - this cell is a wall

                        # Outer walls (border)
                        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                        [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],

                        # Create a simple but challenging maze with adjacent cell movement
                        # Robot starts at (4,4), pickup at (1,8), dropoff at (8,1)

                        # Add some obstacles to create a more interesting path
                        [2, 2], [2, 3], [2, 4], [2, 6], [2, 7],  # Block direct path in row 2
                        [3, 2], [3, 3], [3, 4], [3, 6], [3, 7],  # Block direct path in row 3
                        [4, 1],
                        [5, 3], [5, 4], [5, 5], [5, 6], [5, 7],  # Block direct path in row 5
                        [6, 2], [6, 3], [6, 4], [6, 6], [6, 7],  # Block direct path in row 6
                        [7, 2], [7, 3], [7, 4], [7, 6], [7, 7], [7, 8],  # Block direct path in row 7

                        # Create alternative paths and dead ends based on user's drawing
                        # Row 1: walls at cols 1-7 (red crosses should be walls)
                        [1, 1], [1, 2], [1, 3], [1, 4],
                        # Row 8: open at cols 2-8 (green circles should be open - remove these walls)
                        # Note: (8,1) should remain accessible for dropoff
                    ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
                    }
                }
            },
            "13": {
                "name": "Advanced Maze Challenge",
                "description": "Complex maze with multiple dead ends and longer paths requiring strategic navigation",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 1, "col": 8, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                    "wall_cells": [
                        # Cells that are walls - robot cannot enter these
                        # Format: [row, col] - this cell is a wall

                        # Outer walls (border)
                        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                        [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],

                        # Create a simple but challenging maze with adjacent cell movement
                        # Robot starts at (4,4), pickup at (1,8), dropoff at (8,1)

                        # Add some obstacles to create a more interesting path
                        [1, 4],
                        [2, 2], [2, 4], [2, 6], [2, 7], [2, 8], # Block direct path in row 2
                        [3, 2], [3, 4], [3, 6], [3, 7],  # Block direct path in row 3
                        [4, 2],
                        [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7],  # Block direct path in row 5
                        [6, 6], [6, 7],  # Block direct path in row 6
                        [7, 1], [7, 2], [7, 3], [7, 4], [7, 6], [7, 7], [7, 8],  # Block direct path in row 7

                    ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
                    }
                }
            },
            "14": {
                "name": "Multi-path Maze",
                "description": "A maze with multiple paths and dead ends, requiring exploration.",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 2, "col": 7, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                    "wall_cells": [
                        # Cells that are walls - robot cannot enter these
                        # Format: [row, col] - this cell is a wall

                        # Outer walls (border)
                        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                        [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],

                        # Add some obstacles to create a more interesting path
                        [1, 1], [1, 2], [1, 3], [1, 8], 
                        [2, 1], [2, 2], [2, 3], [2, 6], 
                        [3, 6], [3, 7],  
                        [4, 2], [4, 3],
                        [5, 2], 
                        [6, 5], [6, 7], [6, 8],  
                        [7, 2], [7, 4], [7, 5], [7, 7], [7, 8], 
                        [8, 7], [8, 8], 
                    ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
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
        return {}
    
    def get_world_info(self, world_id):
        """Get name and description for a specific world."""
        if world_id in self.worlds:
            return self.worlds[world_id]["name"], self.worlds[world_id]["description"]
        return None, None
    
    def get_initial_state(self, world_id):
        """Get initial state (robot position, objects) for a specific world."""
        if world_id not in self.worlds:
            return None
            
        # Default initial state for all worlds
        initial_state = {
            "robot_row": 4,
            "robot_col": 4, 
            "robot_facing": "N",
            "robot_state": "stand",
            "current_action": "idle",
            "has_object": False,
            "carried_object_name": None,
            "arm_status": "stowed",
            "gripper_status": "closed",
            "objects": {}
        }
        
        # World-specific object configurations
        if world_id == "1":  # Simple Pick & Drop
            initial_state["objects"] = {
                "tomato_can": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "apple": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "bottle": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "2":  # Grocery Store
            initial_state["objects"] = {
                "water_bottle": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "soda_can": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "juice_box": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "3":  # Warehouse
            initial_state["objects"] = {
                "package": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "box": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "crate": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "4":  # Office Building
            initial_state["objects"] = {
                "document": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "folder": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "file": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "5":  # Hospital
            initial_state["objects"] = {
                "medicine": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "supplies": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "equipment": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "6":  # Restaurant Kitchen
            initial_state["objects"] = {
                "ingredient": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "utensil": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "container": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "7":  # Laboratory
            initial_state["objects"] = {
                "sample": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "specimen": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "vial": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "8":  # Retail Store
            initial_state["objects"] = {
                "product": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "item": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "merchandise": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "9":  # Airport Terminal
            initial_state["objects"] = {
                "luggage": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "bag": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "suitcase": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "10":  # Construction Site
            initial_state["objects"] = {
                "tool": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "material": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "equipment": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "11":  # Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 2, "col": 7, "present": True, "color": "#f85149"}
            }
        elif world_id == "12":  # Advanced Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 1, "col": 8, "present": True, "color": "#f85149"}
            }
        elif world_id == "13":  # Advanced Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 1, "col": 8, "present": True, "color": "#f85149"}
            }
        elif world_id == "14":  # Multi-path Maze
            initial_state["objects"] = {
                "maze_item": {"row": 2, "col": 7, "present": True, "color": "#f85149"}
            }
        else:
            # Default objects for unknown worlds
            initial_state["objects"] = {
                "object1": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "object2": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "object3": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        
        return initial_state
    
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