#!/usr/bin/env python3
"""
Standalone World Builder - Interactive World Creation Tool
A simplified world builder that allows simultaneous editing of all world elements.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import os
import re
from pathlib import Path


class StandaloneWorldBuilder:
    """Standalone world builder with simultaneous editing capabilities."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("World Builder")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Fix mouse cursor visibility in XLaunch
        self.root.configure(cursor="arrow")
        self.root.option_add('*cursor', 'arrow')
        
        # World state
        self.grid_rows = 10
        self.grid_cols = 10
        self.obstacle_cells = []
        self.pickup_locations = {}  # {name: {row, col, direction, objects}}
        self.dropoff_locations = {}  # {name: {row, col, direction, tags}}
        self.checkpoint_locations = {}  # {name: {row, col, direction}}
        self.robot_start_position = {"row": 4, "col": 4, "facing": "N"}  # Default robot start
        self.current_mode = "wall"  # wall, pickup, dropoff, checkpoint, robot_start
        self.editing_location = None  # For editing existing locations
        
        # Drag state for continuous drawing
        self.dragging = False
        self.last_drag_cell = None
        
        # Object categories for pickup locations
        self.object_categories = self._get_object_categories()
        
        # UI components
        self.canvas = None
        self.cell_size = 40
        self.hovered_cell = None
        
        # Create UI
        self._create_ui()
        self._initialize_world()
        self._refresh_world_list()
        self._draw_grid()
    
    def _get_object_categories(self):
        """Get object categories for pickup locations."""
        return {
            "PICKUP_BEVERAGES": ["water_bottle", "soda_can", "juice_box"],
            "PICKUP_PRODUCE": ["apple", "banana", "orange"],
            "PICKUP_DAIRY": ["milk_carton", "cheddar_block", "yogurt_cup"],
            "PICKUP_INCOMING": ["pallet", "shipping_box", "barcode_label"],
            "PICKUP_ELECTRONICS": ["circuit_board", "power_supply", "hdmi_cable"],
            "PICKUP_TEXTILES": ["fabric_roll", "cotton_bale", "yarn_spool"],
            "PICKUP_TOOLS": ["wrench", "screwdriver", "pliers"],
            "PICKUP_STATIONERY": ["pen", "pencil", "marker"],
            "PICKUP_OFFICE_SUPPLIES": ["stapler", "paper_clips", "folder"],
            "PICKUP_COMPUTERS": ["laptop", "keyboard", "mouse"],
            "PICKUP_COFFEE": ["coffee_cup", "espresso_pod", "sugar_packet"],
            "PICKUP_MEDICATIONS": ["pill_bottle", "syringe", "bandage"],
            "PICKUP_PPE": ["gloves_box", "face_mask", "sanitizer_bottle"],
            "PICKUP_EMERGENCY_SUPPLIES": ["first_aid_kit", "defibrillator", "oxygen_tank"],
            "PICKUP_INGREDIENTS": ["flour_bag", "tomatoes", "spices_jar"],
            "PICKUP_KNIVES": ["chef_knife", "paring_knife", "honing_rod"],
            "PICKUP_UTENSILS": ["fork", "spoon", "tongs"],
            "PICKUP_GLASSWARE": ["test_tube", "beaker", "vial"],
            "PICKUP_MICROSCOPES": ["microscope", "glass_slide", "cover_slip"],
            "PICKUP_SAMPLES": ["blood_sample", "tissue_sample", "culture_dish"],
            "PICKUP_ELECTRONICS_PHONES": ["smartphone", "charger", "earbuds"],
            "PICKUP_APPAREL_TOPS": ["shirt", "t_shirt", "jacket"],
            "PICKUP_TOYS": ["toy_car", "puzzle_box", "plush_bear"],
            "PICKUP_SECURITY_ITEMS": ["security_bin", "tray", "belt_bucket"],
            "PICKUP_TICKETING": ["boarding_pass", "luggage_tag", "passport"],
            "PICKUP_MAPS_INFO": ["terminal_map", "brochure", "guide"],
            "PICKUP_HANDTOOLS": ["hammer", "tape_measure", "chisel"],
            "PICKUP_LUMBER": ["wood_plank", "timber_beam", "plywood_sheet"],
            "PICKUP_SAFETY_HELMETS": ["hard_hat", "safety_vest", "ear_protectors"],
            "PICKUP_MAZE_ITEM": ["maze_package"],
            "WHITE_BOX": ["tomato_can"],
        }
    
    def _create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        toolbar = tk.Frame(main_frame, bg='#161b22', height=60)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        toolbar.pack_propagate(False)
        
        # Grid dimensions
        dim_frame = tk.Frame(toolbar, bg='#161b22')
        dim_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(dim_frame, text="Grid:", bg='#161b22', fg='#f0f6fc').pack(side=tk.LEFT)
        
        self.rows_var = tk.StringVar(value=str(self.grid_rows))
        rows_entry = tk.Entry(dim_frame, textvariable=self.rows_var, width=5)
        rows_entry.pack(side=tk.LEFT, padx=(5, 2))
        
        tk.Label(dim_frame, text="x", bg='#161b22', fg='#f0f6fc').pack(side=tk.LEFT)
        
        self.cols_var = tk.StringVar(value=str(self.grid_cols))
        cols_entry = tk.Entry(dim_frame, textvariable=self.cols_var, width=5)
        cols_entry.pack(side=tk.LEFT, padx=(2, 5))
        
        tk.Button(dim_frame, text="Update Grid", command=self._update_grid,
                 bg='#007AFF', fg='#f0f6fc', font=("Arial", 9, "bold"),
                 relief='flat', padx=10, pady=2).pack(side=tk.LEFT, padx=(5, 0))
        
        tk.Button(dim_frame, text="Refresh Grid", command=self._refresh_grid,
                 bg='#ff6b6b', fg='#f0f6fc', font=("Arial", 9, "bold"),
                 relief='flat', padx=10, pady=2).pack(side=tk.LEFT, padx=(5, 0))
        
        # Mode selector
        mode_frame = tk.Frame(toolbar, bg='#161b22')
        mode_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(mode_frame, text="Mode:", bg='#161b22', fg='#f0f6fc').pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="wall")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                 values=["wall", "pickup", "dropoff", "checkpoint", "robot_start"], state='readonly', width=10)
        mode_combo.pack(side=tk.LEFT, padx=(5, 0))
        mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change)
        
        # File operations
        file_frame = tk.Frame(toolbar, bg='#161b22')
        file_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # World selection dropdown
        world_select_frame = tk.Frame(file_frame, bg='#161b22')
        world_select_frame.pack(side=tk.TOP, pady=(0, 5))
        
        tk.Label(world_select_frame, text="World:", bg='#161b22', fg='#f0f6fc', 
                font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.world_var = tk.StringVar()
        self.world_combo = ttk.Combobox(world_select_frame, textvariable=self.world_var, 
                                       state='readonly', width=20)
        self.world_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.world_combo.bind('<<ComboboxSelected>>', self._on_world_select)
        
        # Action buttons
        action_frame = tk.Frame(file_frame, bg='#161b22')
        action_frame.pack(side=tk.TOP)
        
        tk.Button(action_frame, text="New", command=self._new_world,
                 bg='#39d353', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 3))
        
        tk.Button(action_frame, text="Load", command=self._load_selected_world,
                 bg='#58a6ff', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 3))
        
        tk.Button(action_frame, text="Edit", command=self._edit_selected_world,
                 bg='#28a745', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 3))
        
        tk.Button(action_frame, text="Save", command=self._save_world,
                 bg='#ffa657', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 3))
        
        tk.Button(action_frame, text="Delete", command=self._delete_selected_world,
                 bg='#dc3545', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 3))
        
        tk.Button(action_frame, text="Refresh", command=self._refresh_world_list,
                 bg='#6c757d', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(3, 0))
        
        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg='#1a1a1a')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='#0d1117', width=800, height=600)
        
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg='#161b22', height=30)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready", bg='#161b22', fg='#8b949e')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Right panel for location management
        self._create_location_panel(main_frame)
    
    def _create_location_panel(self, parent):
        """Create right panel for managing pickup/dropoff locations."""
        panel = tk.Frame(parent, bg='#161b22', width=300)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        panel.pack_propagate(False)
        
        # Title
        tk.Label(panel, text="Location Manager", font=("Arial", 12, "bold"),
                bg='#161b22', fg='#f0f6fc').pack(pady=(20, 10))
        
        # Pickup locations
        pickup_frame = tk.LabelFrame(panel, text="Pickup Locations", bg='#161b22', fg='#ffa657',
                                    font=("Arial", 10, "bold"))
        pickup_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.pickup_listbox = tk.Listbox(pickup_frame, height=6, bg='#2d2d2d', fg='#f0f6fc',
                                        selectbackground='#ffa657')
        self.pickup_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.pickup_listbox.bind('<Double-1>', self._edit_pickup)
        
        pickup_buttons = tk.Frame(pickup_frame, bg='#161b22')
        pickup_buttons.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        tk.Button(pickup_buttons, text="Edit", command=self._edit_pickup,
                 bg='#ffa657', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(pickup_buttons, text="Delete", command=self._delete_pickup,
                 bg='#f85149', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT)
        
        # Dropoff locations
        dropoff_frame = tk.LabelFrame(panel, text="Dropoff Locations", bg='#161b22', fg='#00d4aa',
                                     font=("Arial", 10, "bold"))
        dropoff_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.dropoff_listbox = tk.Listbox(dropoff_frame, height=6, bg='#2d2d2d', fg='#f0f6fc',
                                          selectbackground='#00d4aa')
        self.dropoff_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.dropoff_listbox.bind('<Double-1>', self._edit_dropoff)
        
        dropoff_buttons = tk.Frame(dropoff_frame, bg='#161b22')
        dropoff_buttons.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        tk.Button(dropoff_buttons, text="Edit", command=self._edit_dropoff,
                 bg='#00d4aa', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(dropoff_buttons, text="Delete", command=self._delete_dropoff,
                 bg='#f85149', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT)
        
        # Checkpoint Locations
        checkpoint_frame = tk.LabelFrame(panel, text="Checkpoint Locations", bg='#161b22', fg='#8b5cf6',
                                        font=("Arial", 10, "bold"))
        checkpoint_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.checkpoint_listbox = tk.Listbox(checkpoint_frame, height=6, bg='#2d2d2d', fg='#f0f6fc',
                                            selectbackground='#8b5cf6')
        self.checkpoint_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.checkpoint_listbox.bind('<Double-1>', self._edit_checkpoint)
        
        checkpoint_buttons = tk.Frame(checkpoint_frame, bg='#161b22')
        checkpoint_buttons.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        tk.Button(checkpoint_buttons, text="Edit", command=self._edit_checkpoint,
                 bg='#8b5cf6', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(checkpoint_buttons, text="Delete", command=self._delete_checkpoint,
                 bg='#f85149', fg='#f0f6fc', font=("Arial", 8, "bold"),
                 relief='flat', padx=8, pady=2).pack(side=tk.LEFT)
        
        # Instructions
        instructions = tk.Text(panel, height=8, bg='#2d2d2d', fg='#8b949e',
                              font=("Arial", 9), wrap=tk.WORD)
        instructions.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 20))
        
        instructions.insert(tk.END, """Instructions:

1. Set grid dimensions and click 'Update Grid'
2. Select mode: wall, pickup, dropoff, or checkpoint
3. Click cells to add/remove elements
4. Double-click locations in lists to edit
5. Use File menu to save/load worlds

Wall Mode: Click to toggle walls
Pickup Mode: Click empty cells to add pickup locations
Dropoff Mode: Click empty cells to add dropoff locations
Checkpoint Mode: Click empty cells to add sequential checkpoints (A, B, C, ...)""")
        
        instructions.config(state=tk.DISABLED)
    
    def _initialize_world(self):
        """Initialize a new world."""
        self.obstacle_cells = []
        self.pickup_locations = {}
        self.dropoff_locations = {}
        self.checkpoint_locations = {}
    
    def _update_grid(self):
        """Update grid dimensions while preserving existing elements."""
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get())
            
            if new_rows < 5 or new_rows > 20 or new_cols < 5 or new_cols > 20:
                messagebox.showerror("Invalid Dimensions", "Dimensions must be between 5 and 20.")
                return
            
            old_rows = self.grid_rows
            old_cols = self.grid_cols
            
            self.grid_rows = new_rows
            self.grid_cols = new_cols
            
            # Remove elements that fall outside the new grid boundaries
            self._remove_outside_elements(old_rows, old_cols)
            
            self._draw_grid()
            self._update_location_lists()
            
            self.status_label.config(text=f"Grid updated to {new_rows}x{new_cols} - elements preserved")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for dimensions.")
    
    def _refresh_grid(self):
        """Clear all elements and create a fresh grid."""
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get())
            
            if new_rows < 5 or new_rows > 20 or new_cols < 5 or new_cols > 20:
                messagebox.showerror("Invalid Dimensions", "Dimensions must be between 5 and 20.")
                return
            
            self.grid_rows = new_rows
            self.grid_cols = new_cols
            
            # Clear all existing data
            self.obstacle_cells = []
            self.pickup_locations = {}
            self.dropoff_locations = {}
            self.checkpoint_locations = {}
            
            # Reinitialize with new dimensions
            self._initialize_world()
            self._draw_grid()
            self._update_location_lists()
            
            self.status_label.config(text=f"Grid refreshed to {new_rows}x{new_cols} - all elements cleared")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for dimensions.")
    
    def _remove_outside_elements(self, old_rows, old_cols):
        """Remove elements that fall outside the new grid boundaries."""
        # Remove walls outside new boundaries
        self.obstacle_cells = [cell for cell in self.obstacle_cells 
                          if 0 <= cell[0] < self.grid_rows and 0 <= cell[1] < self.grid_cols]
        
        # Remove pickups outside new boundaries
        pickups_to_remove = []
        for name, data in self.pickup_locations.items():
            if data['row'] >= self.grid_rows or data['col'] >= self.grid_cols:
                pickups_to_remove.append(name)
        for name in pickups_to_remove:
            del self.pickup_locations[name]
        
        # Remove dropoffs outside new boundaries
        dropoffs_to_remove = []
        for name, data in self.dropoff_locations.items():
            if data['row'] >= self.grid_rows or data['col'] >= self.grid_cols:
                dropoffs_to_remove.append(name)
        for name in dropoffs_to_remove:
            del self.dropoff_locations[name]
        
        # Remove checkpoints outside new boundaries
        checkpoints_to_remove = []
        for name, data in self.checkpoint_locations.items():
            if data['row'] >= self.grid_rows or data['col'] >= self.grid_cols:
                checkpoints_to_remove.append(name)
        for name in checkpoints_to_remove:
            del self.checkpoint_locations[name]
    
    def _on_mode_change(self, event=None):
        """Handle mode change."""
        self.current_mode = self.mode_var.get()
        self.status_label.config(text=f"Mode: {self.current_mode}")
    
    def _draw_grid(self):
        """Draw the interactive grid."""
        self.canvas.delete("all")
        
        # Calculate grid dimensions
        grid_width = self.grid_cols * self.cell_size
        grid_height = self.grid_rows * self.cell_size
        
        # Set scroll region
        self.canvas.configure(scrollregion=(0, 0, grid_width + 100, grid_height + 100))
        
        # Draw cells
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Determine cell type and color
                cell_type = self._get_cell_type(row, col)
                fill_color, outline_color = self._get_cell_colors(cell_type)
                
                # Draw cell background
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=fill_color, outline=outline_color, width=1,
                                           tags=f"cell_{row}_{col}")
                
                # Draw cell content based on type
                if cell_type == "wall":
                    self._draw_wall_cell(x1, y1, x2, y2)
                elif cell_type == "pickup":
                    self._draw_pickup_cell(x1, y1, x2, y2, row, col)
                elif cell_type == "dropoff":
                    self._draw_dropoff_cell(x1, y1, x2, y2, row, col)
                elif cell_type == "checkpoint":
                    self._draw_checkpoint_cell(x1, y1, x2, y2, row, col)
                elif cell_type == "robot_start":
                    self._draw_robot_start_cell(x1, y1, x2, y2, row, col)
        
        # Draw robot extensions after cells, before grid lines
        self._draw_robot_extensions()

        # Draw grid lines
        for i in range(self.grid_rows + 1):
            y = i * self.cell_size
            self.canvas.create_line(0, y, grid_width, y, 
                                  fill="#ffffff", width=1, tags="grid")
        
        for i in range(self.grid_cols + 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, grid_height, 
                                  fill="#ffffff", width=1, tags="grid")
        
        # Draw coordinate labels
        for row in range(self.grid_rows):
            x = -20
            y = row * self.cell_size + self.cell_size // 2
            self.canvas.create_text(x, y, text=str(row), fill="#58a6ff", 
                                  font=("Arial", 10, "bold"), tags="labels")
        
        for col in range(self.grid_cols):
            x = col * self.cell_size + self.cell_size // 2
            y = -20
            self.canvas.create_text(x, y, text=str(col), fill="#58a6ff", 
                                  font=("Arial", 10, "bold"), tags="labels")
    
    def _get_cell_type(self, row, col):
        """Determine the type of cell at given coordinates."""
        # Check if it's a wall
        if [row, col] in self.obstacle_cells:
            return "wall"
        
        # Check if it's a pickup location
        for name, data in self.pickup_locations.items():
            if data['row'] == row and data['col'] == col:
                return "pickup"
        
        # Check if it's a dropoff location
        for name, data in self.dropoff_locations.items():
            if data['row'] == row and data['col'] == col:
                return "dropoff"
        
        # Check if it's a checkpoint location
        for name, data in self.checkpoint_locations.items():
            if data['row'] == row and data['col'] == col:
                return "checkpoint"
        
        # Check if it's robot start position (robot is 1 cell wide, centered)
        robot_row = self.robot_start_position['row']
        robot_col = self.robot_start_position['col']
        if robot_row == row and robot_col == col:
            return "robot_start"
        
        return "empty"
    
    def _get_cell_colors(self, cell_type):
        """Get fill and outline colors for cell type."""
        colors = {
            "empty": ("#2a2a2a", "#444444"),
            "wall": ("#444444", "#666666"),
            "pickup": ("#ffa657", "#ff7b72"),
            "dropoff": ("#00d4aa", "#39d353"),
            "checkpoint": ("#8b5cf6", "#a78bfa"),
            "robot_start": ("#ff6b6b", "#ff5252")
        }
        return colors.get(cell_type, ("#2a2a2a", "#444444"))
    
    def _draw_wall_cell(self, x1, y1, x2, y2):
        """Draw wall cell with X pattern."""
        # Draw X pattern
        self.canvas.create_line(x1 + 5, y1 + 5, x2 - 5, y2 - 5, 
                              fill="#666666", width=2, tags="wall")
        self.canvas.create_line(x1 + 5, y2 - 5, x2 - 5, y1 + 5, 
                              fill="#666666", width=2, tags="wall")
    
    def _draw_pickup_cell(self, x1, y1, x2, y2, row, col):
        """Draw pickup cell with label."""
        # Find pickup name
        pickup_name = None
        for name, data in self.pickup_locations.items():
            if data['row'] == row and data['col'] == col:
                pickup_name = name
                break
        
        if pickup_name:
            # Draw pickup square
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            size = self.cell_size // 3
            
            self.canvas.create_rectangle(center_x - size, center_y - size,
                                       center_x + size, center_y + size,
                                       fill="#ffa657", outline="#ff7b72", width=2,
                                       tags="pickup")
            
            # Draw label
            self.canvas.create_text(center_x, y1 - 15, text=pickup_name.replace("PICKUP_", ""),
                                  fill="#ffa657", font=("Arial", 8, "bold"), tags="pickup")
            
            # Draw direction arrow
            direction = self.pickup_locations[pickup_name]['direction']
            self._draw_direction_arrow(center_x, center_y, direction, "#ff7b72")
    
    def _draw_dropoff_cell(self, x1, y1, x2, y2, row, col):
        """Draw dropoff cell with label."""
        # Find dropoff name
        dropoff_name = None
        for name, data in self.dropoff_locations.items():
            if data['row'] == row and data['col'] == col:
                dropoff_name = name
                break
        
        if dropoff_name:
            # Draw dropoff circle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = self.cell_size // 3
            
            self.canvas.create_oval(center_x - radius, center_y - radius,
                                  center_x + radius, center_y + radius,
                                  fill="#2d2d2d", outline="#00d4aa", width=2,
                                  stipple="gray25", tags="dropoff")
            
            # Draw label
            self.canvas.create_text(center_x, y1 - 15, text=dropoff_name,
                                  fill="#00d4aa", font=("Arial", 8, "bold"), tags="dropoff")
            
            # Draw direction arrow
            direction = self.dropoff_locations[dropoff_name]['direction']
            self._draw_direction_arrow(center_x, center_y, direction, "#00d4aa")
    
    def _draw_checkpoint_cell(self, x1, y1, x2, y2, row, col):
        """Draw checkpoint cell with letter label."""
        # Find checkpoint name
        checkpoint_name = None
        for name, data in self.checkpoint_locations.items():
            if data['row'] == row and data['col'] == col:
                checkpoint_name = name
                break
        
        if checkpoint_name:
            # Extract letter from checkpoint name (e.g., "CHECKPOINT_A" -> "A")
            letter = checkpoint_name.split('_')[-1] if '_' in checkpoint_name else checkpoint_name
            
            # Draw checkpoint diamond
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            size = self.cell_size // 3
            
            # Create diamond shape
            points = [
                center_x, center_y - size,  # top
                center_x + size, center_y,  # right
                center_x, center_y + size,  # bottom
                center_x - size, center_y  # left
            ]
            
            self.canvas.create_polygon(points, fill="#2d2d2d", outline="#ffffff", width=2,
                                     stipple="gray25", tags="checkpoint")
            
            # Draw letter label in white
            self.canvas.create_text(center_x, center_y, text=letter,
                                  fill="#ffffff", font=("Arial", 12, "bold"), tags="checkpoint")
            
            # Draw direction arrow in white
            direction = self.checkpoint_locations[checkpoint_name]['direction']
            self._draw_direction_arrow(center_x, center_y, direction, "#ffffff")
    
    def _draw_robot_start_cell(self, x1, y1, x2, y2, row, col):
        """Draw robot start position cell with directional extension."""
        direction = self.robot_start_position['facing']
        
        # Draw main robot cell (fully colored)
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                   fill="#ff6b6b", outline="#ff5252", width=2, tags="robot_start")
        
        # Draw direction arrow at cell center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        self._draw_direction_arrow(center_x, center_y, direction, "#ffffff")
    
    def _draw_robot_extensions(self):
        """Draw robot half-cell overlays after all cells are rendered."""
        row = self.robot_start_position['row']
        col = self.robot_start_position['col']
        direction = self.robot_start_position['facing']
        
        if direction == "N":
            if row > 0:
                self._draw_robot_extension(row - 1, col, "south_half")
            if row < self.grid_rows - 1:
                self._draw_robot_extension(row + 1, col, "north_half")
        elif direction == "S":
            if row > 0:
                self._draw_robot_extension(row - 1, col, "south_half")
            if row < self.grid_rows - 1:
                self._draw_robot_extension(row + 1, col, "north_half")
        elif direction == "E":
            if col > 0:
                self._draw_robot_extension(row, col - 1, "east_half")
            if col < self.grid_cols - 1:
                self._draw_robot_extension(row, col + 1, "west_half")
        elif direction == "W":
            if col > 0:
                self._draw_robot_extension(row, col - 1, "east_half")
            if col < self.grid_cols - 1:
                self._draw_robot_extension(row, col + 1, "west_half")
    
    def _draw_robot_extension(self, row, col, half_side):
        """Draw robot extension on adjacent cell (half cell colored)."""
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        # Calculate half cell coordinates based on which half to color
        if half_side == "north_half":
            # Color top half
            self.canvas.create_rectangle(x1, y1, x2, y1 + self.cell_size // 2,
                                       fill="#ff6b6b", outline="#ff5252", width=1, tags="robot_start")
        elif half_side == "south_half":
            # Color bottom half
            self.canvas.create_rectangle(x1, y1 + self.cell_size // 2, x2, y2,
                                       fill="#ff6b6b", outline="#ff5252", width=1, tags="robot_start")
        elif half_side == "east_half":
            # Color right half
            self.canvas.create_rectangle(x1 + self.cell_size // 2, y1, x2, y2,
                                       fill="#ff6b6b", outline="#ff5252", width=1, tags="robot_start")
        elif half_side == "west_half":
            # Color left half
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size // 2, y2,
                                       fill="#ff6b6b", outline="#ff5252", width=1, tags="robot_start")
    
    def _draw_direction_arrow(self, center_x, center_y, direction, color):
        """Draw direction arrow."""
        arrow_length = self.cell_size // 4
        
        if direction == "N":
            end_x, end_y = center_x, center_y - arrow_length
        elif direction == "S":
            end_x, end_y = center_x, center_y + arrow_length
        elif direction == "E":
            end_x, end_y = center_x + arrow_length, center_y
        elif direction == "W":
            end_x, end_y = center_x - arrow_length, center_y
        else:
            end_x, end_y = center_x, center_y - arrow_length
        
        self.canvas.create_line(center_x, center_y, end_x, end_y,
                              fill=color, width=2, arrow=tk.LAST, tags="arrow")
    
    def _on_canvas_click(self, event):
        """Handle canvas click events - start dragging."""
        # Get clicked cell coordinates
        col = int(event.x // self.cell_size)
        row = int(event.y // self.cell_size)
        
        # Validate coordinates
        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return
        
        # Start dragging
        self.dragging = True
        self.last_drag_cell = (row, col)
        
        # Handle click based on current mode
        if self.current_mode == "wall":
            self._handle_wall_click(row, col)
        elif self.current_mode == "pickup":
            self._handle_pickup_click(row, col)
        elif self.current_mode == "dropoff":
            self._handle_dropoff_click(row, col)
        elif self.current_mode == "checkpoint":
            self._handle_checkpoint_click(row, col)
        elif self.current_mode == "robot_start":
            self._handle_robot_start_click(row, col)
        
        self._draw_grid()
        self._update_location_lists()
    
    def _on_canvas_drag(self, event):
        """Handle canvas drag events - continue drawing while dragging."""
        if not self.dragging:
            return
            
        # Get dragged cell coordinates
        col = int(event.x // self.cell_size)
        row = int(event.y // self.cell_size)
        
        # Validate coordinates
        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return
        
        # Only process if we moved to a different cell
        if (row, col) == self.last_drag_cell:
            return
        
        self.last_drag_cell = (row, col)
        
        # Handle drag based on current mode
        if self.current_mode == "wall":
            self._handle_wall_click(row, col)
        elif self.current_mode == "pickup":
            self._handle_pickup_click(row, col)
        elif self.current_mode == "dropoff":
            self._handle_dropoff_click(row, col)
        elif self.current_mode == "checkpoint":
            self._handle_checkpoint_click(row, col)
        elif self.current_mode == "robot_start":
            self._handle_robot_start_click(row, col)
        
        self._draw_grid()
        self._update_location_lists()
    
    def _on_canvas_release(self, event):
        """Handle canvas release events - stop dragging."""
        self.dragging = False
        self.last_drag_cell = None
    
    def _handle_wall_click(self, row, col):
        """Handle wall cell click."""
        cell_pos = [row, col]
        
        if cell_pos in self.obstacle_cells:
            self.obstacle_cells.remove(cell_pos)
        else:
            # Add wall
            self.obstacle_cells.append(cell_pos)
    
    def _handle_pickup_click(self, row, col):
        """Handle pickup location click."""
        # Check if cell is available
        if self._is_cell_available(row, col):
            self._show_pickup_dialog(row, col)
        else:
            # Check if there's an existing pickup to edit
            pickup_name = self._get_pickup_at_location(row, col)
            if pickup_name:
                self._show_pickup_dialog(row, col, pickup_name)
    
    def _handle_dropoff_click(self, row, col):
        """Handle dropoff location click."""
        # Check if cell is available
        if self._is_cell_available(row, col):
            self._show_dropoff_dialog(row, col)
        else:
            # Check if there's an existing dropoff to edit
            dropoff_name = self._get_dropoff_at_location(row, col)
            if dropoff_name:
                self._show_dropoff_dialog(row, col, dropoff_name)
    
    def _handle_checkpoint_click(self, row, col):
        """Handle checkpoint location click - direct add without dialog."""
        # For checkpoints, only add on initial click, not during drag
        if self.dragging and self.last_drag_cell != (row, col):
            return
            
        # First check if there's an existing checkpoint to remove
        checkpoint_name = self._get_checkpoint_at_location(row, col)
        if checkpoint_name:
            # Remove existing checkpoint and renumber remaining ones
            self._remove_and_renumber_checkpoints(checkpoint_name)
            self._draw_grid()
            self._update_location_lists()
        else:
            # Check if cell is available for new checkpoint
            if self._is_cell_available_for_checkpoint(row, col):
                # Directly add checkpoint with next letter
                next_letter = self._get_next_checkpoint_letter()
                checkpoint_name = f"CHECKPOINT_{next_letter}"
                
                self.checkpoint_locations[checkpoint_name] = {
                    'row': row,
                    'col': col,
                    'direction': 'N'  # Default direction
                }
                self._draw_grid()
                self._update_location_lists()
    
    def _handle_robot_start_click(self, row, col):
        """Handle robot start position click with rotation support."""
        robot_row = self.robot_start_position['row']
        robot_col = self.robot_start_position['col']
        
        # Check if clicking on existing robot position
        if robot_row == row and robot_col == col:
            # Rotate robot direction
            current_direction = self.robot_start_position['facing']
            directions = ["N", "E", "S", "W"]
            current_index = directions.index(current_direction)
            next_index = (current_index + 1) % len(directions)
            new_direction = directions[next_index]
            
            # Always allow rotation (collision check only for placement, not rotation)
            self.robot_start_position['facing'] = new_direction
            self._draw_grid()
            self.status_label.config(text=f"Robot rotated to face {new_direction}")
        else:
            # Place robot at new position
            if self._can_place_robot_at(row, col):
                self.robot_start_position = {"row": row, "col": col, "facing": "N"}
                self._draw_grid()
                self.status_label.config(text=f"Robot start position set to ({row}, {col})")
            else:
                self.status_label.config(text="Cannot place robot here - not enough space or blocked")
    
    def _can_place_robot_at(self, row, col):
        """Check if robot can be placed at given position with directional extension."""
        # Check if the main cell is not a wall
        if [row, col] in self.obstacle_cells:
            return False
        
        # Check if the main cell is not occupied by other elements
        for name, data in self.pickup_locations.items():
            if data['row'] == row and data['col'] == col:
                return False
        
        for name, data in self.dropoff_locations.items():
            if data['row'] == row and data['col'] == col:
                return False
        
        for name, data in self.checkpoint_locations.items():
            if data['row'] == row and data['col'] == col:
                return False
        
        # Check if robot can extend in current direction
        direction = self.robot_start_position['facing']
        if not self._can_robot_extend_in_direction(row, col, direction):
            return False
        
        return True
    
    def _can_robot_extend_in_direction(self, row, col, direction):
        """Check if robot can extend in given direction."""
        if direction == "N":
            # Check if cell above exists and is not blocked
            if row > 0:
                return not ([row-1, col] in self.obstacle_cells)
        elif direction == "S":
            # Check if cell below exists and is not blocked
            if row < self.grid_rows - 1:
                return not ([row+1, col] in self.obstacle_cells)
        elif direction == "E":
            # Check if cell to the right exists and is not blocked
            if col < self.grid_cols - 1:
                return not ([row, col+1] in self.obstacle_cells)
        elif direction == "W":
            # Check if cell to the left exists and is not blocked
            if col > 0:
                return not ([row, col-1] in self.obstacle_cells)
        
        return True
    
    def _remove_and_renumber_checkpoints(self, checkpoint_to_remove):
        """Remove a checkpoint and renumber remaining checkpoints to maintain sequence."""
        # Remove the checkpoint
        del self.checkpoint_locations[checkpoint_to_remove]
        
        # Get all remaining checkpoints sorted by their current letter
        remaining_checkpoints = []
        for name, data in self.checkpoint_locations.items():
            if name.startswith('CHECKPOINT_'):
                letter = name.split('_')[1]
                if len(letter) == 1 and letter.isalpha():
                    remaining_checkpoints.append((letter, name, data))
        
        # Sort by letter
        remaining_checkpoints.sort(key=lambda x: x[0])
        
        # Renumber them sequentially starting from A
        new_checkpoints = {}
        for i, (old_letter, old_name, data) in enumerate(remaining_checkpoints):
            new_letter = chr(ord('A') + i)
            new_name = f"CHECKPOINT_{new_letter}"
            new_checkpoints[new_name] = data
        
        # Replace the checkpoint locations with renumbered ones
        self.checkpoint_locations = new_checkpoints
    
    def _is_cell_available(self, row, col):
        """Check if cell is available for pickup/dropoff."""
        # Cannot place on walls
        if [row, col] in self.obstacle_cells:
            return False
        
        # Cannot place on robot (robot is 1 cell wide, centered)
        robot_row = self.robot_start_position['row']
        robot_col = self.robot_start_position['col']
        if robot_row == row and robot_col == col:
            return False
        
        # Cannot place on existing pickup
        for data in self.pickup_locations.values():
            if data['row'] == row and data['col'] == col:
                return False
        
        # Cannot place on existing dropoff
        for data in self.dropoff_locations.values():
            if data['row'] == row and data['col'] == col:
                return False
        
        return True
    
    def _is_cell_available_for_checkpoint(self, row, col):
        """Check if cell is available for checkpoint (can overlap with other checkpoints)."""
        # Cannot place on walls
        if [row, col] in self.obstacle_cells:
            return False
        
        # Cannot place on robot (robot is 1 cell wide, centered)
        robot_row = self.robot_start_position['row']
        robot_col = self.robot_start_position['col']
        if robot_row == row and robot_col == col:
            return False
        
        # Cannot place on existing pickup
        for data in self.pickup_locations.values():
            if data['row'] == row and data['col'] == col:
                return False
        
        # Cannot place on existing dropoff
        for data in self.dropoff_locations.values():
            if data['row'] == row and data['col'] == col:
                return False
        
        return True
    
    def _get_pickup_at_location(self, row, col):
        """Get pickup name at given location."""
        for name, data in self.pickup_locations.items():
            if data['row'] == row and data['col'] == col:
                return name
        return None
    
    def _get_dropoff_at_location(self, row, col):
        """Get dropoff name at given location."""
        for name, data in self.dropoff_locations.items():
            if data['row'] == row and data['col'] == col:
                return name
        return None
    
    def _get_checkpoint_at_location(self, row, col):
        """Get checkpoint name at given location."""
        for name, data in self.checkpoint_locations.items():
            if data['row'] == row and data['col'] == col:
                return name
        return None
    
    def _get_next_checkpoint_letter(self):
        """Get the next checkpoint letter (A, B, C, ...)."""
        if not self.checkpoint_locations:
            return 'A'
        
        # Find the highest letter used
        letters = []
        for name in self.checkpoint_locations.keys():
            if name.startswith('CHECKPOINT_'):
                letter = name.split('_')[1]
                if len(letter) == 1 and letter.isalpha():
                    letters.append(letter)
        
        if not letters:
            return 'A'
        
        # Return next letter in sequence
        last_letter = max(letters)
        return chr(ord(last_letter) + 1)
    
    def _show_pickup_dialog(self, row, col, existing_name=None):
        """Show pickup location configuration dialog."""
        dialog = PickupLocationDialog(self.root, row, col, existing_name)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            name, direction, objects = dialog.result
            self.pickup_locations[name] = {
                'row': row,
                'col': col,
                'direction': direction,
                'objects': objects
            }
            self._draw_grid()
            self._update_location_lists()
    
    def _show_dropoff_dialog(self, row, col, existing_name=None):
        """Show dropoff location configuration dialog."""
        dialog = DropoffLocationDialog(self.root, row, col, existing_name)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            name, direction, tags = dialog.result
            self.dropoff_locations[name] = {
                'row': row,
                'col': col,
                'direction': direction,
                'tags': tags
            }
            self._draw_grid()
            self._update_location_lists()
    
    
    def _update_location_lists(self):
        """Update the location listboxes."""
        # Update pickup list
        self.pickup_listbox.delete(0, tk.END)
        for name, data in self.pickup_locations.items():
            self.pickup_listbox.insert(tk.END, f"{name} ({data['row']},{data['col']})")
        
        # Update dropoff list
        self.dropoff_listbox.delete(0, tk.END)
        for name, data in self.dropoff_locations.items():
            self.dropoff_listbox.insert(tk.END, f"{name} ({data['row']},{data['col']})")
        
        # Update checkpoint list
        self.checkpoint_listbox.delete(0, tk.END)
        for name, data in self.checkpoint_locations.items():
            self.checkpoint_listbox.insert(tk.END, f"{name} ({data['row']},{data['col']})")
    
    def _edit_pickup(self, event=None):
        """Edit selected pickup location."""
        selection = self.pickup_listbox.curselection()
        if not selection:
            return
        
        pickup_name = list(self.pickup_locations.keys())[selection[0]]
        data = self.pickup_locations[pickup_name]
        self._show_pickup_dialog(data['row'], data['col'], pickup_name)
    
    def _edit_dropoff(self, event=None):
        """Edit selected dropoff location."""
        selection = self.dropoff_listbox.curselection()
        if not selection:
            return
        
        dropoff_name = list(self.dropoff_locations.keys())[selection[0]]
        data = self.dropoff_locations[dropoff_name]
        self._show_dropoff_dialog(data['row'], data['col'], dropoff_name)
    
    def _delete_pickup(self):
        """Delete selected pickup location."""
        selection = self.pickup_listbox.curselection()
        if not selection:
            return
        
        pickup_name = list(self.pickup_locations.keys())[selection[0]]
        del self.pickup_locations[pickup_name]
        self._draw_grid()
        self._update_location_lists()
    
    def _delete_dropoff(self):
        """Delete selected dropoff location."""
        selection = self.dropoff_listbox.curselection()
        if not selection:
            return
        
        dropoff_name = list(self.dropoff_locations.keys())[selection[0]]
        del self.dropoff_locations[dropoff_name]
        self._draw_grid()
        self._update_location_lists()
    
    def _edit_checkpoint(self, event=None):
        """Edit selected checkpoint location - now just removes it."""
        selection = self.checkpoint_listbox.curselection()
        if not selection:
            return
        
        checkpoint_name = list(self.checkpoint_locations.keys())[selection[0]]
        # Remove the checkpoint
        del self.checkpoint_locations[checkpoint_name]
        self._draw_grid()
        self._update_location_lists()
    
    def _delete_checkpoint(self):
        """Delete selected checkpoint location."""
        selection = self.checkpoint_listbox.curselection()
        if not selection:
            return
        
        checkpoint_name = list(self.checkpoint_locations.keys())[selection[0]]
        del self.checkpoint_locations[checkpoint_name]
        self._draw_grid()
        self._update_location_lists()
    
    def _on_canvas_motion(self, event):
        """Handle canvas mouse motion for hover effects."""
        col = int(event.x // self.cell_size)
        row = int(event.y // self.cell_size)
        
        # Validate coordinates
        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            self.hovered_cell = None
            return
        
        self.hovered_cell = (row, col)
    
    def _on_canvas_leave(self, event):
        """Handle canvas leave event."""
        self.hovered_cell = None
    
    def _new_world(self):
        """Create a new world."""
        if messagebox.askyesno("New World", "Create a new world? All current changes will be lost."):
            self._initialize_world()
            self._draw_grid()
            self._update_location_lists()
            self.status_label.config(text="New world created")
    
    def _load_world(self):
        """Load an existing world."""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Load World",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=Path(__file__).parent / "worlds"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    world_data = json.load(f)
                
                # Load world data
                self.grid_rows = world_data.get('grid_dimensions', {}).get('rows', 10)
                self.grid_cols = world_data.get('grid_dimensions', {}).get('cols', 10)
                self.rows_var.set(str(self.grid_rows))
                self.cols_var.set(str(self.grid_cols))
                
                # Load robot start position
                self.robot_start_position = world_data.get('robot_start_position', {"row": 4, "col": 4, "facing": "N"})
                
                config = world_data.get('config', {})
                self.obstacle_cells = config.get('obstacle_cells', [])
                
                # Load pickup locations
                waypoints = config.get('waypoints', {})
                self.pickup_locations = {}
                for name, data in waypoints.items():
                    self.pickup_locations[name] = {
                        'row': data['row'],
                        'col': data['col'],
                        'direction': data.get('pick_direction', 'N'),
                        'objects': data.get('objects', [])
                    }
                
                # Load dropoff locations
                zones = config.get('zones', {})
                self.dropoff_locations = {}
                for name, data in zones.items():
                    self.dropoff_locations[name] = {
                        'row': data['row'],
                        'col': data['col'],
                        'direction': data.get('direction', 'N'),
                        'tags': data.get('tags', [])
                    }
                
                self._draw_grid()
                self._update_location_lists()
                self.status_label.config(text=f"Loaded world: {world_data.get('name', 'Unknown')}")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load world: {str(e)}")
    
    def _delete_world(self):
        """Delete an existing world."""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Delete World",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=Path(__file__).parent / "worlds"
        )
        
        if file_path:
            try:
                # Load world data to get name
                with open(file_path, 'r') as f:
                    world_data = json.load(f)
                
                world_name = world_data.get('name', 'Unknown')
                
                # Confirm deletion
                if messagebox.askyesno("Confirm Delete", 
                                    f"Are you sure you want to delete world '{world_name}'?\n\nThis action cannot be undone."):
                    # Delete the file
                    Path(file_path).unlink()
                    
                    # Reload world manager to remove the deleted world
                    self._reload_world_manager()
                    
                    messagebox.showinfo("Success", f"World '{world_name}' deleted successfully!\n\nIt has been removed from main.py")
                    self.status_label.config(text=f"Deleted world: {world_name}")
                    
            except Exception as e:
                messagebox.showerror("Delete Error", f"Failed to delete world: {str(e)}")
    
    def _save_world(self):
        """Save the current world."""
        # Get world name and description
        name = simpledialog.askstring("Save World", "Enter world name:")
        if not name:
            return
        
        description = simpledialog.askstring("Save World", "Enter world description:")
        if not description:
            return
        
        # No validation required - users can save worlds with any configuration
        
        try:
            # Generate synonyms
            synonyms = self._generate_synonyms()
            
            # Create world configuration
            world_config = {
                "name": name,
                "description": description,
                "is_custom": True,
                "grid_dimensions": {"rows": self.grid_rows, "cols": self.grid_cols},
                "robot_start_position": self.robot_start_position,
                "config": {
                    "waypoints": {},
                    "zones": {},
                    "checkpoints": {},
                    "obstacle_cells": self.obstacle_cells,
                    "synonyms": synonyms
                }
            }
            
            # Add pickup locations
            for name, data in self.pickup_locations.items():
                world_config["config"]["waypoints"][name] = {
                    "row": data["row"],
                    "col": data["col"],
                    "pick_direction": data["direction"],
                    "objects": data["objects"]
                }
            
            # Add dropoff locations
            for name, data in self.dropoff_locations.items():
                world_config["config"]["zones"][name] = {
                    "row": data["row"],
                    "col": data["col"],
                    "direction": data["direction"],
                    "tags": data["tags"]
                }
            
            # Add checkpoint locations
            for name, data in self.checkpoint_locations.items():
                world_config["config"]["checkpoints"][name] = {
                    "row": data["row"],
                    "col": data["col"],
                    "direction": data["direction"]
                }
            
            # Save to file
            filename = self._sanitize_filename(name)
            worlds_dir = Path(__file__).parent / "worlds"
            worlds_dir.mkdir(exist_ok=True)
            filepath = worlds_dir / f"custom_{filename}.json"
            
            with open(filepath, 'w') as f:
                json.dump(world_config, f, indent=2)
            
            # Automatically reload world manager to include the new world
            self._reload_world_manager()
            
            # Refresh the world dropdown
            self._refresh_world_list()
            
            messagebox.showinfo("Success", f"World '{name}' saved successfully!\n\nIt is now available in main.py")
            self.status_label.config(text=f"Saved world: {name}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save world: {str(e)}")
    
    def _reload_world_manager(self):
        """Reload the world manager to include newly saved worlds."""
        try:
            # Import world manager
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from world_manager import WorldManager
            
            # Create a temporary world manager instance to reload custom worlds
            temp_wm = WorldManager()
            temp_wm.load_custom_worlds()
            
            # The world manager will automatically detect the new world file
            # when main.py is restarted or when the world manager is reloaded
            print(f"✓ World manager reloaded - new worlds will be available in main.py")
            
        except Exception as e:
            print(f"Warning: Could not reload world manager: {e}")
            print("New world will be available after restarting main.py")
    
    def _refresh_world_list(self):
        """Refresh the world dropdown list."""
        try:
            worlds_dir = Path(__file__).parent / "worlds"
            if not worlds_dir.exists():
                self.world_combo['values'] = []
                return
            
            world_files = list(worlds_dir.glob("custom_*.json"))
            world_names = []
            
            for file_path in world_files:
                try:
                    with open(file_path, 'r') as f:
                        world_data = json.load(f)
                        name = world_data.get('name', file_path.stem)
                        world_names.append(f"{name} ({file_path.name})")
                except:
                    world_names.append(f"Unknown ({file_path.name})")
            
            self.world_combo['values'] = world_names
            
            if world_names:
                self.world_combo.set(world_names[0])
            
        except Exception as e:
            print(f"Error refreshing world list: {e}")
            self.world_combo['values'] = []
    
    def _on_world_select(self, event=None):
        """Handle world selection from dropdown."""
        selected = self.world_var.get()
        if selected:
            self.status_label.config(text=f"Selected: {selected}")
    
    def _load_selected_world(self):
        """Load the currently selected world."""
        selected = self.world_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a world from the dropdown.")
            return
        
        # Extract filename from selection
        filename = selected.split('(')[-1].rstrip(')')
        
        try:
            worlds_dir = Path(__file__).parent / "worlds"
            filepath = worlds_dir / filename
            
            if not filepath.exists():
                messagebox.showerror("File Not Found", f"World file not found: {filename}")
                return
            
            with open(filepath, 'r') as f:
                world_data = json.load(f)
            
            # Load world data
            self.grid_rows = world_data.get('grid_dimensions', {}).get('rows', 10)
            self.grid_cols = world_data.get('grid_dimensions', {}).get('cols', 10)
            self.rows_var.set(str(self.grid_rows))
            self.cols_var.set(str(self.grid_cols))
            
            # Load robot start position
            self.robot_start_position = world_data.get('robot_start_position', {"row": 4, "col": 4, "facing": "N"})
            
            config = world_data.get('config', {})
            self.obstacle_cells = config.get('obstacle_cells', [])
            
            # Load pickup locations
            waypoints = config.get('waypoints', {})
            self.pickup_locations = {}
            for name, data in waypoints.items():
                self.pickup_locations[name] = {
                    'row': data['row'],
                    'col': data['col'],
                    'direction': data.get('pick_direction', 'N'),
                    'objects': data.get('objects', [])
                }
            
            # Load dropoff locations
            zones = config.get('zones', {})
            self.dropoff_locations = {}
            for name, data in zones.items():
                self.dropoff_locations[name] = {
                    'row': data['row'],
                    'col': data['col'],
                    'direction': data.get('direction', 'N'),
                    'tags': data.get('tags', [])
                }
            
            # Load checkpoint locations
            checkpoints = config.get('checkpoints', {})
            self.checkpoint_locations = {}
            for name, data in checkpoints.items():
                self.checkpoint_locations[name] = {
                    'row': data['row'],
                    'col': data['col'],
                    'direction': data.get('direction', 'N')
                }
            
            self._draw_grid()
            self._update_location_lists()
            self.status_label.config(text=f"Loaded world: {world_data.get('name', 'Unknown')}")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load world: {str(e)}")
    
    def _edit_selected_world(self):
        """Edit the currently selected world (same as load but indicates editing mode)."""
        self._load_selected_world()
        if self.world_var.get():
            self.status_label.config(text=f"Editing: {self.world_var.get()}")
    
    def _delete_selected_world(self):
        """Delete the currently selected world."""
        selected = self.world_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a world from the dropdown.")
            return
        
        # Extract filename and name from selection
        name = selected.split('(')[0].strip()
        filename = selected.split('(')[-1].rstrip(')')
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", 
                            f"Are you sure you want to delete world '{name}'?\n\nThis action cannot be undone."):
            try:
                worlds_dir = Path(__file__).parent / "worlds"
                filepath = worlds_dir / filename
                
                if filepath.exists():
                    filepath.unlink()
                    
                    # Reload world manager to remove the deleted world
                    self._reload_world_manager()
                    
                    # Refresh the dropdown
                    self._refresh_world_list()
                    
                    messagebox.showinfo("Success", f"World '{name}' deleted successfully!\n\nIt has been removed from main.py")
                    self.status_label.config(text=f"Deleted world: {name}")
                else:
                    messagebox.showerror("File Not Found", f"World file not found: {filename}")
                    
            except Exception as e:
                messagebox.showerror("Delete Error", f"Failed to delete world: {str(e)}")
    
    def _generate_synonyms(self):
        """Generate comprehensive synonyms for pickup and dropoff locations."""
        synonyms = {}
        
        # Add pickup synonyms with comprehensive mapping
        for name, data in self.pickup_locations.items():
            # Add lowercase version
            synonyms[name.lower()] = name
            
            # Add object synonyms
            for obj in data["objects"]:
                synonyms[obj.lower()] = name
                synonyms[obj.replace("_", " ").lower()] = name
            
            # Add comprehensive category synonyms
            category_synonyms = self._get_category_synonyms(name)
            for synonym in category_synonyms:
                synonyms[synonym.lower()] = name
            
            # Add word parts
            words = name.replace("PICKUP_", "").replace("_", " ").lower().split()
            for word in words:
                synonyms[word] = name
            if len(words) > 1:
                synonyms[" ".join(words)] = name
        
        # Add dropoff synonyms
        for name, data in self.dropoff_locations.items():
            # Add lowercase version
            synonyms[name.lower()] = name
            
            # Add tag synonyms
            for tag in data["tags"]:
                synonyms[tag.lower()] = name
            
            # Add comprehensive zone synonyms
            zone_synonyms = self._get_zone_synonyms(name)
            for synonym in zone_synonyms:
                synonyms[synonym.lower()] = name
            
            # Add word parts
            words = name.replace("_", " ").lower().split()
            for word in words:
                synonyms[word] = name
            if len(words) > 1:
                synonyms[" ".join(words)] = name
        
        # Add checkpoint synonyms
        for name, data in self.checkpoint_locations.items():
            # Add lowercase version
            synonyms[name.lower()] = name
            
            # Extract letter from checkpoint name (e.g., "CHECKPOINT_A" -> "A")
            letter = name.split('_')[-1] if '_' in name else name
            synonyms[letter.lower()] = name
            synonyms[letter] = name
            
            # Add common checkpoint synonyms
            checkpoint_synonyms = [
                f"checkpoint {letter}",
                f"checkpoint {letter.lower()}",
                f"point {letter}",
                f"point {letter.lower()}",
                f"waypoint {letter}",
                f"waypoint {letter.lower()}",
                f"marker {letter}",
                f"marker {letter.lower()}",
                f"stop {letter}",
                f"stop {letter.lower()}",
                f"visit {letter}",
                f"visit {letter.lower()}",
                f"go to {letter}",
                f"go to {letter.lower()}",
                f"navigate to {letter}",
                f"navigate to {letter.lower()}"
            ]
            for synonym in checkpoint_synonyms:
                synonyms[synonym] = name
        
        return synonyms
    
    def _get_category_synonyms(self, pickup_name):
        """Get comprehensive synonyms for pickup categories."""
        category_synonyms = {
            "PICKUP_BEVERAGES": ["beverages", "drinks", "drink station", "beverage station", "soda", "water bottle", "drink aisle"],
            "PICKUP_PRODUCE": ["fruits", "produce", "apples", "bananas", "oranges", "fruit section", "vegetables"],
            "PICKUP_DAIRY": ["dairy", "milk", "cheese", "yogurt", "butter", "dairy section", "milk products"],
            "PICKUP_INCOMING": ["incoming goods", "loading dock", "dock", "receiving", "incoming"],
            "PICKUP_ELECTRONICS": ["electronics", "electronic items", "gadgets", "tech", "electronic section"],
            "PICKUP_TEXTILES": ["textiles", "fabric", "clothing", "cloth", "textile section"],
            "PICKUP_TOOLS": ["tools", "hardware", "tool section", "equipment"],
            "PICKUP_STATIONERY": ["pens", "pencils", "markers", "writing", "stationery", "office supplies"],
            "PICKUP_OFFICE_SUPPLIES": ["supplies", "office supplies", "staplers", "paperclips", "office equipment"],
            "PICKUP_COMPUTERS": ["computers", "laptops", "tech", "computer section", "IT equipment"],
            "PICKUP_COFFEE": ["coffee", "espresso", "cappuccino", "coffee station", "cafe"],
            "PICKUP_MEDICATIONS": ["medicine", "pills", "medications", "pharmacy", "medical supplies"],
            "PICKUP_PPE": ["gloves", "masks", "ppe", "protective equipment", "safety gear"],
            "PICKUP_EMERGENCY_SUPPLIES": ["emergency kit", "first aid", "defibrillator", "emergency supplies"],
            "PICKUP_INGREDIENTS": ["ingredients", "vegetables", "meat", "sauce", "cooking supplies"],
            "PICKUP_KNIVES": ["knives", "chef knife", "cutlery", "kitchen tools"],
            "PICKUP_UTENSILS": ["forks", "utensils", "cutlery", "kitchen utensils"],
            "PICKUP_GLASSWARE": ["test tubes", "glass tubes", "lab equipment", "scientific equipment"],
            "PICKUP_MICROSCOPES": ["microscopes", "microscope", "lab equipment", "scientific instruments"],
            "PICKUP_SAMPLES": ["samples", "specimens", "vials", "lab samples"],
            "PICKUP_ELECTRONICS_PHONES": ["phones", "smartphones", "electronics", "mobile devices"],
            "PICKUP_APPAREL_TOPS": ["shirts", "t shirts", "clothing", "apparel", "tops"],
            "PICKUP_TOYS": ["toys", "board games", "children's toys", "playthings"],
            "PICKUP_SECURITY_ITEMS": ["security tray", "bins", "belt", "security equipment"],
            "PICKUP_TICKETING": ["tickets", "boarding passes", "travel documents", "ticket counter"],
            "PICKUP_MAPS_INFO": ["map", "information map", "guides", "information"],
            "PICKUP_HANDTOOLS": ["hammer", "hand tools", "hammers", "construction tools"],
            "PICKUP_LUMBER": ["wood", "timber", "lumber", "building materials"],
            "PICKUP_SAFETY_HELMETS": ["helmet", "hard hat", "helmets", "safety equipment"],
            "PICKUP_MAZE_ITEM": ["maze item", "item", "object", "package"],
        }
        return category_synonyms.get(pickup_name, [])
    
    def _get_zone_synonyms(self, zone_name):
        """Get comprehensive synonyms for dropoff zones."""
        zone_synonyms = {
            "DeliveryArea": ["delivery area", "drop-off area", "destination", "delivery zone", "drop zone"],
            "Checkout": ["checkout", "cashier", "payment", "register", "checkout counter"],
            "ShippingArea": ["shipping area", "shipping", "outbound", "dispatch", "shipping zone"],
            "HighValueZone": ["secure zone", "valuable items", "secure area", "high value zone"],
            "Desks": ["workstations", "desks", "employees", "work area", "office space"],
            "BossOffice": ["boss office", "executive", "management", "private", "executive office"],
            "Patients": ["patients", "rooms", "beds", "patient area", "ward"],
            "ICU": ["critical", "icu", "intensive", "critical care", "intensive care"],
            "Surgery": ["surgery", "prep", "operating", "surgical", "operating room"],
            "Counter": ["service", "pass", "orders", "service counter", "order pickup"],
            "Fridge": ["cold", "refrigerated", "fresh", "refrigerator", "cold storage"],
            "Stove": ["hot", "cooking", "grill", "cooking area", "kitchen"],
            "Trash": ["waste", "disposal", "hazardous", "waste disposal", "trash area"],
            "CleanRoom": ["sterile", "clean", "contamination-free", "clean room", "sterile area"],
            "Chemicals": ["hazardous", "chemicals", "dangerous", "chemical storage", "hazardous materials"],
            "Help": ["service", "help", "information", "customer service", "help desk"],
            "Fitting": ["fitting", "changing", "rooms", "fitting room", "changing area"],
            "Furniture": ["home", "furniture", "decor", "furniture section", "home goods"],
            "Gate": ["gate", "boarding", "departure", "boarding gate", "departure gate"],
            "Baggage": ["baggage", "arrival", "luggage", "baggage claim", "luggage area"],
            "Shopping": ["shopping", "duty-free", "luxury", "shopping area", "retail"],
            "Food": ["food", "restaurants", "dining", "food court", "dining area"],
            "Building": ["construction", "work", "building", "construction site", "work area"],
            "Danger": ["dangerous", "hazardous", "caution", "danger zone", "hazard area"],
            "Safe": ["safe", "clean", "finished", "safe area", "clean zone"],
            "MazeExit": ["exit", "delivery", "destination", "drop-off", "maze exit"],
        }
        return zone_synonyms.get(zone_name, [])
    
    def _sanitize_filename(self, name):
        """Sanitize filename by removing invalid characters."""
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.lower()
    
    def run(self):
        """Run the world builder."""
        self.root.mainloop()


class PickupLocationDialog:
    """Dialog for configuring pickup locations."""
    
    def __init__(self, parent, row, col, existing_name=None):
        self.result = None
        self.row = row
        self.col = col
        self.existing_name = existing_name
        # Store reference to parent for accessing pickup_locations
        self.parent_widget = parent
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Pickup Location")
        self.dialog.geometry("500x600")
        self.dialog.configure(bg='#1a1a1a')
        self.dialog.transient(parent)
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"500x600+{x}+{y}")
        
        # Ensure dialog is not too large for screen
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        if x + 500 > screen_width:
            x = screen_width - 520
        if y + 600 > screen_height:
            y = screen_height - 620
        self.dialog.geometry(f"500x600+{x}+{y}")
        
        self._create_widgets()
        
        # Set grab after widgets are created and dialog is visible
        self.dialog.update_idletasks()
        self.dialog.grab_set()
        
        # Add keyboard shortcuts
        self.dialog.bind('<Return>', lambda e: self._on_ok())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
        
        # Ensure grab is released when dialog is destroyed (e.g., clicking X button)
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Configure fields after everything is created and visible
        self.dialog.after(100, self._configure_fields)
    
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_text = "Edit Pickup Location" if self.existing_name else "Add Pickup Location"
        title_label = tk.Label(main_frame, text=title_text, 
                              font=("Arial", 14, "bold"), bg='#1a1a1a', fg='#f0f6fc')
        title_label.pack(pady=(0, 8))
        
        # Location info
        location_label = tk.Label(main_frame, text=f"Location: Row {self.row}, Column {self.col}",
                                font=("Arial", 9), bg='#1a1a1a', fg='#8b949e')
        location_label.pack(pady=(0, 15))
        
        # Name input
        tk.Label(main_frame, text="Name:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.name_var = tk.StringVar(value=self.existing_name or "PICKUP_")
        self.name_entry = tk.Entry(main_frame, textvariable=self.name_var, width=35,
                                  bg='white', fg='black', state='normal')
        self.name_entry.pack(fill=tk.X, pady=(3, 10))
        
        # Direction input
        tk.Label(main_frame, text="Pick Direction:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.direction_var = tk.StringVar(value="N")
        direction_combo = ttk.Combobox(main_frame, textvariable=self.direction_var, 
                                     values=["N", "S", "E", "W"], state='readonly', width=10)
        direction_combo.pack(anchor=tk.W, pady=(3, 10))
        
        # Object category selection
        category_label = tk.Label(main_frame, text="Object Category (optional):", 
                                bg='#1a1a1a', fg='#ffa657', font=("Arial", 10, "bold"))
        category_label.pack(anchor=tk.W)
        
        # Get object categories from parent - try both master and parent_widget
        parent_widget = getattr(self, 'parent_widget', self.dialog.master)
        parent_object_categories = getattr(parent_widget, 'object_categories', {})
        category_names = ["None"] + list(parent_object_categories.keys())
        
        self.category_var = tk.StringVar(value="None")
        category_combo = ttk.Combobox(main_frame, textvariable=self.category_var, 
                                    values=category_names, state='readonly', width=35)
        category_combo.pack(fill=tk.X, pady=(3, 8))
        category_combo.bind('<<ComboboxSelected>>', self._on_category_change)
        
        # Objects selection section
        objects_label = tk.Label(main_frame, text="Objects at this location:", 
                                bg='#1a1a1a', fg='#39d353', font=("Arial", 10, "bold"))
        objects_label.pack(anchor=tk.W, pady=(8, 3))
        
        # Frame for objects listbox with scrollbar
        objects_frame = tk.Frame(main_frame, bg='#1a1a1a')
        objects_frame.pack(fill=tk.BOTH, expand=True, pady=(3, 8))
        
        # Listbox for objects with selection
        scrollbar = tk.Scrollbar(objects_frame, orient=tk.VERTICAL)
        self.objects_listbox = tk.Listbox(objects_frame, 
                                         selectmode=tk.EXTENDED,
                                         bg='#2d2d2d', fg='#f0f6fc',
                                         selectbackground='#007AFF',
                                         selectforeground='#ffffff',
                                         font=("Arial", 9),
                                         yscrollcommand=scrollbar.set,
                                         height=8)
        scrollbar.config(command=self.objects_listbox.yview)
        self.objects_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame for object management buttons
        object_buttons_frame = tk.Frame(main_frame, bg='#1a1a1a')
        object_buttons_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Add custom object button
        tk.Button(object_buttons_frame, text="Add Custom Object", 
                 command=self._add_custom_object,
                 bg='#39d353', fg='#f0f6fc', font=("Arial", 9, "bold"),
                 relief='flat', padx=10, pady=3).pack(side=tk.LEFT, padx=(0, 5))
        
        # Remove selected objects button
        tk.Button(object_buttons_frame, text="Remove Selected", 
                 command=self._remove_selected_objects,
                 bg='#f85149', fg='#f0f6fc', font=("Arial", 9, "bold"),
                 relief='flat', padx=10, pady=3).pack(side=tk.LEFT)
        
        # Initialize objects list - if editing, try to detect category and load existing objects
        if self.existing_name:
            self._load_existing_objects()
        else:
            self._update_objects_list()
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Button(button_frame, text="OK", command=self._on_ok,
                 bg='#007AFF', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Button(button_frame, text="Cancel", command=self._on_cancel,
                 bg='#21262d', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT)
    
    def _configure_fields(self):
        """Configure fields after dialog is fully created and visible."""
        try:
            # Ensure all entry fields are properly configured
            self.name_entry.config(state='normal', bg='white', fg='black')
            
            # Focus on name entry and move cursor to end
            self.name_entry.focus_set()
            self.name_entry.icursor(tk.END)
            
            print("DEBUG: Pickup dialog fields configured successfully")
        except Exception as e:
            print(f"DEBUG: Error configuring pickup fields: {e}")
    
    def _load_existing_objects(self):
        """Load existing objects when editing a pickup location."""
        parent_widget = getattr(self, 'parent_widget', self.dialog.master)
        if not hasattr(parent_widget, 'pickup_locations') or self.existing_name not in parent_widget.pickup_locations:
            self._update_objects_list()
            return
        
        existing_data = parent_widget.pickup_locations[self.existing_name]
        existing_objects = existing_data.get('objects', [])
        existing_direction = existing_data.get('direction', 'N')
        
        # Set the direction
        self.direction_var.set(existing_direction)
        
        # Try to detect category from existing objects
        parent_object_categories = getattr(parent_widget, 'object_categories', {})
        matching_category = None
        for category, category_objects in parent_object_categories.items():
            if existing_objects and all(obj in category_objects for obj in existing_objects):
                matching_category = category
                break
        
        if matching_category:
            self.category_var.set(matching_category)
        
        # Update objects list (will show category objects + existing objects)
        self._update_objects_list()
        
        # Ensure existing objects are visible (they might be from category or custom)
        current_items = list(self.objects_listbox.get(0, tk.END))
        for obj in existing_objects:
            if obj not in current_items:
                self.objects_listbox.insert(tk.END, obj)
    
    def _on_category_change(self, event=None):
        """Handle category change."""
        self._update_objects_list()
    
    def _update_objects_list(self):
        """Update objects listbox based on selected category."""
        # Clear current list
        self.objects_listbox.delete(0, tk.END)
        
        category = self.category_var.get()
        if category == "None":
            # If no category selected, show existing objects if editing
            if self.existing_name:
                parent_widget = getattr(self, 'parent_widget', self.dialog.master)
                if hasattr(parent_widget, 'pickup_locations') and self.existing_name in parent_widget.pickup_locations:
                    existing_objects = parent_widget.pickup_locations[self.existing_name].get('objects', [])
                    for obj in existing_objects:
                        self.objects_listbox.insert(tk.END, obj)
            return
        
        # Get objects from selected category
        parent_widget = getattr(self, 'parent_widget', self.dialog.master)
        parent_object_categories = getattr(parent_widget, 'object_categories', {})
        objects = parent_object_categories.get(category, [])
        
        # Add all objects from category to listbox
        for obj in objects:
            self.objects_listbox.insert(tk.END, obj)
        
        # If editing existing location, also include any existing objects not in category
        if self.existing_name:
            if hasattr(parent_widget, 'pickup_locations') and self.existing_name in parent_widget.pickup_locations:
                existing_objects = parent_widget.pickup_locations[self.existing_name].get('objects', [])
                for obj in existing_objects:
                    if obj not in objects:
                        self.objects_listbox.insert(tk.END, obj)
    
    def _add_custom_object(self):
        """Add a custom object to the list."""
        # Temporarily release grab for the dialog
        try:
            self.dialog.grab_release()
        except:
            pass
        
        custom_obj = simpledialog.askstring("Add Custom Object", 
                                           "Enter object name:",
                                           parent=self.dialog)
        
        # Restore grab after dialog closes
        try:
            self.dialog.grab_set()
        except:
            pass
        
        if custom_obj and custom_obj.strip():
            obj_name = custom_obj.strip()
            # Check if already exists
            current_items = self.objects_listbox.get(0, tk.END)
            if obj_name not in current_items:
                self.objects_listbox.insert(tk.END, obj_name)
                # Select the newly added item
                index = self.objects_listbox.size() - 1
                self.objects_listbox.selection_set(index)
                self.objects_listbox.see(index)
        else:
                messagebox.showinfo("Object Exists", f"'{obj_name}' is already in the list.", parent=self.dialog)
    
    def _remove_selected_objects(self):
        """Remove selected objects from the list."""
        selected_indices = self.objects_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select objects to remove.", parent=self.dialog)
            return
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.objects_listbox.delete(index)
    
    def _manual_enable_fields(self):
        """Manually enable fields - debug function."""
        try:
            print("DEBUG: Manual field enable requested")
            self.name_entry.config(state='normal', bg='white', fg='black')
            
            # Force focus and cursor
            self.name_entry.focus_set()
            self.name_entry.icursor(0)
            
            print("DEBUG: Fields manually enabled")
        except Exception as e:
            print(f"DEBUG: Error in manual enable: {e}")
    
    def _on_ok(self):
        """Handle OK button."""
        name = self.name_var.get().strip()
        direction = self.direction_var.get()
        
        if not name:
            messagebox.showwarning("Name Required", "Please enter a name for the pickup location.", parent=self.dialog)
            return
        
        # Get selected objects from listbox
        objects = list(self.objects_listbox.get(0, tk.END))
        
        if not objects:
            # Ask user if they want to proceed without objects
            proceed = messagebox.askyesno("No Objects", 
                                        "No objects are defined for this pickup location.\n\n"
                                        "Do you want to continue without objects?",
                                        parent=self.dialog)
            if not proceed:
                return
        
        # Release grab before destroying dialog
        try:
            self.dialog.grab_release()
        except:
            pass
        
        self.result = (name, direction, objects)
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button."""
        # Release grab before destroying dialog
        try:
            self.dialog.grab_release()
        except:
            pass
        
        self.result = None
        self.dialog.destroy()


class DropoffLocationDialog:
    """Dialog for configuring dropoff locations."""
    
    def __init__(self, parent, row, col, existing_name=None):
        self.result = None
        self.row = row
        self.col = col
        self.existing_name = existing_name
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Dropoff Location")
        self.dialog.geometry("450x350")
        self.dialog.configure(bg='#1a1a1a')
        self.dialog.transient(parent)
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (350 // 2)
        self.dialog.geometry(f"450x350+{x}+{y}")
        
        # Ensure dialog is not too large for screen
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        if x + 450 > screen_width:
            x = screen_width - 470
        if y + 350 > screen_height:
            y = screen_height - 370
        self.dialog.geometry(f"450x350+{x}+{y}")
        
        self._create_widgets()
        
        # Set grab after widgets are created and dialog is visible
        self.dialog.update_idletasks()
        self.dialog.grab_set()
        
        # Add keyboard shortcuts
        self.dialog.bind('<Return>', lambda e: self._on_ok())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
        
        # Configure fields after everything is created and visible
        self.dialog.after(100, self._configure_fields)
    
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_text = "Edit Dropoff Location" if self.existing_name else "Add Dropoff Location"
        title_label = tk.Label(main_frame, text=title_text, 
                              font=("Arial", 14, "bold"), bg='#1a1a1a', fg='#f0f6fc')
        title_label.pack(pady=(0, 8))
        
        # Location info
        location_label = tk.Label(main_frame, text=f"Location: Row {self.row}, Column {self.col}",
                                font=("Arial", 9), bg='#1a1a1a', fg='#8b949e')
        location_label.pack(pady=(0, 15))
        
        # Name input
        tk.Label(main_frame, text="Name:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.name_var = tk.StringVar(value=self.existing_name or "DeliveryArea")
        self.name_entry = tk.Entry(main_frame, textvariable=self.name_var, width=35,
                                  bg='white', fg='black', state='normal')
        self.name_entry.pack(fill=tk.X, pady=(3, 10))
        
        # Direction input
        tk.Label(main_frame, text="Direction:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.direction_var = tk.StringVar(value="N")
        direction_combo = ttk.Combobox(main_frame, textvariable=self.direction_var, 
                                     values=["N", "S", "E", "W"], state='readonly', width=10)
        direction_combo.pack(anchor=tk.W, pady=(3, 10))
        
        # Tags input - make it more prominent
        tags_label = tk.Label(main_frame, text="Tags (comma-separated, optional):", 
                             bg='#1a1a1a', fg='#00d4aa', font=("Arial", 10, "bold"))
        tags_label.pack(anchor=tk.W)
        self.tags_var = tk.StringVar()
        self.tags_entry = tk.Entry(main_frame, textvariable=self.tags_var, width=35,
                                  bg='white', fg='black', state='normal', font=("Arial", 10))
        self.tags_entry.pack(fill=tk.X, pady=(3, 8))
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Button(button_frame, text="OK", command=self._on_ok,
                 bg='#007AFF', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Button(button_frame, text="Cancel", command=self._on_cancel,
                 bg='#21262d', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT)
    
    def _configure_fields(self):
        """Configure fields after dialog is fully created and visible."""
        try:
            # Ensure all entry fields are properly configured
            self.name_entry.config(state='normal', bg='white', fg='black')
            self.tags_entry.config(state='normal', bg='white', fg='black')
            
            # Focus on name entry and move cursor to end
            self.name_entry.focus_set()
            self.name_entry.icursor(tk.END)
            
            print("DEBUG: Dropoff dialog fields configured successfully")
        except Exception as e:
            print(f"DEBUG: Error configuring dropoff fields: {e}")
    
    def _on_ok(self):
        """Handle OK button."""
        name = self.name_var.get().strip()
        direction = self.direction_var.get()
        tags_text = self.tags_var.get().strip()
        
        if not name:
            messagebox.showwarning("Name Required", "Please enter a name for the dropoff location.")
            return
        
        # Parse tags
        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
        
        self.result = (name, direction, tags)
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()


class CheckpointLocationDialog:
    """Dialog for configuring checkpoint locations."""
    
    def __init__(self, parent, row, col, existing_name=None):
        self.result = None
        self.row = row
        self.col = col
        self.existing_name = existing_name
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Checkpoint Location")
        self.dialog.geometry("450x350")
        self.dialog.configure(bg='#1a1a1a')
        self.dialog.transient(parent)
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (350 // 2)
        self.dialog.geometry(f"450x350+{x}+{y}")
        
        # Ensure dialog is not too large for screen
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        if x + 450 > screen_width:
            x = screen_width - 470
        if y + 350 > screen_height:
            y = screen_height - 370
        self.dialog.geometry(f"450x350+{x}+{y}")
        
        self._create_widgets()
        
        # Set grab after widgets are created and dialog is visible
        self.dialog.update_idletasks()
        self.dialog.grab_set()
        
        # Add keyboard shortcuts
        self.dialog.bind('<Return>', lambda e: self._on_ok())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
        
        # Configure fields after everything is created and visible
        self.dialog.after(100, self._configure_fields)
    
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_text = "Edit Checkpoint Location" if self.existing_name else "Add Checkpoint Location"
        title_label = tk.Label(main_frame, text=title_text, 
                              font=("Arial", 14, "bold"), bg='#1a1a1a', fg='#f0f6fc')
        title_label.pack(pady=(0, 8))
        
        # Location info
        location_label = tk.Label(main_frame, text=f"Location: Row {self.row}, Column {self.col}",
                                font=("Arial", 9), bg='#1a1a1a', fg='#8b949e')
        location_label.pack(pady=(0, 15))
        
        # Name input - auto-generate checkpoint name
        tk.Label(main_frame, text="Checkpoint Name:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.name_var = tk.StringVar(value=self.existing_name)
        self.name_entry = tk.Entry(main_frame, textvariable=self.name_var, width=35,
                                  bg='white', fg='black', state='normal')
        self.name_entry.pack(fill=tk.X, pady=(3, 10))
        
        # Direction input
        tk.Label(main_frame, text="Direction:", bg='#1a1a1a', fg='#f0f6fc').pack(anchor=tk.W)
        self.direction_var = tk.StringVar(value="N")
        direction_combo = ttk.Combobox(main_frame, textvariable=self.direction_var, 
                                     values=["N", "S", "E", "W"], state='readonly', width=10)
        direction_combo.pack(anchor=tk.W, pady=(3, 8))
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Button(button_frame, text="OK", command=self._on_ok,
                 bg='#007AFF', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Button(button_frame, text="Cancel", command=self._on_cancel,
                 bg='#21262d', fg='#f0f6fc', font=("Arial", 10, "bold"),
                 relief='flat', padx=20, pady=5).pack(side=tk.RIGHT)
    
    def _configure_fields(self):
        """Configure fields after dialog is fully created and visible."""
        try:
            # Ensure all entry fields are properly configured
            self.name_entry.config(state='normal', bg='white', fg='black')
            
            # Focus on name entry and move cursor to end
            self.name_entry.focus_set()
            self.name_entry.icursor(tk.END)
            
            print("DEBUG: Checkpoint dialog fields configured successfully")
        except Exception as e:
            print(f"DEBUG: Error configuring checkpoint fields: {e}")
    
    def _on_ok(self):
        """Handle OK button."""
        name = self.name_var.get().strip()
        direction = self.direction_var.get()
        
        if not name:
            messagebox.showwarning("Name Required", "Please enter a name for the checkpoint location.")
            return
        
        self.result = (name, direction)
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()


def main():
    """Main entry point for standalone world builder."""
    app = StandaloneWorldBuilder()
    app.run()


if __name__ == "__main__":
    main()
