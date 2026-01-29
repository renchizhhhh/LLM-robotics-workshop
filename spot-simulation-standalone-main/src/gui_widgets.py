#!/usr/bin/env python3
"""
GUI Widgets Module - UI Components
Handles all UI widget creation, styling, and layout.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
from message_bus import rospy, String


# Font helpers with fallbacks
# Note: Inter and JetBrains Mono fonts are preferred but may not be installed
# Tkinter will automatically fallback to system defaults if fonts are not available
def get_inter_font(size=10, weight="normal"):
    """Get Inter font with automatic system fallback."""
    # Tkinter font specification: (family, size, style)
    # If Inter is not available, Tkinter will use system default sans-serif
    if weight == "bold":
        return ("Inter", size, "bold")
    return ("Inter", size)

def get_mono_font(size=10):
    """Get JetBrains Mono font with automatic system fallback."""
    # Tkinter font specification: (family, size)
    # If JetBrains Mono is not available, Tkinter will use system default monospace
    return ("JetBrains Mono", size)


class GUIWidgets:
    """Manages all GUI widgets and UI components."""
    
    def __init__(self, root, main_frame, sidebar_frame, bottom_frame, world_manager, current_world_id, world_name, gui_core=None):
        self.root = root
        self.main_frame = main_frame
        self.sidebar_frame = sidebar_frame
        self.bottom_frame = bottom_frame
        self.world_manager = world_manager
        self.current_world_id = current_world_id
        self.world_name = world_name
        self.gui_core = gui_core
        
        # Widget references
        self.state_label = None
        self.position_label = None
        self.action_label = None
        self.arm_label = None
        self.gripper_label = None
        self.connection_label = None
        self.world_var = None
        self.world_combo = None
        self.model_var = None
        self.model_combo = None
        self.prompt_var = None
        self.prompt_combo = None
        self.reset_button = None
        self.stop_button = None
        self.feedback_text = None
        self.control_mode = None
        self.mode_combo = None
        self.inputs_control_frame = None
        self.load_inputs_button = None
        self.inputs_summary_label = None
        self.inputs_table_frame = None
        self.inputs_tree = None
        self.inputs_button_frame = None
        self.show_all_trajectories_button = None
        self.heatmap_toggle_button = None
        self.preview_trajectory_button = None
        self.command_entry = None
        self.send_button = None
        self.clear_button = None
        self.single_preview_trajectory_button = None
        self.single_approve_button = None
        self.single_dismiss_button = None
        self.strict_scoring_var = None
        self.strict_scoring_toggle = None
        self.wall_enforcement_var = None
        self.wall_enforcement_toggle = None
        
        # Baseline mode widget references
        self.baseline_controls_frame = None
        self.baseline_selector_var = None
        self.baseline_combo = None
        self.baseline_show_button = None
        self.baseline_prompt_text = None
        self.baseline_plan_text = None
        self.baseline_execute_button = None
        
        # New widget references
        self.presenter_mode_var = None
        self.presenter_mode_toggle = None
        self.plan_preview_card = None
        self.plan_steps_list = None
        self.prompt_entry = None
        self.generate_plan_button = None
        self.control_bar_frame = None
        self.play_button = None
        self.pause_button = None
        self.step_button = None
        self.speed_slider = None
        self.abort_button = None
        
        # Plan preview state
        self.current_plan_actions = []
        self.plan_card_frame = None
        
        # Initialize legacy widget references for backward compatibility
        self.preview_trajectory_button = None
        self.batch_approve_button = None
        self.single_preview_trajectory_button = None
        self.single_approve_button = None
        self.single_dismiss_button = None
        self.clear_button = None
        
        # Initialize widgets
        try:
            self._create_tabbed_sidebar()
            self._create_fixed_control_bar()
            self._setup_styling()
            
            # Set up command_entry alias for backward compatibility (after widgets are created)
            # Ensure prompt_entry was created successfully
            if not hasattr(self, 'prompt_entry') or self.prompt_entry is None:
                rospy.logerr("ERROR: prompt_entry was not created successfully!")
                raise RuntimeError("prompt_entry widget was not created - check _create_plan_tab()")
            
            self.command_entry = self.prompt_entry
            
            if self.generate_plan_button:
                self.send_button = self.generate_plan_button
            else:
                rospy.logwarn("generate_plan_button was not created")
        except Exception as e:
            rospy.logerr(f"Error creating GUI widgets: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise
    
    def _create_tabbed_sidebar(self):
        """Create tabbed sidebar with Plan, Status, and Logs tabs."""
        # Presenter mode toggle in top right
        top_bar = tk.Frame(self.sidebar_frame, bg='#161b22', height=32)
        top_bar.pack(fill=tk.X, padx=8, pady=8)
        top_bar.pack_propagate(False)
        
        self.presenter_mode_var = tk.BooleanVar(value=False)
        self.presenter_mode_toggle = tk.Checkbutton(
            top_bar,
            text="Presenter mode",
            variable=self.presenter_mode_var,
            command=self._on_toggle_presenter_mode,
            bg='#161b22',
            fg='#f0f6fc',
            selectcolor='#30363d',
            activebackground='#161b22',
            activeforeground='#f0f6fc',
            font=get_inter_font(10, "bold"),
            highlightthickness=0,
            bd=0
        )
        self.presenter_mode_toggle.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#161b22', borderwidth=0)
        style.configure('TNotebook.Tab', background='#21262d', foreground='#8b949e', 
                       padding=[16, 8], font=get_inter_font(10, "bold"))
        style.map('TNotebook.Tab',
                 background=[('selected', '#161b22')],
                 foreground=[('selected', '#f0f6fc')])
        
        self.notebook = ttk.Notebook(self.sidebar_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        
        # Plan tab
        self.plan_frame = tk.Frame(self.notebook, bg='#161b22', padx=16, pady=16)
        self.notebook.add(self.plan_frame, text="Plan")
        
        # Status tab
        self.status_frame = tk.Frame(self.notebook, bg='#161b22', padx=16, pady=16)
        self.notebook.add(self.status_frame, text="Status")
        
        # Logs tab (hidden by default in presenter mode)
        self.logs_frame = tk.Frame(self.notebook, bg='#161b22', padx=16, pady=16)
        self.notebook.add(self.logs_frame, text="Logs")
        
        # Create content for each tab
        self._create_plan_tab()
        self._create_status_tab()
        self._create_logs_tab()
        
        # Start with Plan tab active
        self.notebook.select(0)
    
    def _create_plan_tab(self):
        """Create Plan tab content with prompt input and plan preview."""
        # Mode switcher (Single vs Batch) - at the top for logical flow
        self._create_mode_switcher_plan()
        
        # Model and prompt template selectors - configuration settings
        self._create_model_selector_plan()
        self._create_prompt_selector_plan()
        
        # Separator before command input area
        ttk.Separator(self.plan_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(16, 8))
        
        # Prompt input section (for single command mode)
        self.prompt_label = tk.Label(self.plan_frame, text="Enter command:", 
                               font=get_inter_font(16, "bold"), bg='#161b22', fg='#f0f6fc')
        self.prompt_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Prompt entry - store as instance variable for show/hide (multi-line Text widget)
        self.entry_frame = tk.Frame(self.plan_frame, bg='#161b22')
        self.entry_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Create a frame for the text widget with border - takes full width
        text_container = tk.Frame(self.entry_frame, bg='#0d1117', relief='flat', bd=1)
        text_container.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Text widget for multi-line input - taller for better visibility
        text_scrollbar = tk.Scrollbar(text_container, orient=tk.VERTICAL)
        self.prompt_entry = tk.Text(text_container, height=12, wrap=tk.WORD,
                                    font=get_inter_font(18), 
                                    bg='#0d1117', fg='#f0f6fc',
                                    insertbackground='#f0f6fc', relief='flat', bd=0,
                                    yscrollcommand=text_scrollbar.set,
                                    padx=8, pady=8)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.config(command=self.prompt_entry.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame below the text area - always visible
        button_frame = tk.Frame(self.entry_frame, bg='#161b22')
        button_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.generate_plan_button = tk.Button(button_frame, text="Generate plan",
                                             command=self._on_generate_plan,
                                             bg='#007AFF', fg='#f0f6fc', 
                                             font=get_inter_font(12, "bold"),
                                             activebackground='#0051d5', 
                                             activeforeground='#f0f6fc',
                                             relief='flat', bd=0, padx=16, pady=8, 
                                             highlightthickness=0)
        self.generate_plan_button.pack()
        
        # Display last entered prompt below the input field with status dot
        self.prompt_frame = tk.Frame(self.plan_frame, bg='#161b22')
        self.prompt_frame.pack(fill=tk.X, anchor=tk.W, pady=(4, 8), padx=(0, 0))
        
        # Status dot container (fixed width, always visible on left)
        self.status_dot_frame = tk.Frame(self.prompt_frame, bg='#0d1117', width=32, height=24)
        self.status_dot_frame.pack(side=tk.LEFT, anchor=tk.W, padx=(0, 12), pady=2)
        self.status_dot_frame.pack_propagate(False)
        
        # Status dot canvas for animation (futuristic indicator - always visible on left with dark background)
        self.plan_status_canvas = tk.Canvas(self.status_dot_frame, bg='#0d1117', width=32, height=24,
                                           highlightthickness=0, bd=0)
        self.plan_status_canvas.pack(expand=True, fill=tk.BOTH)
        
        # Keep Label for backward compatibility (hidden)
        self.plan_status_dot = tk.Label(self.status_dot_frame, text="◉", 
                                        font=get_inter_font(16, "bold"), bg='#0d1117', fg='#4a5568',
                                        width=2, height=1, anchor='center')
        # Don't pack the label, we use canvas instead
        
        # Animation state
        self.plan_status_loading = False
        self.plan_status_animation_id = None
        self.plan_status_angle = 0
        
        # Initialize with grey inactive state
        center_x, center_y = 16, 12
        self.plan_status_canvas.create_oval(
            center_x - 6, center_y - 6,
            center_x + 6, center_y + 6,
            fill='#4a5568', outline='#4a5568'
        )
        
        # Text and timestamp container (takes remaining space)
        text_frame = tk.Frame(self.prompt_frame, bg='#161b22')
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor=tk.W)
        
        # Last prompt label (with fixed max width to prevent pushing timestamp out)
        self.last_prompt_label = tk.Label(text_frame, text="", 
                                         font=get_inter_font(16), bg='#161b22', fg='#c9d1d9',
                                         wraplength=600, justify=tk.LEFT, anchor='w')
        self.last_prompt_label.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)
        
        # Timestamp label (on the same line, right-aligned, fixed width)
        self.plan_status_time = tk.Label(text_frame, text="", 
                                         font=get_inter_font(8), bg='#161b22', fg='#8b949e',
                                         anchor='e', width=12)
        self.plan_status_time.pack(side=tk.RIGHT, anchor=tk.E, padx=(8, 0))
        
        # Batch controls (initially hidden)
        self._create_batch_controls_plan()
        
        # Baseline controls (initially hidden)
        self._create_baseline_controls_plan()
        
        # Plan preview card (initially hidden)
        self.plan_card_frame = None
    
    def _create_mode_switcher_plan(self):
        """Create mode switcher in Plan tab."""
        mode_frame = tk.Frame(self.plan_frame, bg='#161b22')
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        
        mode_label = tk.Label(mode_frame, text="Mode:", font=get_inter_font(10, "bold"),
                             bg='#161b22', fg='#8b949e')
        mode_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.control_mode = tk.StringVar(value="Single Command")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.control_mode,
                                       font=get_inter_font(10), state='readonly', width=20)
        self.mode_combo['values'] = ('Single Command', 'Batch Commands', 'Baseline Mode')
        self.mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change_plan)
    
    def _create_batch_controls_plan(self):
        """Create batch processing controls in Plan tab."""
        self.batch_controls_frame = tk.Frame(self.plan_frame, bg='#161b22')
        # Initially hidden - will be shown when Batch mode is selected
        
        # Inputs file selector
        file_row = tk.Frame(self.batch_controls_frame, bg='#161b22')
        file_row.pack(fill=tk.X, pady=(0, 8))
        
        file_label = tk.Label(file_row, text="Inputs file:",
                             font=get_inter_font(10, "bold"), bg='#161b22', fg='#8b949e')
        file_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.inputs_file_var = tk.StringVar(value='')
        self.inputs_file_combo = ttk.Combobox(file_row, textvariable=self.inputs_file_var,
                                             font=get_inter_font(10), state='readonly', width=22)
        try:
            user_data_dir = Path(__file__).resolve().parents[1] / 'user_data'
            files = [p.name for p in user_data_dir.glob('inputs*.json')]
            files = sorted([f for f in files if f != 'inputs.json'])
        except Exception:
            files = []
        self.inputs_file_combo['values'] = tuple(files)
        if files:
            self.inputs_file_combo.set(files[0])
        self.inputs_file_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        # Buttons row
        buttons_row = tk.Frame(self.batch_controls_frame, bg='#161b22')
        buttons_row.pack(fill=tk.X, pady=(0, 8))
        
        self.refresh_inputs_button = tk.Button(buttons_row, text="Refresh",
                                              command=self._on_refresh_input_files,
                                              bg='#58a6ff', fg='white', font=get_inter_font(9, "bold"),
                                              activebackground='#1f6feb', activeforeground='white',
                                              relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.refresh_inputs_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.load_inputs_button = tk.Button(buttons_row, text="Load Inputs",
                                           command=self._on_load_inputs,
                                           bg='#007AFF', fg='white', font=get_inter_font(9, "bold"),
                                           activebackground='#0051d5', activeforeground='white',
                                           relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.load_inputs_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.copy_user_data_button = tk.Button(buttons_row, text="Copy & Clear",
                                              command=self._on_copy_and_clear_user_data,
                                              bg='#FF6B6B', fg='white', font=get_inter_font(9, "bold"),
                                              activebackground='#ff5252', activeforeground='white',
                                              relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.copy_user_data_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.save_baseline_button = tk.Button(buttons_row, text="Save",
                                              command=self._on_save_as_baseline,
                                              bg='#39d353', fg='white', font=get_inter_font(9, "bold"),
                                              activebackground='#2ea043', activeforeground='white',
                                              relief='flat', bd=0, padx=12, pady=6, highlightthickness=0,
                                              state='disabled')
        self.save_baseline_button.pack(side=tk.LEFT)
        
        self.inputs_summary_label = tk.Label(self.batch_controls_frame, text="No inputs loaded",
                                            font=get_inter_font(9), bg='#161b22', fg='#8b949e')
        self.inputs_summary_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Inputs table
        self.inputs_table_frame = tk.Frame(self.batch_controls_frame, bg='#161b22')
        self.inputs_table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        inputs_scrollbar = tk.Scrollbar(self.inputs_table_frame, orient=tk.VERTICAL)
        self.inputs_tree = ttk.Treeview(self.inputs_table_frame,
                                       columns=('command', 'status', 'score'),
                                       show='headings',
                                       height=8,  # Increased height for better visibility
                                       style='InputsTree.Treeview')
        self.inputs_tree.heading('command', text='Command')
        self.inputs_tree.column('command', anchor=tk.W, stretch=True, width=200)
        self.inputs_tree.heading('status', text='Status')
        self.inputs_tree.column('status', anchor=tk.W, stretch=False, width=80)
        self.inputs_tree.heading('score', text='Score')
        self.inputs_tree.column('score', anchor=tk.CENTER, stretch=False, width=60)
        self.inputs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inputs_scrollbar.config(command=self.inputs_tree.yview)
        inputs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inputs_tree.configure(yscrollcommand=inputs_scrollbar.set)
        
        # Action buttons for batch
        self.inputs_button_frame = tk.Frame(self.batch_controls_frame, bg='#161b22')
        self.inputs_button_frame.pack(fill=tk.X, pady=(8, 0))
        
        # Checkboxes
        checkbox_frame = tk.Frame(self.inputs_button_frame, bg='#161b22')
        checkbox_frame.pack(side=tk.RIGHT)
        
        self.strict_scoring_var = tk.BooleanVar(value=False)
        self.strict_scoring_toggle = tk.Checkbutton(
            checkbox_frame,
            text="Strict scoring",
            variable=self.strict_scoring_var,
            command=self._on_toggle_strict_scoring,
            bg='#161b22',
            fg='#f0f6fc',
            selectcolor='#30363d',
            activebackground='#161b22',
            activeforeground='#f0f6fc',
            font=get_inter_font(9),
            highlightthickness=0,
            bd=0
        )
        self.strict_scoring_toggle.pack(side=tk.LEFT, padx=4)
        
        self.wall_enforcement_var = tk.BooleanVar(value=False)
        self.wall_enforcement_toggle = tk.Checkbutton(
            checkbox_frame,
            text="Enforce obstacles",
            variable=self.wall_enforcement_var,
            command=self._on_toggle_wall_enforcement,
            bg='#161b22',
            fg='#f0f6fc',
            selectcolor='#30363d',
            activebackground='#161b22',
            activeforeground='#f0f6fc',
            font=get_inter_font(9),
            highlightthickness=0,
            bd=0
        )
        self.wall_enforcement_toggle.pack(side=tk.LEFT, padx=4)
        
        # Action buttons
        self.show_all_trajectories_button = tk.Button(self.inputs_button_frame, text="Show All",
                                               command=self._on_show_all_previews,
                                               bg='#5E5CE6', fg='white', font=get_inter_font(9, "bold"),
                                               activebackground='#4c4abf', activeforeground='white',
                                               relief='flat', bd=0, padx=12, pady=6, highlightthickness=0,
                                               state='disabled')
        self.show_all_trajectories_button.pack(side=tk.LEFT, padx=(0, 4))
        
        self.heatmap_toggle_button = tk.Button(self.inputs_button_frame, text="Heatmap",
                                              command=self._on_toggle_heatmap,
                                              bg='#FF9F0A', fg='black', font=get_inter_font(9, "bold"),
                                              activebackground='#cc7f08', activeforeground='black',
                                              relief='flat', bd=0, padx=12, pady=6, highlightthickness=0,
                                              state='disabled')
        self.heatmap_toggle_button.pack(side=tk.LEFT, padx=(0, 4))
        
        self.preview_trajectory_button = tk.Button(self.inputs_button_frame, text="Preview",
                                               command=self._on_preview_trajectory,
                                               bg='#FFD700', fg='black', font=get_inter_font(9, "bold"),
                                               activebackground='#FFA500', activeforeground='black',
                                               relief='flat', bd=0, padx=12, pady=6, highlightthickness=0,
                                               state='disabled')
        self.preview_trajectory_button.pack(side=tk.LEFT, padx=(0, 4))
        
        self.batch_approve_button = tk.Button(self.inputs_button_frame, text="Approve",
                                            command=self._on_approve,
                                            bg='#39d353', fg='white', font=get_inter_font(9, "bold"),
                                            activebackground='#2ea043', activeforeground='white',
                                            relief='flat', bd=0, padx=12, pady=6, highlightthickness=0,
                                            state='disabled')
        self.batch_approve_button.pack(side=tk.LEFT)
    
    def _create_baseline_controls_plan(self):
        """Create baseline mode controls in Plan tab."""
        self.baseline_controls_frame = tk.Frame(self.plan_frame, bg='#161b22')
        # Initially hidden - will be shown when Baseline mode is selected
        
        # Baseline selector row
        selector_row = tk.Frame(self.baseline_controls_frame, bg='#161b22')
        selector_row.pack(fill=tk.X, pady=(0, 12))
        
        selector_label = tk.Label(selector_row, text="Select Baseline:",
                                 font=get_inter_font(10, "bold"), bg='#161b22', fg='#8b949e')
        selector_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.baseline_selector_var = tk.StringVar(value='Baseline 1')
        self.baseline_combo = ttk.Combobox(selector_row, textvariable=self.baseline_selector_var,
                                          font=get_inter_font(10), state='readonly', width=22)
        # Dynamically discover available baselines
        baseline_names = self._discover_baselines()
        self.baseline_combo['values'] = tuple(baseline_names) if baseline_names else ('Baseline 1', 'Baseline 2', 'Baseline 3')
        self.baseline_combo.current(0)
        self.baseline_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        self.baseline_show_button = tk.Button(selector_row, text="Show",
                                             bg='#007AFF', fg='white', font=get_inter_font(9, "bold"),
                                             activebackground='#0051d5', activeforeground='white',
                                             relief='flat', bd=0, padx=20, pady=6, highlightthickness=0)
        self.baseline_show_button.pack(side=tk.LEFT)
        
        # Prompt display section
        prompt_label = tk.Label(self.baseline_controls_frame, text="Prompt:",
                               font=get_inter_font(10, "bold"), bg='#161b22', fg='#8b949e')
        prompt_label.pack(anchor=tk.W, pady=(0, 4))
        
        prompt_frame = tk.Frame(self.baseline_controls_frame, bg='#0d1117', relief='solid', bd=1)
        prompt_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 12))
        
        prompt_scrollbar = tk.Scrollbar(prompt_frame, orient=tk.VERTICAL)
        self.baseline_prompt_text = tk.Text(prompt_frame, height=8, wrap=tk.WORD,
                                           bg='#0d1117', fg='#c9d1d9',
                                           font=get_mono_font(16),
                                           relief='flat', bd=0,
                                           yscrollcommand=prompt_scrollbar.set)
        self.baseline_prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        prompt_scrollbar.config(command=self.baseline_prompt_text.yview)
        prompt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.baseline_prompt_text.config(state='disabled')
        
        # Plan display section
        plan_label = tk.Label(self.baseline_controls_frame, text="Plan:",
                             font=get_inter_font(10, "bold"), bg='#161b22', fg='#8b949e')
        plan_label.pack(anchor=tk.W, pady=(0, 4))
        
        plan_frame = tk.Frame(self.baseline_controls_frame, bg='#0d1117', relief='solid', bd=1)
        plan_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        plan_scrollbar = tk.Scrollbar(plan_frame, orient=tk.VERTICAL)
        self.baseline_plan_text = tk.Text(plan_frame, height=10, wrap=tk.WORD,
                                         bg='#0d1117', fg='#c9d1d9',
                                         font=get_mono_font(9),
                                         relief='flat', bd=0,
                                         yscrollcommand=plan_scrollbar.set)
        self.baseline_plan_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        plan_scrollbar.config(command=self.baseline_plan_text.yview)
        plan_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.baseline_plan_text.config(state='disabled')
        
        # Execute button
        execute_button_frame = tk.Frame(self.baseline_controls_frame, bg='#161b22')
        execute_button_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.baseline_execute_button = tk.Button(execute_button_frame, text="Execute Plan",
                                                bg='#39d353', fg='white', font=get_inter_font(10, "bold"),
                                                activebackground='#2ea043', activeforeground='white',
                                                relief='flat', bd=0, padx=20, pady=8, highlightthickness=0,
                                                state='disabled')
        self.baseline_execute_button.pack(side=tk.LEFT)
    
    def _on_mode_change_plan(self, event=None):
        """Handle mode change in Plan tab."""
        mode = self.control_mode.get()
        if mode == "Batch Commands":
            # Hide single command controls
            if hasattr(self, 'prompt_label') and self.prompt_label:
                self.prompt_label.pack_forget()
            if hasattr(self, 'entry_frame') and self.entry_frame:
                self.entry_frame.pack_forget()
            # Hide prompt frame (contains last_prompt_label, status dot, and timestamp)
            if hasattr(self, 'prompt_frame') and self.prompt_frame:
                self.prompt_frame.pack_forget()
            # Hide plan status indicator
            if hasattr(self, 'update_plan_status'):
                self.update_plan_status(show=False)
            # Hide baseline controls
            if hasattr(self, 'baseline_controls_frame') and self.baseline_controls_frame:
                self.baseline_controls_frame.pack_forget()
            
            # Show batch controls
            self.batch_controls_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
            # Update feedback
            if hasattr(self, 'feedback_text') and self.feedback_text:
                self._update_feedback_text("Switched to Batch Commands mode")
        elif mode == "Baseline Mode":
            # Hide single command controls
            if hasattr(self, 'prompt_label') and self.prompt_label:
                self.prompt_label.pack_forget()
            if hasattr(self, 'entry_frame') and self.entry_frame:
                self.entry_frame.pack_forget()
            # Hide prompt frame (contains last_prompt_label, status dot, and timestamp)
            if hasattr(self, 'prompt_frame') and self.prompt_frame:
                self.prompt_frame.pack_forget()
            # Hide plan status indicator
            if hasattr(self, 'update_plan_status'):
                self.update_plan_status(show=False)
            # Hide batch controls
            self.batch_controls_frame.pack_forget()
            
            # Show baseline controls
            if hasattr(self, 'baseline_controls_frame') and self.baseline_controls_frame:
                self.baseline_controls_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
            # Update feedback
            if hasattr(self, 'feedback_text') and self.feedback_text:
                self._update_feedback_text("Switched to Baseline Mode")
        else:
            # Hide batch controls
            self.batch_controls_frame.pack_forget()
            # Hide baseline controls
            if hasattr(self, 'baseline_controls_frame') and self.baseline_controls_frame:
                self.baseline_controls_frame.pack_forget()
            
            # Show single command controls
            if hasattr(self, 'prompt_label') and self.prompt_label:
                self.prompt_label.pack(anchor=tk.W, pady=(0, 8))
            if hasattr(self, 'entry_frame') and self.entry_frame:
                self.entry_frame.pack(fill=tk.X, pady=(0, 8))
            # Show prompt frame (contains last_prompt_label, status dot, and timestamp)
            if hasattr(self, 'prompt_frame') and self.prompt_frame:
                self.prompt_frame.pack(fill=tk.X, anchor=tk.W, pady=(4, 8), padx=(0, 0))
            
            # Update feedback
            if hasattr(self, 'feedback_text') and self.feedback_text:
                self._update_feedback_text("Switched to Single Command mode")
    
    def _create_model_selector_plan(self):
        """Create model selection in Plan tab."""
        self.model_var = tk.StringVar(value='Gemini Flash 2.0 (lite)')
        model_row = tk.Frame(self.plan_frame, bg='#161b22')
        model_row.pack(fill=tk.X, pady=(8, 0))
        
        model_label = tk.Label(model_row, text="Model:", font=get_inter_font(10, "bold"),
                              bg='#161b22', fg='#8b949e')
        model_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var,
                                       font=get_inter_font(10), state='readonly', width=20)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        model_values = (
            'Gemini Flash 2.0 (lite)',
            'Gemini Pro 2.5 (reasoning)',
        )
        self.model_combo['values'] = model_values
        try:
            self.model_combo.configure(height=len(model_values))
        except Exception:
            pass
        self.model_combo.current(0)
    
    def _create_prompt_selector_plan(self):
        """Create prompt template selector in Plan tab."""
        self.prompt_var = tk.StringVar(value='Sim')
        prompt_row = tk.Frame(self.plan_frame, bg='#161b22')
        prompt_row.pack(fill=tk.X, pady=(8, 0))
        
        prompt_label = tk.Label(prompt_row, text="Prompt:", font=get_inter_font(10, "bold"),
                               bg='#161b22', fg='#8b949e')
        prompt_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.prompt_combo = ttk.Combobox(prompt_row, textvariable=self.prompt_var,
                                        font=get_inter_font(10), state='readonly', width=16)
        self.prompt_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Discover available prompt templates
        self._available_prompts = {}
        try:
            try:
                from nl_control import get_available_prompts
                if callable(get_available_prompts):
                    self._available_prompts = get_available_prompts()
            except ImportError:
                self._available_prompts = {}
        except Exception:
            self._available_prompts = {}
        
        if self._available_prompts:
            prompt_names = [name.capitalize() for name in sorted(self._available_prompts.keys())]
            self.prompt_combo['values'] = prompt_names
            if 'sim' in self._available_prompts:
                default_index = prompt_names.index('Sim') if 'Sim' in prompt_names else 0
            elif 'base' in self._available_prompts:
                default_index = prompt_names.index('Base') if 'Base' in prompt_names else 0
            else:
                default_index = 0
            self.prompt_combo.current(default_index)
        else:
            self.prompt_combo['values'] = ('Sim', 'Base', 'Reasoning')
            self.prompt_combo.current(0)
    
    def _create_status_tab(self):
        """Create Status tab with status tiles."""
        # World selector at top
        self._create_world_selector_status()
        
        ttk.Separator(self.status_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=16)
        
        # Status tiles in a grid
        tiles_frame = tk.Frame(self.status_frame, bg='#161b22')
        tiles_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create 6 status tiles
        self._create_status_tile(tiles_frame, "State", "unknown", "#00d4aa", 0, 0)
        self._create_status_tile(tiles_frame, "Position", "(0.0, 0.0, 0.0°)", "#58a6ff", 0, 1)
        self._create_status_tile(tiles_frame, "Action", "idle", "#f85149", 1, 0)
        self._create_status_tile(tiles_frame, "Arm", "stowed", "#ffa657", 1, 1)
        self._create_status_tile(tiles_frame, "Gripper", "closed", "#d2a8ff", 2, 0)
        self._create_status_tile(tiles_frame, "Connection", "connected", "#39d353", 2, 1)
        
        # Control buttons
        ttk.Separator(self.status_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=16)
        self._create_control_buttons_status()
    
    def _create_status_tile(self, parent, label, value, color, row, col):
        """Create a status tile with icon, value, and health dot."""
        tile_frame = tk.Frame(parent, bg='#21262d', relief='flat', bd=1)
        tile_frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        parent.grid_columnconfigure(col, weight=1)
        
        # Icon and label
        icon_label = tk.Label(tile_frame, text="●", font=get_inter_font(12), 
                             bg='#21262d', fg=color)
        icon_label.pack(anchor=tk.W, padx=8, pady=(8, 0))
        
        # Value
        value_label = tk.Label(tile_frame, text=value, font=get_inter_font(10, "bold"),
                              bg='#21262d', fg='#f0f6fc', wraplength=150)
        value_label.pack(anchor=tk.W, padx=8, pady=(4, 0))
        
        # Label
        label_label = tk.Label(tile_frame, text=label, font=get_inter_font(9),
                              bg='#21262d', fg='#8b949e')
        label_label.pack(anchor=tk.W, padx=8, pady=(4, 8))
        
        # Health dot in top right corner (green by default)
        health_dot = tk.Label(tile_frame, text="●", font=get_inter_font(8),
                            bg='#21262d', fg='#39d353')
        health_dot.place(relx=1.0, x=-12, y=8, anchor='ne')
        
        # Store references for updates
        tile_dict = {'frame': tile_frame, 'value': value_label, 'health': health_dot, 'icon': icon_label}
        if label == "State":
            self.state_tile = tile_dict
            self.state_label = value_label
        elif label == "Position":
            self.position_tile = tile_dict
            self.position_label = value_label
        elif label == "Action":
            self.action_tile = tile_dict
            self.action_label = value_label
        elif label == "Arm":
            self.arm_tile = tile_dict
            self.arm_label = value_label
        elif label == "Gripper":
            self.gripper_tile = tile_dict
            self.gripper_label = value_label
        elif label == "Connection":
            self.connection_tile = tile_dict
            self.connection_label = value_label
    
    def _create_world_selector_status(self):
        """Create world selection in Status tab."""
        world_row = tk.Frame(self.status_frame, bg='#161b22')
        world_row.pack(fill=tk.X, pady=(0, 0))
        
        world_label = tk.Label(world_row, text="World:", font=get_inter_font(10, "bold"),
                              bg='#161b22', fg='#8b949e')
        world_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.world_var = tk.StringVar(value=self.world_name)
        self.world_combo = ttk.Combobox(world_row, textvariable=self.world_var,
                                       font=get_inter_font(10), state='readonly', width=20)
        self.world_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Populate world dropdown - only show custom worlds
        world_list = self.world_manager.get_world_list()
        # Filter to only custom_sim and custom_real
        custom_worlds = [
            ("custom_sim", "with three checkpoints", ""),
            ("custom_sim_4CP", "with four checkpoints", ""),
            ("custom_small_map", "small map", ""),
        ]
        world_options = [f"{world_id}. {name}" for world_id, name, _ in custom_worlds]
        self.world_combo['values'] = world_options
        
        # Set current selection based on world_id
        current_display = f"{self.current_world_id}. {self.world_name}"
        if self.current_world_id == "custom_sim":
            current_display = "custom_sim. Simulation World"
        elif self.current_world_id == "custom_real":
            current_display = "custom_real. Real World"
        self.world_combo.set(current_display)
    
    def _create_control_buttons_status(self):
        """Create control buttons in Status tab."""
        control_button_frame = tk.Frame(self.status_frame, bg='#161b22')
        control_button_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.reset_button = tk.Button(control_button_frame, text="Reset", 
                                     command=self.reset_simulation,
                                     bg='#21262d', fg='#f0f6fc', 
                                     font=get_inter_font(10, "bold"),
                                     activebackground='#30363d', 
                                     activeforeground='#f0f6fc',
                                     relief='flat', bd=0, padx=16, pady=8, 
                                     highlightthickness=0)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.stop_button = tk.Button(control_button_frame, text="Stop", 
                                    command=self._on_stop,
                                    bg='#f85149', fg='#f0f6fc', 
                                    font=get_inter_font(10, "bold"),
                                    activebackground='#ff7b72', 
                                    activeforeground='#f0f6fc',
                                    relief='flat', bd=0, padx=16, pady=8, 
                                    highlightthickness=0)
        self.stop_button.pack(side=tk.LEFT)
    
    def _create_logs_tab(self):
        """Create Logs tab with console output."""
        # Create scrollable text widget for logs
        self.feedback_text = tk.Text(self.logs_frame, 
                                   font=get_mono_font(10), bg='#0d1117', fg='#f0f6fc',
                                   wrap=tk.WORD, 
                                   insertbackground='#f0f6fc', selectbackground='#404040',
                                   relief='flat', bd=0, padx=8, pady=8)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.logs_frame, orient=tk.VERTICAL, 
                                command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack text widget and scrollbar
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert initial text
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        initial_text = f"[{timestamp}] System initialized - Ready for commands\n"
        self.feedback_text.insert(tk.END, initial_text)
        self.feedback_text.config(state=tk.DISABLED)  # Make read-only
        
        # Bind mouse wheel scrolling
        if self.feedback_text:
            self.feedback_text.bind("<MouseWheel>", self._on_mousewheel)
    
    def _create_fixed_control_bar(self):
        """Create fixed control bar at bottom of window."""
        self.control_bar_frame = tk.Frame(self.root, bg='#21262d', height=56, relief='flat')
        self.control_bar_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.control_bar_frame.pack_propagate(False)
        # Note: row 1 is already configured in gui_core.py
        
        # Inner frame for content
        inner_frame = tk.Frame(self.control_bar_frame, bg='#21262d')
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        
        # Play button
        self.play_button = tk.Button(inner_frame, text="> Play", 
                                    command=self._on_play,
                                    bg='#39d353', fg='white', 
                                    font=get_inter_font(10, "bold"),
                                    activebackground='#2ea043', 
                                    activeforeground='white',
                                    relief='flat', bd=0, padx=16, pady=8, 
                                    highlightthickness=0)
        self.play_button.pack(side=tk.LEFT, padx=(0, 8))
        
        # Pause button
        self.pause_button = tk.Button(inner_frame, text="|| Pause", 
                                     command=self._on_pause,
                                     bg='#FF9F0A', fg='white', 
                                     font=get_inter_font(10, "bold"),
                                     activebackground='#cc7f08', 
                                     activeforeground='white',
                                     relief='flat', bd=0, padx=16, pady=8, 
                                     highlightthickness=0)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))
        
        # Execute button (for real robot execution)
        self.execute_button = tk.Button(inner_frame, text=">> Execute", 
                                       command=self._on_execute,
                                       bg='#5E5CE6', fg='white', 
                                       font=get_inter_font(10, "bold"),
                                       activebackground='#4c4abf', 
                                       activeforeground='white',
                                       relief='flat', bd=0, padx=16, pady=8, 
                                       highlightthickness=0)
        self.execute_button.pack(side=tk.LEFT, padx=(0, 8))
        
        # Toggle path visibility button (switches between Clear Path and Show Path)
        self.path_visible = True  # Track path visibility state
        self.toggle_path_button = tk.Button(inner_frame, text="Hide Path", 
                                            command=self._on_toggle_path,
                                            bg='#8b949e', fg='white', 
                                            font=get_inter_font(10, "bold"),
                                            activebackground='#6e7681', 
                                            activeforeground='white',
                                            relief='flat', bd=0, padx=16, pady=8, 
                                            highlightthickness=0)
        self.toggle_path_button.pack(side=tk.LEFT, padx=(0, 8))
        
        # Abort button
        self.abort_button = tk.Button(inner_frame, text="X Abort", 
                                    command=self._on_abort,
                                    bg='#f85149', fg='white', 
                                    font=get_inter_font(10, "bold"),
                                    activebackground='#da3633', 
                                    activeforeground='white',
                                    relief='flat', bd=0, padx=16, pady=8, 
                                    highlightthickness=0)
        self.abort_button.pack(side=tk.RIGHT)
    
    def _create_plan_preview_card(self, plan_actions):
        """Create plan preview card with numbered steps."""
        if self.plan_card_frame:
            self.plan_card_frame.destroy()
        
        self.current_plan_actions = plan_actions
        
        # Card frame with shadow effect (simulated with border)
        self.plan_card_frame = tk.Frame(self.plan_frame, bg='#21262d', relief='flat', bd=1)
        self.plan_card_frame.pack(fill=tk.X, pady=(16, 0))
        
        # Card title
        title_label = tk.Label(self.plan_card_frame, text="Plan", 
                              font=get_inter_font(12, "bold"), bg='#21262d', fg='#f0f6fc')
        title_label.pack(anchor=tk.W, padx=16, pady=(16, 8))
        
        # Numbered steps list
        steps_frame = tk.Frame(self.plan_card_frame, bg='#21262d')
        steps_frame.pack(fill=tk.X, padx=16, pady=(0, 8))
        
        self.plan_steps_list = []
        for i, action in enumerate(plan_actions, 1):
            step_text = f"{i}. {action}"
            step_label = tk.Label(steps_frame, text=step_text, 
                                 font=get_inter_font(10), bg='#21262d', fg='#8b949e',
                                 anchor='w', justify='left')
            step_label.pack(anchor=tk.W, pady=2)
            self.plan_steps_list.append(step_label)
        
        # Action buttons
        buttons_frame = tk.Frame(self.plan_card_frame, bg='#21262d')
        buttons_frame.pack(fill=tk.X, padx=16, pady=(0, 16))
        
        accept_btn = tk.Button(buttons_frame, text="Accept", 
                             command=self._on_accept_plan,
                             bg='#39d353', fg='white', font=get_inter_font(10, "bold"),
                             activebackground='#2ea043', activeforeground='white',
                             relief='flat', bd=0, padx=16, pady=8, highlightthickness=0)
        accept_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        edit_btn = tk.Button(buttons_frame, text="Edit", 
                           command=self._on_edit_plan,
                           bg='#58a6ff', fg='white', font=get_inter_font(10, "bold"),
                           activebackground='#1f6feb', activeforeground='white',
                           relief='flat', bd=0, padx=16, pady=8, highlightthickness=0)
        edit_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        clear_btn = tk.Button(buttons_frame, text="Clear", 
                            command=self._on_clear_plan,
                            bg='#8b949e', fg='white', font=get_inter_font(10, "bold"),
                            activebackground='#6e7681', activeforeground='white',
                            relief='flat', bd=0, padx=16, pady=8, highlightthickness=0)
        clear_btn.pack(side=tk.LEFT)
    
    def _on_toggle_presenter_mode(self):
        """Handle presenter mode toggle."""
        if self.gui_core:
            self.gui_core.presenter_mode = self.presenter_mode_var.get()
            self._apply_presenter_mode()
    
    def _apply_presenter_mode(self):
        """Apply presenter mode styling."""
        scale = 1.4 if self.presenter_mode_var.get() else 1.0
        
        # Hide/show Status and Logs tabs (only show Plan in presenter mode)
        if self.presenter_mode_var.get():
            # Hide Status and Logs tabs
            for i in range(self.notebook.index("end")):
                tab_text = self.notebook.tab(i, "text")
                if tab_text in ["Status", "Logs"]:
                    self.notebook.hide(i)
            # Make sure Plan tab is selected
            self.notebook.select(0)
        else:
            # Show all tabs
            for i in range(self.notebook.index("end")):
                tab_text = self.notebook.tab(i, "text")
                if tab_text == "Status":
                    try:
                        self.notebook.add(self.status_frame, text="Status")
                    except:
                        pass  # Already added
                elif tab_text == "Logs":
                    try:
                        self.notebook.add(self.logs_frame, text="Logs")
                    except:
                        pass  # Already added
        
        # Update font sizes (would need to recreate widgets for full effect)
        # For now, just toggle tab visibility
    
    def _on_generate_plan(self):
        """Handle generate plan button click."""
        # Hide previous plan status when generating a new plan
        if hasattr(self, 'update_plan_status'):
            self.update_plan_status(show=False)
        
        # Capture and display the prompt before sending
        if hasattr(self, 'prompt_entry') and self.prompt_entry:
            if isinstance(self.prompt_entry, tk.Text):
                prompt_text = self.prompt_entry.get("1.0", tk.END).strip()
            else:
                prompt_text = self.prompt_entry.get().strip()
            if prompt_text and hasattr(self, 'last_prompt_label'):
                self.last_prompt_label.config(text=f"Last prompt: {prompt_text}", fg='#c9d1d9')
        
        # Forward to handlers
        if hasattr(self, 'gui_core') and self.gui_core and hasattr(self.gui_core, 'handlers'):
            # Use the same handler as Enter key
            self.gui_core.handlers._on_send_single_command()
    
    def _on_accept_plan(self):
        """Handle accept plan button click."""
        # This will be handled by handlers
        pass
    
    def _on_edit_plan(self):
        """Handle edit plan button click."""
        # This will be handled by handlers
        pass
    
    def _on_clear_plan(self):
        """Handle clear plan button click."""
        if self.plan_card_frame:
            self.plan_card_frame.destroy()
            self.plan_card_frame = None
        self.current_plan_actions = []
        # Hide plan status indicator
        if hasattr(self, 'update_plan_status'):
            self.update_plan_status(show=False)
    
    def _on_play(self):
        """Handle play button click."""
        from message_bus import rospy
        rospy.loginfo("DEBUG: Play button clicked")
        
        # Get the handlers from gui_core
        if hasattr(self, 'gui_core') and self.gui_core and hasattr(self.gui_core, 'handlers'):
            handlers = self.gui_core.handlers
            preview_manager = handlers.preview_manager
            
            rospy.loginfo(f"DEBUG: _preview_frames length: {len(preview_manager._preview_frames) if preview_manager._preview_frames else 0}")
            rospy.loginfo(f"DEBUG: _preview_animation_running: {preview_manager._preview_animation_running}")
            rospy.loginfo(f"DEBUG: _preview_paused: {preview_manager._preview_paused}")
            
            # Check if animation is paused - if so, resume it
            if preview_manager._preview_animation_running and preview_manager._preview_paused:
                rospy.loginfo("DEBUG: Resuming paused animation")
                preview_manager.resume_preview_animation()
            # Otherwise start a new animation
            elif preview_manager._preview_frames and not preview_manager._preview_animation_running:
                rospy.loginfo("DEBUG: Starting new animation")
                preview_manager.start_preview_animation()
            # If no preview loaded, check if we're in batch mode with a selection
            else:
                rospy.loginfo("DEBUG: No preview frames or animation already running")
                mode = self.control_mode.get()
                rospy.loginfo(f"DEBUG: Current mode: {mode}")
                if mode == "Batch Commands" and hasattr(self, 'inputs_tree'):
                    # Try to preview the selected batch command
                    selection = self.inputs_tree.selection()
                    if selection:
                        handlers._on_preview_trajectory()
        else:
            rospy.loginfo("DEBUG: gui_core or handlers not available")
    
    def _on_pause(self):
        """Handle pause button click."""
        if hasattr(self, 'gui_core') and self.gui_core and hasattr(self.gui_core, 'handlers'):
            preview_manager = self.gui_core.handlers.preview_manager
            if preview_manager._preview_animation_running and not preview_manager._preview_paused:
                preview_manager.pause_preview_animation()
    
    def _on_execute(self):
        """Handle execute button click - approve and execute plan on real robot."""
        if hasattr(self, 'gui_core') and self.gui_core and hasattr(self.gui_core, 'handlers'):
            handlers = self.gui_core.handlers
            # Call the approve handler to execute the plan
            handlers._on_approve()
    
    def _on_toggle_path(self):
        """Handle toggle path button click - switches between showing and hiding the path."""
        if hasattr(self, 'gui_core') and self.gui_core:
            # Get preview_manager from either handlers or gui_core
            preview_manager = None
            if hasattr(self.gui_core, 'handlers') and self.gui_core.handlers:
                preview_manager = getattr(self.gui_core.handlers, 'preview_manager', None)
            if not preview_manager and hasattr(self.gui_core, 'preview_manager'):
                preview_manager = self.gui_core.preview_manager
            
            if preview_manager:
                # Toggle visibility state
                self.path_visible = not self.path_visible
                
                # Store the path data temporarily if hiding
                if not self.path_visible:
                    # Hide the path by temporarily storing it and clearing display
                    # Don't actually clear the data, just don't pass it to draw_world
                    preview_manager._path_hidden = True
                    # Delete the path from canvas
                    preview_manager.canvas.delete("preview_path")
                else:
                    # Show the path again
                    preview_manager._path_hidden = False
                
                # Update button text
                if self.path_visible:
                    self.toggle_path_button.config(text="Hide Path")
                else:
                    self.toggle_path_button.config(text="Show Path")
            
            # Update display to show/hide the path
            self.gui_core.update_display()
    
    def _on_abort(self):
        """Handle abort button click."""
        if hasattr(self, 'gui_core') and self.gui_core and hasattr(self.gui_core, 'handlers'):
            preview_manager = self.gui_core.handlers.preview_manager
            if preview_manager._preview_animation_running:
                preview_manager.cancel_preview_animation()
    
    def _setup_styling(self):
        """Setup widget styling."""
        # Configure themed widgets (comboboxes & table)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                        fieldbackground='#0d1117',
                        background='#0d1117',
                        foreground='#f0f6fc',
                        borderwidth=1,
                        relief='flat',
                        font=get_inter_font(10))
        style.map('TCombobox',
                 fieldbackground=[('readonly', '#0d1117')],
                 background=[('readonly', '#0d1117')],
                 foreground=[('readonly', '#f0f6fc')])
        style.configure('InputsTree.Treeview',
                        background='#0d1117',
                        fieldbackground='#0d1117',
                        foreground='#f0f6fc',
                        bordercolor='#30363d',
                        rowheight=26,
                        relief='flat')
        style.configure('InputsTree.Treeview.Heading',
                        background='#21262d',
                        foreground='#f0f6fc',
                        font=get_inter_font(10, "bold"))
        style.map('InputsTree.Treeview',
                 background=[('selected', '#1f6feb')],
                 foreground=[('selected', '#ffffff')])
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in feedback text."""
        self.feedback_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def reset_simulation(self):
        """Reset the simulation."""
        pass
    
    def _on_stop(self):
        """Handle stop button click."""
        pass
    
    # Legacy methods for backward compatibility
    def _create_status_panel(self):
        """Legacy method - no longer used."""
        pass
    
    def _create_bottom_panels(self):
        """Legacy method - no longer used."""
        pass
    
    def _create_model_selector(self):
        """Legacy method - replaced by _create_model_selector_plan."""
        pass
    
    def _create_prompt_selector(self):
        """Legacy method - replaced by _create_prompt_selector_plan."""
        pass
    
    def _create_control_buttons(self):
        """Legacy method - replaced by _create_control_buttons_status."""
        pass
    
    def _create_mode_switcher(self):
        """Legacy method - kept for compatibility."""
        pass
    
    def _create_batch_controls(self):
        """Legacy method - kept for compatibility."""
        pass
    
    def _create_single_controls(self):
        """Legacy method - kept for compatibility."""
        # Single command entry (for backward compatibility)
        self.command_entry = self.prompt_entry  # Alias
        self.send_button = self.generate_plan_button  # Alias
        self.clear_button = None  # Will be created if needed
    
    def _on_load_inputs(self):
        """Handle load inputs button click."""
        pass
    
    def _on_show_all_previews(self):
        """Handle show all previews button click."""
        pass
    
    def _on_toggle_heatmap(self):
        """Handle heatmap toggle button click."""
        pass
    
    def _on_preview_trajectory(self):
        """Handle preview trajectory button click."""
        pass
    
    def _on_toggle_wall_enforcement(self):
        """Handle wall enforcement toggle."""
        pass
    
    def _on_toggle_strict_scoring(self):
        """Handle strict scoring toggle."""
        pass
    
    def _on_refresh_input_files(self):
        """Handle refreshing the inputs file dropdown."""
        pass
    
    def _on_copy_and_clear_user_data(self):
        """Handle copy and clear user_data button click."""
        pass
    
    def _on_save_as_baseline(self):
        """Handle save as baseline button click."""
        pass
    
    def _discover_baselines(self):
        """Discover available baseline files."""
        import json
        baseline_names = []
        try:
            baselines_dir = Path(__file__).resolve().parents[1] / 'baselines'
            for f in sorted(baselines_dir.glob('baseline_*.json')):
                try:
                    with open(f, 'r') as bf:
                        data = json.load(bf)
                        name = data.get('name', f.stem.replace('_', ' ').title())
                        baseline_names.append(name)
                except Exception:
                    baseline_names.append(f.stem.replace('_', ' ').title())
        except Exception:
            pass
        return baseline_names if baseline_names else ['Baseline 1', 'Baseline 2', 'Baseline 3']
    
    def _on_send_single_command(self, event=None):
        """Handle send single command."""
        pass
    
    def _on_clear_command(self):
        """Handle clear command button click."""
        pass
    
    def _on_single_preview_trajectory(self):
        """Handle single preview trajectory button click."""
        pass
    
    def _on_single_approve(self):
        """Handle single approve button click."""
        pass
    
    def _on_single_dismiss(self):
        """Handle single dismiss button click."""
        pass
    
    def _on_approve(self):
        """Handle approve button click."""
        pass
    
    def _update_feedback_text(self, message):
        """Update the feedback text widget with a message."""
        if self.feedback_text:
            self.feedback_text.config(state=tk.NORMAL)
            self.feedback_text.insert(tk.END, f"{message}\n")
            self.feedback_text.see(tk.END)
            self.feedback_text.config(state=tk.DISABLED)
    
    def _show_batch_mode(self):
        """Show batch mode interface elements."""
        # Legacy method - kept for compatibility
        pass
    
    def _show_single_mode(self):
        """Show single command mode interface elements."""
        # Legacy method - kept for compatibility
        pass
    
    def _on_mode_change(self, event=None):
        """Handle control mode change between batch and single command modes."""
        # Forward to new plan tab mode change handler
        if hasattr(self, '_on_mode_change_plan'):
            self._on_mode_change_plan(event)
    
    def update_status_tile(self, tile_name, value, health_status="ok"):
        """Update a status tile with new value and health status.
        
        Args:
            tile_name: One of "State", "Position", "Action", "Arm", "Gripper", "Connection"
            value: New value to display
            health_status: "ok" (green), "waiting" (yellow), "fault" (red)
        """
        tile_attr = f"{tile_name.lower()}_tile"
        if hasattr(self, tile_attr):
            tile = getattr(self, tile_attr)
            tile['value'].config(text=str(value))
            
            # Update health dot color
            colors = {"ok": "#39d353", "waiting": "#FF9F0A", "fault": "#f85149"}
            color = colors.get(health_status, "#39d353")
            tile['health'].config(fg=color)
    
    def update_state(self, state):
        """Update robot state."""
        self.update_status_tile("State", state, "ok" if state != "unknown" else "waiting")
    
    def update_position(self, position):
        """Update robot position."""
        self.update_status_tile("Position", position, "ok")
    
    def update_action(self, action):
        """Update current action."""
        status = "waiting" if action == "idle" else "ok"
        self.update_status_tile("Action", action, status)
    
    def update_arm(self, arm_status):
        """Update arm status."""
        self.update_status_tile("Arm", arm_status, "ok")
    
    def update_gripper(self, gripper_status):
        """Update gripper status."""
        self.update_status_tile("Gripper", gripper_status, "ok")
    
    def update_connection(self, connected):
        """Update connection status."""
        status = "ok" if connected else "fault"
        value = "connected" if connected else "disconnected"
        self.update_status_tile("Connection", value, status)
    
    def _animate_loading_dot(self):
        """Animate the loading dot by rotating a green dot around a circle."""
        if not hasattr(self, 'plan_status_canvas') or not self.plan_status_canvas:
            return
        
        if not self.plan_status_loading:
            return
        
        # Clear canvas
        self.plan_status_canvas.delete("all")
        
        # Draw circle border
        size = 16
        center_x, center_y = 16, 12
        radius = size // 2
        self.plan_status_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline='#4a5568', width=1, fill='#0d1117'
        )
        
        # Calculate rotating dot position (around the circle perimeter)
        import math
        orbit_radius = radius + 4  # Slightly outside the circle
        angle_rad = math.radians(self.plan_status_angle)
        dot_x = center_x + orbit_radius * math.cos(angle_rad)
        dot_y = center_y + orbit_radius * math.sin(angle_rad)
        
        # Draw rotating green dot
        dot_size = 3
        self.plan_status_canvas.create_oval(
            dot_x - dot_size, dot_y - dot_size,
            dot_x + dot_size, dot_y + dot_size,
            fill='#00ff88', outline='#00ff88'
        )
        
        # Update angle for next frame
        self.plan_status_angle = (self.plan_status_angle + 15) % 360
        
        # Schedule next frame
        self.plan_status_animation_id = self.root.after(50, self._animate_loading_dot)
    
    def _stop_loading_animation(self):
        """Stop the loading animation."""
        self.plan_status_loading = False
        if self.plan_status_animation_id:
            self.root.after_cancel(self.plan_status_animation_id)
            self.plan_status_animation_id = None
    
    def update_plan_status(self, status_text="Plan generated", show=True, is_error=False, loading=False):
        """Update the plan status dot indicator with timestamp.
        
        Args:
            status_text: Text to display (not used in simple dot version, kept for compatibility)
            show: Whether to show the status (True) or hide it (False)
            is_error: Whether this is an error message (True) or success (False)
            loading: Whether to show loading animation (True) or static state (False)
        """
        if not hasattr(self, 'plan_status_canvas') or not self.plan_status_canvas:
            return
        
        from datetime import datetime
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        
        # Stop loading animation if not loading
        if not loading:
            self._stop_loading_animation()
        
        if loading:
            # Start loading animation
            self.plan_status_loading = True
            self.plan_status_angle = 0
            if hasattr(self, 'status_dot_frame') and self.status_dot_frame:
                self.status_dot_frame.config(bg='#0d1117')
            if hasattr(self, 'plan_status_time') and self.plan_status_time:
                self.plan_status_time.config(text="Loading...", fg='#8b949e')
            self._animate_loading_dot()
        elif show:
            # Stop any loading animation
            self._stop_loading_animation()
            
            # Update dot color based on status with futuristic styling
            self.plan_status_canvas.delete("all")
            
            if is_error:
                # Red pulsing dot for errors (futuristic)
                center_x, center_y = 16, 12
                self.plan_status_canvas.create_oval(
                    center_x - 8, center_y - 8,
                    center_x + 8, center_y + 8,
                    fill='#ff4444', outline='#ff4444'
                )
                # Update container background for glow effect
                if hasattr(self, 'status_dot_frame') and self.status_dot_frame:
                    self.status_dot_frame.config(bg='#1a0d0d')
                if hasattr(self, 'plan_status_time') and self.plan_status_time:
                    self.plan_status_time.config(text=f"Failed {timestamp}", fg='#ff6666')
            else:
                # Green glowing dot for success (futuristic)
                center_x, center_y = 16, 12
                self.plan_status_canvas.create_oval(
                    center_x - 8, center_y - 8,
                    center_x + 8, center_y + 8,
                    fill='#00ff88', outline='#00ff88'
                )
                # Update container background for glow effect
                if hasattr(self, 'status_dot_frame') and self.status_dot_frame:
                    self.status_dot_frame.config(bg='#0d1a14')
                if hasattr(self, 'plan_status_time') and self.plan_status_time:
                    self.plan_status_time.config(text=f"Ready {timestamp}", fg='#00cc99')
        else:
            # Stop any loading animation
            self._stop_loading_animation()
            
            # Grey inactive dot when no plan (futuristic)
            self.plan_status_canvas.delete("all")
            center_x, center_y = 16, 12
            self.plan_status_canvas.create_oval(
                center_x - 6, center_y - 6,
                center_x + 6, center_y + 6,
                fill='#4a5568', outline='#4a5568'
            )
            # Reset container background
            if hasattr(self, 'status_dot_frame') and self.status_dot_frame:
                self.status_dot_frame.config(bg='#0d1117')
            if hasattr(self, 'plan_status_time') and self.plan_status_time:
                self.plan_status_time.config(text="", fg='#8b949e')
