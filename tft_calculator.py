import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import hypergeom
from scipy.stats import multinomial

class UnitInput:
    def __init__(self, parent, row):
        self.frame = ttk.LabelFrame(parent, text=f"Unit {row+1}", padding="5")
        self.frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Cost slider
        ttk.Label(self.frame, text="Cost:").grid(row=0, column=0, padx=5)
        self.cost = tk.IntVar(value=1)
        self.cost_slider = ttk.Scale(self.frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                                   variable=self.cost, length=200)
        self.cost_slider.grid(row=0, column=1, padx=5)
        self.cost_label = ttk.Label(self.frame, text="1")
        self.cost_label.grid(row=0, column=2, padx=5)
        self.cost_slider.configure(command=self.update_cost_label)
        
        # Units taken slider
        ttk.Label(self.frame, text="Units Taken:").grid(row=1, column=0, padx=5)
        self.units_taken = tk.IntVar(value=0)
        self.units_slider = ttk.Scale(self.frame, from_=0, to=29, orient=tk.HORIZONTAL,
                                    variable=self.units_taken, length=200)
        self.units_slider.grid(row=1, column=1, padx=5)
        self.units_label = ttk.Label(self.frame, text="0")
        self.units_label.grid(row=1, column=2, padx=5)
        self.units_slider.configure(command=self.update_units_label)
        
    def update_cost_label(self, value):
        self.cost_label.configure(text=str(int(float(value))))
        self.update_units_slider_max()
        
    def update_units_label(self, value):
        self.units_label.configure(text=str(int(float(value))))
        
    def update_units_slider_max(self):
        cost = int(self.cost.get())
        max_units = {
            1: 29, 2: 22, 3: 18, 4: 12, 5: 10
        }[cost]
        self.units_slider.configure(to=max_units)
        if self.units_taken.get() > max_units:
            self.units_taken.set(max_units)
            self.units_label.configure(text=str(max_units))

class RolldownUnitInput:
    def __init__(self, parent, row):
        self.frame = ttk.LabelFrame(parent, text=f"Unit {row+1}", padding="5")
        self.frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Cost slider
        ttk.Label(self.frame, text="Cost:").grid(row=0, column=0, padx=5)
        self.cost = tk.IntVar(value=1)
        self.cost_slider = ttk.Scale(self.frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                                   variable=self.cost, length=150)
        self.cost_slider.grid(row=0, column=1, padx=5)
        self.cost_label = ttk.Label(self.frame, text="1")
        self.cost_label.grid(row=0, column=2, padx=5)
        self.cost_slider.configure(command=self.update_cost_label)
        
        # Units taken slider
        ttk.Label(self.frame, text="Units Taken:").grid(row=0, column=3, padx=5)
        self.units_taken = tk.IntVar(value=0)
        self.units_slider = ttk.Scale(self.frame, from_=0, to=29, orient=tk.HORIZONTAL,
                                    variable=self.units_taken, length=150)
        self.units_slider.grid(row=0, column=4, padx=5)
        self.units_label = ttk.Label(self.frame, text="0")
        self.units_label.grid(row=0, column=5, padx=5)
        self.units_slider.configure(command=self.update_units_label)
        
        # Units received slider
        ttk.Label(self.frame, text="Units Received:").grid(row=0, column=6, padx=5)
        self.units_received = tk.IntVar(value=0)
        self.received_slider = ttk.Scale(self.frame, from_=0, to=9, orient=tk.HORIZONTAL,
                                       variable=self.units_received, length=150)
        self.received_slider.grid(row=0, column=7, padx=5)
        self.received_label = ttk.Label(self.frame, text="0")
        self.received_label.grid(row=0, column=8, padx=5)
        self.received_slider.configure(command=self.update_received_label)
        
    def update_cost_label(self, value):
        self.cost_label.configure(text=str(int(float(value))))
        self.update_units_slider_max()
        
    def update_units_label(self, value):
        self.units_label.configure(text=str(int(float(value))))
        
    def update_received_label(self, value):
        self.received_label.configure(text=str(int(float(value))))
        
    def update_units_slider_max(self):
        cost = int(self.cost.get())
        max_units = {
            1: 29, 2: 22, 3: 18, 4: 12, 5: 10
        }[cost]
        self.units_slider.configure(to=max_units)
        if self.units_taken.get() > max_units:
            self.units_taken.set(max_units)
            self.units_label.configure(text=str(max_units))

class TFTCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("TFT Rolling Odds Calculator")
        self.root.geometry("1200x800")
        
        # TFT Constants
        self.POOL_SIZES = {
            1: 29, 2: 22, 3: 18, 4: 12, 5: 10
        }
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.odds_tab = ttk.Frame(self.notebook)
        self.rolldown_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.odds_tab, text='Rolling Odds')
        self.notebook.add(self.rolldown_tab, text='Rolldown Analysis')
        
        # Initialize both tabs
        self.init_odds_tab()
        self.init_rolldown_tab()
        
        # Set up window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Store figure references
        self.figures = []
        
    def on_closing(self):
        # Close all matplotlib figures
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        # Destroy the root window
        self.root.destroy()
        
    def init_odds_tab(self):
        # Create main canvas with scrollbar for odds tab
        self.odds_canvas = tk.Canvas(self.odds_tab)
        self.odds_scrollbar = ttk.Scrollbar(self.odds_tab, orient="vertical", command=self.odds_canvas.yview)
        self.odds_scrollable_frame = ttk.Frame(self.odds_canvas)
        
        # Configure canvas
        self.odds_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.odds_canvas.configure(scrollregion=self.odds_canvas.bbox("all"))
        )
        self.odds_canvas.create_window((0, 0), window=self.odds_scrollable_frame, anchor="nw")
        self.odds_canvas.configure(yscrollcommand=self.odds_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.odds_canvas.pack(side="left", fill="both", expand=True)
        self.odds_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to scroll
        self.odds_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Create main frame with padding
        self.odds_main_frame = ttk.Frame(self.odds_scrollable_frame, padding="10")
        self.odds_main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.odds_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.odds_main_frame.grid_columnconfigure(1, weight=1)
        
        # Input variables
        self.current_level = tk.IntVar(value=1)
        self.gold_amount = tk.IntVar(value=50)
        self.xp_to_next = tk.IntVar(value=0)
        
        # List to store unit inputs
        self.unit_inputs = []
        
        # Create input fields
        self.create_input_fields()
        
        # Create buttons frame
        button_frame = ttk.Frame(self.odds_main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Create buttons
        ttk.Button(button_frame, text="Add Unit", command=self.add_unit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate).pack(side=tk.LEFT, padx=5)
        
        # Create scrollable frame for graphs
        self.odds_canvas_frame = ttk.Frame(self.odds_main_frame)
        self.odds_canvas_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Add initial unit input
        self.add_unit()
        
    def init_rolldown_tab(self):
        # Create main canvas with scrollbar for rolldown tab
        self.rolldown_canvas = tk.Canvas(self.rolldown_tab)
        self.rolldown_scrollbar = ttk.Scrollbar(self.rolldown_tab, orient="vertical", command=self.rolldown_canvas.yview)
        self.rolldown_scrollable_frame = ttk.Frame(self.rolldown_canvas)
        
        # Configure canvas
        self.rolldown_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.rolldown_canvas.configure(scrollregion=self.rolldown_canvas.bbox("all"))
        )
        self.rolldown_canvas.create_window((0, 0), window=self.rolldown_scrollable_frame, anchor="nw")
        self.rolldown_canvas.configure(yscrollcommand=self.rolldown_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.rolldown_canvas.pack(side="left", fill="both", expand=True)
        self.rolldown_scrollbar.pack(side="right", fill="y")
        
        # Create main frame with padding
        self.rolldown_main_frame = ttk.Frame(self.rolldown_scrollable_frame, padding="10")
        self.rolldown_main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.rolldown_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.rolldown_main_frame.grid_columnconfigure(1, weight=1)
        
        # Input variables
        self.rolldown_level = tk.IntVar(value=8)
        self.rolldown_gold = tk.IntVar(value=50)
        
        # List to store rolldown unit inputs
        self.rolldown_unit_inputs = []
        
        # Create input fields
        self.create_rolldown_input_fields()
        
        # Create buttons frame
        button_frame = ttk.Frame(self.rolldown_main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Create buttons
        ttk.Button(button_frame, text="Add Unit", command=self.add_rolldown_unit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate_rolldown).pack(side=tk.LEFT, padx=5)
        
        # Create result label
        self.rolldown_result_label = ttk.Label(self.rolldown_main_frame, text="", font=('Arial', 12))
        self.rolldown_result_label.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Add initial unit input
        self.add_rolldown_unit()

    def _on_mousewheel(self, event):
        # Determine which canvas to scroll based on current tab
        if self.notebook.index(self.notebook.select()) == 0:
            self.odds_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            self.rolldown_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_input_fields(self):
        # Create a frame for the main inputs
        input_frame = ttk.LabelFrame(self.odds_main_frame, text="General Settings", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Current Level slider
        ttk.Label(input_frame, text="Current Level:").grid(row=0, column=0, padx=5)
        level_slider = ttk.Scale(input_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                               variable=self.current_level, length=200)
        level_slider.grid(row=0, column=1, padx=5)
        self.level_label = ttk.Label(input_frame, text="1")
        self.level_label.grid(row=0, column=2, padx=5)
        level_slider.configure(command=lambda x: self.level_label.configure(text=str(int(float(x)))))
        
        # Gold Amount slider
        ttk.Label(input_frame, text="Gold to Roll:").grid(row=1, column=0, padx=5)
        gold_slider = ttk.Scale(input_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                              variable=self.gold_amount, length=200)
        gold_slider.grid(row=1, column=1, padx=5)
        self.gold_label = ttk.Label(input_frame, text="50")
        self.gold_label.grid(row=1, column=2, padx=5)
        gold_slider.configure(command=lambda x: self.gold_label.configure(text=str(int(float(x)))))
        
        # XP to Next Level slider
        ttk.Label(input_frame, text="XP to Next Level:").grid(row=2, column=0, padx=5)
        xp_slider = ttk.Scale(input_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                            variable=self.xp_to_next, length=200)
        xp_slider.grid(row=2, column=1, padx=5)
        self.xp_label = ttk.Label(input_frame, text="0")
        self.xp_label.grid(row=2, column=2, padx=5)
        xp_slider.configure(command=lambda x: self.xp_label.configure(text=str(int(float(x)))))

    def create_rolldown_input_fields(self):
        # Create a frame for the main inputs
        input_frame = ttk.LabelFrame(self.rolldown_main_frame, text="General Settings", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Current Level slider
        ttk.Label(input_frame, text="Current Level:").grid(row=0, column=0, padx=5)
        level_slider = ttk.Scale(input_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                               variable=self.rolldown_level, length=200)
        level_slider.grid(row=0, column=1, padx=5)
        self.rolldown_level_label = ttk.Label(input_frame, text="8")
        self.rolldown_level_label.grid(row=0, column=2, padx=5)
        level_slider.configure(command=lambda x: self.rolldown_level_label.configure(text=str(int(float(x)))))
        
        # Gold Amount slider
        ttk.Label(input_frame, text="Gold to Roll:").grid(row=1, column=0, padx=5)
        gold_slider = ttk.Scale(input_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                              variable=self.rolldown_gold, length=200)
        gold_slider.grid(row=1, column=1, padx=5)
        self.rolldown_gold_label = ttk.Label(input_frame, text="50")
        self.rolldown_gold_label.grid(row=1, column=2, padx=5)
        gold_slider.configure(command=lambda x: self.rolldown_gold_label.configure(text=str(int(float(x)))))

    def add_unit(self):
        unit_input = UnitInput(self.odds_main_frame, len(self.unit_inputs) + 1)
        self.unit_inputs.append(unit_input)

    def calculate_odds(self, cost, level, gold, units_taken):
        # Get pool size for the unit cost
        pool_size = self.POOL_SIZES[cost]
        remaining_pool = pool_size - units_taken
        
        if remaining_pool <= 0:
            return [0.0] * 10  # Return array of zeros if pool is empty
        
        # Calculate number of rolls based on gold (2 gold per roll)
        num_rolls = gold // 2
        
        # Get odds for the unit cost at current level
        odds = self.get_odds_for_level(cost, level)
        
        # Calculate probabilities for different numbers of copies
        probabilities = []
        # Add probability for 0 copies
        prob_0 = self.calculate_probability(odds, remaining_pool, 0, num_rolls)
        probabilities.append(prob_0)
        
        # Calculate probabilities for 1-9 copies
        for target_copies in range(1, 10):
            if target_copies > remaining_pool:
                probabilities.append(0.0)
            else:
                prob = self.calculate_probability(odds, remaining_pool, target_copies, num_rolls)
                probabilities.append(prob)
        
        # Validate probabilities
        if not all(np.isfinite(p) for p in probabilities):
            raise ValueError("Invalid probability values calculated")
            
        return probabilities

    def get_odds_for_level(self, cost, level):
        # TFT shop odds by level and cost
        odds_table = {
            1: [100, 0, 0, 0, 0],
            2: [100, 0, 0, 0, 0],
            3: [75, 25, 0, 0, 0],
            4: [55, 30, 15, 0, 0],
            5: [45, 33, 20, 2, 0],
            6: [30, 40, 25, 5, 0],
            7: [19, 30, 40, 10, 1],
            8: [17, 24, 32, 24, 3],
            9: [15, 18, 25, 30, 12],
            10: [5, 10, 20, 40, 25]
        }
        return odds_table[level][cost-1] / 100

    def calculate_probability(self, odds, remaining_pool, target_copies, num_rolls):
        """
        Calculate probability using dynamic programming approach
        odds: probability of unit appearing in shop
        remaining_pool: number of units left in pool
        target_copies: number of copies we want to find
        num_rolls: number of shop refreshes
        """

        if remaining_pool <= 0 or target_copies > remaining_pool or target_copies > 9:
            return 0.0
            
        dp = np.zeros((num_rolls + 1, target_copies + 1, remaining_pool + 1))
        dp[0][0][remaining_pool] = 1.0  # base case
        
        # Calculate probability of finding a unit in one shop
        for r in range(1, num_rolls + 1):
            for c in range(target_copies + 1):
                for p in range(remaining_pool + 1):
                    if dp[r-1][c][p] == 0:
                        continue

                    # Chance of seeing the unit in one shop (5 slots)
                    prob_in_shop = 1 - (1 - odds * (p / self.POOL_SIZES[len(self.POOL_SIZES)])) ** 5

                    # Case 1: Find the unit (if copies and pool allow)
                    if c < target_copies and p > 0:
                        dp[r][c+1][p-1] += dp[r-1][c][p] * prob_in_shop

                    # Case 2: Don't find the unit
                    dp[r][c][p] += dp[r-1][c][p] * (1 - prob_in_shop)
        return float(np.sum(dp[num_rolls][target_copies]))

    def calculate(self):
        try:
            level = self.current_level.get()
            gold = self.gold_amount.get()
            
            # Clear previous graphs
            for widget in self.odds_canvas_frame.winfo_children():
                widget.destroy()
            
            # Clear previous figures
            for fig in self.figures:
                plt.close(fig)
            self.figures.clear()
            
            # Calculate and plot for each unit
            for i, unit_input in enumerate(self.unit_inputs):
                cost = unit_input.cost.get()
                units_taken = unit_input.units_taken.get()
                
                probabilities = self.calculate_odds(cost, level, gold, units_taken)
                
                # Validate probabilities
                if not all(np.isfinite(p) for p in probabilities):
                    raise ValueError(f"Invalid probability values calculated for unit {i+1}")
                
                # Create figure for this unit
                fig, ax = plt.subplots(figsize=(10, 4))
                self.figures.append(fig)  # Store figure reference
                copies = range(len(probabilities))
                bars = ax.bar(copies, probabilities)
                ax.set_xlabel('Number of Copies')
                ax.set_ylabel('Probability')
                ax.set_title(f'Probability of Finding X {cost}-cost Units')
                ax.set_xticks(copies)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    if np.isfinite(height):  # Only add label if height is finite
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1%}',
                               ha='center', va='bottom')
                
                # Add canvas to frame
                canvas = FigureCanvasTkAgg(fig, master=self.odds_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_rolldown_unit(self):
        unit_input = RolldownUnitInput(self.rolldown_main_frame, len(self.rolldown_unit_inputs) + 1)
        self.rolldown_unit_inputs.append(unit_input)

    def calculate_rolldown(self):
        try:
            level = self.rolldown_level.get()
            gold = self.rolldown_gold.get()
            num_rolls = gold // 2
            
            # Calculate probabilities for each unit
            unit_probs = []
            unit_cumulative_probs = []
            
            for unit_input in self.rolldown_unit_inputs:
                cost = unit_input.cost.get()
                units_taken = unit_input.units_taken.get()
                units_received = unit_input.units_received.get()
                
                # Get odds for the unit cost at current level
                odds = self.get_odds_for_level(cost, level)
                
                # Calculate probability of getting exactly units_received copies
                pool_size = self.POOL_SIZES[cost]
                remaining_pool = pool_size - units_taken
                
                if remaining_pool <= 0:
                    exact_prob = 0.0
                    cumulative_prob = 0.0
                else:
                    # Calculate exact probability
                    exact_prob = self.calculate_probability(odds, remaining_pool, units_received, num_rolls)
                    
                    # Calculate cumulative probability (this or worse)
                    cumulative_prob = 0.0
                    for i in range(units_received + 1):
                        if i <= remaining_pool:
                            cumulative_prob += self.calculate_probability(odds, remaining_pool, i, num_rolls)
                
                unit_probs.append(exact_prob)
                unit_cumulative_probs.append(cumulative_prob)
            
            # Calculate combined probabilities
            combined_exact_prob = np.prod(unit_probs)
            combined_cumulative_prob = np.prod(unit_cumulative_probs)
            
            # Update result label
            result_text = f"Probability of this exact outcome: {combined_exact_prob:.2%}\n"
            result_text += f"Probability of this outcome or worse: {combined_cumulative_prob:.2%}\n\n"
            result_text += "Individual probabilities:\n"
            for i, (unit_input, exact_prob, cumulative_prob) in enumerate(zip(self.rolldown_unit_inputs, unit_probs, unit_cumulative_probs)):
                result_text += f"Unit {i+1} ({unit_input.cost.get()}-cost):\n"
                result_text += f"  Exact probability: {exact_prob:.2%}\n"
                result_text += f"  Cumulative probability: {cumulative_prob:.2%}\n"
            
            self.rolldown_result_label.configure(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TFTCalculator(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 