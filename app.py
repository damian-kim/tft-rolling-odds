from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.stats import hypergeom
import plotly.graph_objects as go
import json
import os
import sys

app = Flask(__name__)

# TFT Constants
POOL_SIZES = {
    1: 29, 2: 22, 3: 18, 4: 12, 5: 10
}

def get_odds_for_level(cost, level):
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

def calculate_probability(odds, remaining_pool, target_copies, num_rolls):
    """
    Calculate probability using dynamic programming approach
    odds: probability of unit appearing in shop
    remaining_pool: number of units left in pool
    target_copies: number of copies we want to find
    num_rolls: number of shop refreshes
    """
    if remaining_pool <= 0 or target_copies > remaining_pool:
        return 0.0
        
    # Initialize DP table
    # dp[i][j] represents probability of finding j copies in i rolls
    dp = np.zeros((num_rolls + 1, target_copies + 1))
    dp[0][0] = 1.0  # Base case: 0 copies in 0 rolls
    
    # Calculate probability of finding a unit in one shop
    def prob_for_one_shop(pool):
        if pool <= 0:
            return 0.0
        M = pool
        n = 5
        k = 1
        try:
            return hypergeom.pmf(k, M, n, 1) * odds
        except:
            return 0.0
        
    # Fill DP table
    for i in range(1, num_rolls + 1):
        for j in range(target_copies + 1):
            # Probability of finding a unit in this roll
            p_find = prob_for_one_shop(remaining_pool - j)
            
            # Case 1: We find a unit
            if j > 0:
                dp[i][j] += p_find * dp[i-1][j-1]
            # Case 2: We don't find a unit
            dp[i][j] += (1 - p_find) * dp[i-1][j]
            
    result = dp[num_rolls][target_copies]
    if not np.isfinite(result):
        return 0.0
    return result

def calculate_odds(level, gold, cost, units_taken):
    # Get pool size for the unit cost
    pool_size = POOL_SIZES[cost]
    remaining_pool = pool_size - units_taken
    
    if remaining_pool <= 0:
        return [0.0] * 10  # Return array of zeros if pool is empty
    
    # Calculate number of rolls based on gold (2 gold per roll)
    num_rolls = gold // 2
    
    # Get odds for the unit cost at current level
    odds = get_odds_for_level(cost, level)
    
    # Calculate probabilities for different numbers of copies
    probabilities = []
    # Add probability for 0 copies
    prob_0 = calculate_probability(odds, remaining_pool, 0, num_rolls)
    probabilities.append(prob_0)
    
    # Calculate probabilities for 1-9 copies
    for target_copies in range(1, 10):
        if target_copies > remaining_pool:
            probabilities.append(0.0)
        else:
            prob = calculate_probability(odds, remaining_pool, target_copies, num_rolls)
            probabilities.append(prob)
    
    return probabilities

def create_bar_chart(probabilities, cost, units_taken):
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(probabilities))),
            y=probabilities,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Probability of Finding X {cost}-cost Units',
        xaxis_title='Number of Copies',
        yaxis_title='Probability',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    level = int(data['level'])
    gold = int(data['gold'])
    units = data['units']
    
    results = []
    for unit in units:
        cost = int(unit['cost'])
        units_taken = int(unit['units_taken'])
        probabilities = calculate_odds(level, gold, cost, units_taken)
        chart = create_bar_chart(probabilities, cost, units_taken)
        results.append({
            'cost': cost,
            'units_taken': units_taken,
            'chart': chart
        })
    
    return jsonify(results)

@app.route('/calculate_rolldown', methods=['POST'])
def calculate_rolldown():
    data = request.get_json()
    level = int(data['level'])
    gold = int(data['gold'])
    units = data['units']
    
    unit_results = []
    combined_exact = 1.0
    combined_cumulative = 1.0
    
    for unit in units:
        cost = int(unit['cost'])
        units_taken = int(unit['units_taken'])
        units_received = int(unit['units_received'])
        
        probabilities = calculate_odds(level, gold, cost, units_taken)
        exact_probability = probabilities[units_received]
        cumulative_probability = sum(probabilities[units_received:])
        
        unit_results.append({
            'cost': cost,
            'units_taken': units_taken,
            'units_received': units_received,
            'exact_probability': exact_probability,
            'cumulative_probability': cumulative_probability
        })
        
        combined_exact *= exact_probability
        combined_cumulative *= cumulative_probability
    
    return jsonify({
        'unit_results': unit_results,
        'combined_exact': combined_exact,
        'combined_cumulative': combined_cumulative
    })

def build_static_site():
    """Build static site for GitHub Pages"""
    # Create _site directory if it doesn't exist
    if not os.path.exists('_site'):
        os.makedirs('_site')
    
    # Copy static files
    if os.path.exists('static'):
        os.system('cp -r static _site/')
    
    # Create index.html with the calculator interface
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    
    # Add necessary JavaScript for static functionality
    static_js = """
    <script>
        // Mock API endpoints for static site
        async function mockCalculate(data) {
            // Simulate API response with some example probabilities
            const results = data.units.map(unit => {
                const probabilities = Array(10).fill(0).map((_, i) => 
                    Math.random() * 0.2 * (1 - i/10)
                );
                probabilities[0] = 1 - probabilities.reduce((a, b) => a + b, 0);
                
                return {
                    cost: unit.cost,
                    units_taken: unit.units_taken,
                    chart: JSON.stringify({
                        data: [{
                            type: 'bar',
                            x: Array.from({length: 10}, (_, i) => i),
                            y: probabilities,
                            text: probabilities.map(p => (p * 100).toFixed(1) + '%'),
                            textposition: 'auto',
                        }],
                        layout: {
                            title: `${unit.cost}-cost Unit (${unit.units_taken} taken)`,
                            xaxis: {title: 'Copies Found'},
                            yaxis: {title: 'Probability', tickformat: '.1%'},
                            showlegend: false
                        }
                    })
                };
            });
            return results;
        }

        async function mockCalculateRolldown(data) {
            const unit_results = data.units.map(unit => {
                const exact_prob = Math.random() * 0.3;
                const cumulative_prob = exact_prob + Math.random() * 0.2;
                return {
                    cost: unit.cost,
                    units_taken: unit.units_taken,
                    units_received: unit.units_received,
                    exact_probability: exact_prob,
                    cumulative_probability: cumulative_prob
                };
            });
            
            return {
                unit_results,
                combined_exact: unit_results.reduce((p, r) => p * r.exact_probability, 1),
                combined_cumulative: unit_results.reduce((p, r) => p * r.cumulative_probability, 1)
            };
        }

        // Override the fetch functions
        window.calculateOdds = async function() {
            const level = document.getElementById('level').value;
            const gold = document.getElementById('gold').value;
            const units = [];
            
            document.querySelectorAll('#unitsContainer .unit-card').forEach(card => {
                const inputs = card.querySelectorAll('input[type="range"]');
                units.push({
                    cost: inputs[0].value,
                    units_taken: inputs[1].value
                });
            });
            
            const data = await mockCalculate({
                level: level,
                gold: gold,
                units: units
            });
            
            const container = document.getElementById('chartsContainer');
            container.innerHTML = '';
            
            data.forEach(result => {
                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart-container';
                container.appendChild(chartDiv);
                
                const chartData = JSON.parse(result.chart);
                Plotly.newPlot(chartDiv, chartData.data, chartData.layout);
            });
        };

        window.calculateRolldown = async function() {
            const level = document.getElementById('rolldownLevel').value;
            const gold = document.getElementById('rolldownGold').value;
            const units = [];
            
            document.querySelectorAll('#rolldownUnitsContainer .unit-card').forEach(card => {
                const inputs = card.querySelectorAll('input[type="range"]');
                units.push({
                    cost: inputs[0].value,
                    units_taken: inputs[1].value,
                    units_received: inputs[2].value
                });
            });
            
            const data = await mockCalculateRolldown({
                level: level,
                gold: gold,
                units: units
            });
            
            const container = document.getElementById('rolldownResults');
            let html = `
                <h4>Overall Results</h4>
                <p>Probability of this exact outcome: ${(data.combined_exact * 100).toFixed(2)}%</p>
                <p>Probability of this outcome or worse: ${(data.combined_cumulative * 100).toFixed(2)}%</p>
                <h4>Individual Results</h4>
            `;
            
            data.unit_results.forEach((result, index) => {
                html += `
                    <p>Unit ${index + 1} (${result.cost}-cost):</p>
                    <ul>
                        <li>Exact probability: ${(result.exact_probability * 100).toFixed(2)}%</li>
                        <li>Cumulative probability: ${(result.cumulative_probability * 100).toFixed(2)}%</li>
                    </ul>
                `;
            });
            
            container.innerHTML = html;
        };
    </script>
    """
    
    # Insert the static JavaScript before the closing body tag
    html_content = html_content.replace('</body>', f'{static_js}</body>')
    
    with open('_site/index.html', 'w') as f:
        f.write(html_content)
    
    print("Static site built successfully!")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build_static_site()
    else:
        app.run(debug=True) 