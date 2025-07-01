import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product

class ExpandedConstructalExamples:
    """
    Extended examples of constructal patterns in nature and technology
    """
    
    def visualize_neural_networks(self):
        """Show how neural networks follow constructal-like patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Single neuron dendritic tree
        ax = axes[0, 0]
        self._draw_neuron(ax)
        
        # 2. Neural network connectivity
        ax = axes[0, 1]
        self._draw_neural_network(ax)
        
        # 3. Brain vasculature
        ax = axes[0, 2]
        self._draw_brain_vasculature(ax)
        
        # 4. Lightning (similar physics)
        ax = axes[1, 0]
        self._draw_lightning(ax)
        
        # 5. City networks
        ax = axes[1, 1]
        self._draw_city_network(ax)
        
        # 6. Internet topology
        ax = axes[1, 2]
        self._draw_internet_topology(ax)
        
        plt.tight_layout()
        return fig
    
    def _draw_neuron(self, ax):
        """Single neuron with dendritic tree"""
        # Cell body - using valid color names
        cell_body = Circle((0, 0), 0.1, facecolor='purple', edgecolor='indigo', linewidth=2)
        ax.add_patch(cell_body)
        
        # Axon
        ax.plot([0, 0], [0, -1], 'purple', linewidth=3, label='Axon')
        
        # Axon terminals
        for x in [-0.3, -0.1, 0.1, 0.3]:
            ax.plot([0, x], [-1, -1.2], 'purple', linewidth=2)
            ax.scatter([x], [-1.2], c='purple', s=50)
        
        # Dendritic tree (constructal branching)
        def draw_dendrite(x, y, angle, length, level, max_level=4):
            if level > max_level:
                return
            
            # Main branch
            x_end = x + length * np.cos(angle)
            y_end = y + length * np.sin(angle)
            ax.plot([x, x_end], [y, y_end], 'purple', 
                   linewidth=3*(1-level/max_level), alpha=0.8)
            
            # Sub-branches following constructal angles
            if level < max_level:
                # Optimal branching angle ~37 degrees
                branch_angle = np.pi/5
                new_length = length * 0.7  # Length reduction ratio
                
                draw_dendrite(x_end, y_end, angle + branch_angle, 
                            new_length, level + 1, max_level)
                draw_dendrite(x_end, y_end, angle - branch_angle, 
                            new_length, level + 1, max_level)
        
        # Draw main dendrites
        for angle in np.linspace(np.pi/4, 3*np.pi/4, 3):
            draw_dendrite(0, 0, angle, 0.4, 0)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1)
        ax.set_aspect('equal')
        ax.set_title('Neuron Dendritic Tree\nConstructal branching for signal collection')
        ax.text(0.5, -1.4, 'Dendrites collect signals\nfrom VOLUME → cell body (POINT)', 
                fontsize=9, ha='center')
    
    def _draw_neural_network(self, ax):
        """Neural network connectivity patterns"""
        # Hub neurons (following power law distribution)
        np.random.seed(42)
        n_neurons = 50
        
        # Position neurons
        positions = np.random.rand(n_neurons, 2) * 2 - 1
        
        # Assign connectivity (power law - few hubs, many peripherals)
        degrees = np.random.pareto(2, n_neurons) + 1
        degrees = degrees / degrees.max() * 20
        
        # Draw neurons
        for i, (pos, degree) in enumerate(zip(positions, degrees)):
            size = 20 + degree * 10
            ax.scatter(pos[0], pos[1], s=size, c='darkblue', alpha=0.7)
        
        # Draw connections (preferentially to hubs)
        for i in range(n_neurons):
            n_connections = int(degrees[i] / 2)
            # Connect to nearest high-degree neurons
            distances = np.sum((positions - positions[i])**2, axis=1)
            distances[i] = np.inf  # Don't connect to self
            
            # Weight by degree (prefer connecting to hubs)
            weights = degrees / (distances + 0.1)
            weights[i] = 0
            
            if n_connections > 0 and weights.sum() > 0:
                targets = np.random.choice(n_neurons, size=min(n_connections, n_neurons-1), 
                                         replace=False, p=weights/weights.sum())
                
                for target in targets:
                    ax.plot([positions[i, 0], positions[target, 0]], 
                           [positions[i, 1], positions[target, 1]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title('Neural Network Topology\nHub-based connectivity (small-world)')
        ax.text(0, -1.4, 'Information flows through hub neurons\nminimizing path lengths', 
                fontsize=9, ha='center')
    
    def _draw_brain_vasculature(self, ax):
        """Brain blood vessel network"""
        # Main arteries
        main_arteries = [
            ([0, 0], [0, 0.8]),  # Middle cerebral
            ([0, -0.3], [0, 0]),  # Basilar
            ([0, -0.3], [-0.4, -0.1]),  # Vertebral left
            ([0, -0.3], [0.4, -0.1])   # Vertebral right
        ]
        
        for start, end in main_arteries:
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   'red', linewidth=4, alpha=0.8)
        
        # Branching pattern
        def draw_vessels(x, y, angle, size, level, max_level=4):
            if level > max_level or size < 0.01:
                return
            
            # Length proportional to size^(1/3) (Murray's law)
            length = 0.3 * (size ** 0.33)
            x_end = x + length * np.cos(angle)
            y_end = y + length * np.sin(angle)
            
            ax.plot([x, x_end], [y, y_end], 'red', 
                   linewidth=size*20, alpha=0.6)
            
            # Optimal branching
            if level < max_level:
                # Daughter vessel sizes (Murray's law)
                size1 = size * 0.7
                size2 = size * 0.5
                
                # Branching angles
                angle1 = angle + np.pi/6
                angle2 = angle - np.pi/5
                
                draw_vessels(x_end, y_end, angle1, size1, level + 1, max_level)
                draw_vessels(x_end, y_end, angle2, size2, level + 1, max_level)
        
        # Draw branching from main arteries
        for i in range(5):
            angle = np.pi/2 + (i-2) * np.pi/8
            y_start = 0.3 + i * 0.1
            draw_vessels(0, y_start, angle, 0.15, 0)
        
        # Brain outline
        brain = Circle((0, 0.3), 0.7, fill=False, edgecolor='gray', 
                      linewidth=2, linestyle='--')
        ax.add_patch(brain)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Brain Vasculature\nOptimal blood distribution')
        ax.text(0, -0.4, 'Follows Murray\'s Law: d₀³ = d₁³ + d₂³', 
                fontsize=9, ha='center')
    
    def _draw_lightning(self, ax):
        """Lightning - similar physics but different mechanism"""
        # Lightning follows path of least resistance
        # but creates the path as it goes (not pre-optimized)
        
        def draw_lightning_branch(x, y, angle, level, max_level=6):
            if level > max_level:
                return
            
            # Random walk with bias downward
            length = np.random.uniform(0.1, 0.3) * (1 - level/max_level)
            angle_variation = np.random.normal(0, np.pi/6)
            new_angle = angle + angle_variation
            
            # Ensure generally downward movement
            if new_angle < -np.pi:
                new_angle = -np.pi
            elif new_angle > 0:
                new_angle = 0
            
            x_end = x + length * np.cos(new_angle)
            y_end = y + length * np.sin(new_angle)
            
            ax.plot([x, x_end], [y, y_end], 'yellow', 
                   linewidth=3*(1-level/max_level), alpha=0.9)
            
            # Branching probability decreases with level
            if np.random.random() < 0.7 * (1 - level/max_level):
                draw_lightning_branch(x_end, y_end, new_angle, level + 1, max_level)
            
            # Side branches
            if np.random.random() < 0.3 and level < max_level - 1:
                side_angle = new_angle + np.random.choice([-1, 1]) * np.pi/4
                draw_lightning_branch(x_end, y_end, side_angle, level + 2, max_level)
        
        # Dark background
        ax.set_facecolor('darkblue')
        
        # Main lightning bolt
        draw_lightning_branch(0, 1, -np.pi/2, 0)
        
        # Ground
        ax.fill_between([-1, 1], [-1, -1], [-0.8, -0.8], color='brown')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1.2)
        ax.set_title('Lightning\nNOT constructal - creates path dynamically', color='white')
        ax.text(0, -0.95, 'Follows momentary path of least resistance', 
                fontsize=9, ha='center', color='white')
    
    def _draw_city_network(self, ax):
        """City transportation networks"""
        # City blocks
        for x in np.linspace(-0.8, 0.8, 9):
            ax.plot([x, x], [-0.8, 0.8], 'lightgray', linewidth=0.5)
        for y in np.linspace(-0.8, 0.8, 9):
            ax.plot([-0.8, 0.8], [y, y], 'lightgray', linewidth=0.5)
        
        # Main arterial roads (constructal)
        # These emerge from traffic flow optimization
        ax.plot([0, 0], [-1, 1], 'black', linewidth=4, label='Main arterial')
        ax.plot([-1, 1], [0, 0], 'black', linewidth=4)
        
        # Secondary roads
        for pos in [-0.5, 0.5]:
            ax.plot([pos, pos], [-1, 1], 'gray', linewidth=2)
            ax.plot([-1, 1], [pos, pos], 'gray', linewidth=2)
        
        # Diagonal arterials (emerge in real cities)
        ax.plot([-0.7, 0.7], [-0.7, 0.7], 'darkgray', linewidth=3, linestyle='--')
        ax.plot([-0.7, 0.7], [0.7, -0.7], 'darkgray', linewidth=3, linestyle='--')
        
        # Traffic flow indicators
        arrow_props = dict(arrowstyle='->', color='red', lw=2, alpha=0.5)
        ax.annotate('', xy=(0, 0.6), xytext=(0, 0.4), arrowprops=arrow_props)
        ax.annotate('', xy=(0.6, 0), xytext=(0.4, 0), arrowprops=arrow_props)
        
        # City center (destination point)
        center = Circle((0, 0), 0.15, facecolor='red', alpha=0.3)
        ax.add_patch(center)
        ax.text(0, 0, 'CBD', ha='center', va='center', fontsize=10, weight='bold')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title('City Transportation Network\nTraffic flow optimization')
        ax.text(0, -1.15, 'Emerges from commuter flow patterns\nAREA → POINT (city center)', 
                fontsize=9, ha='center')
    
    def _draw_internet_topology(self, ax):
        """Internet backbone topology"""
        # Tier 1 providers (major hubs)
        tier1_positions = [
            (0, 0),      # Central hub
            (-0.6, 0.5), # Regional hubs
            (0.6, 0.5),
            (-0.6, -0.5),
            (0.6, -0.5)
        ]
        
        # Draw Tier 1 (backbone)
        for i, pos1 in enumerate(tier1_positions):
            ax.scatter(pos1[0], pos1[1], s=300, c='red', marker='s', 
                      edgecolor='darkred', linewidth=2)
            # Connect all Tier 1s to each other (full mesh)
            for j, pos2 in enumerate(tier1_positions):
                if i < j:
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           'red', linewidth=2, alpha=0.5)
        
        # Tier 2 providers
        tier2_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        tier2_positions = [(0.8*np.cos(a), 0.8*np.sin(a)) for a in tier2_angles]
        
        for pos in tier2_positions:
            ax.scatter(pos[0], pos[1], s=150, c='orange', marker='o')
            # Connect to nearest Tier 1
            distances = [np.sqrt((pos[0]-t1[0])**2 + (pos[1]-t1[1])**2) 
                        for t1 in tier1_positions]
            nearest = tier1_positions[np.argmin(distances)]
            ax.plot([pos[0], nearest[0]], [pos[1], nearest[1]], 
                   'orange', linewidth=1, alpha=0.7)
        
        # End users (Tier 3)
        n_users = 50
        np.random.seed(42)
        user_positions = np.random.normal(0, 0.5, (n_users, 2))
        user_positions = np.clip(user_positions, -1, 1)
        
        for pos in user_positions:
            ax.scatter(pos[0], pos[1], s=20, c='lightblue', alpha=0.5)
            # Connect to nearest Tier 2
            distances = [np.sqrt((pos[0]-t2[0])**2 + (pos[1]-t2[1])**2) 
                        for t2 in tier2_positions]
            nearest = tier2_positions[np.argmin(distances)]
            ax.plot([pos[0], nearest[0]], [pos[1], nearest[1]], 
                   'lightblue', linewidth=0.3, alpha=0.3)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Internet Topology\nHierarchical hub structure')
        ax.text(0, -1.35, 'Information flows through backbone hubs\nminimizing latency', 
                fontsize=9, ha='center')
        
        # Legend
        ax.text(-1.1, 1.1, '■ Tier 1', color='red', fontsize=8, weight='bold')
        ax.text(-1.1, 0.95, '● Tier 2', color='orange', fontsize=8, weight='bold')
        ax.text(-1.1, 0.8, '● Users', color='lightblue', fontsize=8)

# Create visualizations
examples = ExpandedConstructalExamples()
fig1 = examples.visualize_neural_networks()
plt.show()
