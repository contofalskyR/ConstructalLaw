import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.patches import Rectangle, Circle

class InformationConstructal:
    """
    Explore connections between information theory and constructal law
    """
    
    def __init__(self):
        self.fig_num = 0
    
    def visualize_information_flow(self):
        """Main visualization showing information-constructal connections"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Shannon's channel capacity as flow
        ax = axes[0, 0]
        self._draw_channel_capacity(ax)
        
        # 2. Decision tree as constructal branching
        ax = axes[0, 1]
        self._draw_decision_tree(ax)
        
        # 3. Huffman coding tree
        ax = axes[0, 2]
        self._draw_huffman_tree(ax)
        
        # 4. Information flow in neural network
        ax = axes[1, 0]
        self._draw_neural_information_flow(ax)
        
        # 5. Categorization as flow optimization
        ax = axes[1, 1]
        self._draw_categorization_flow(ax)
        
        # 6. Mutual information landscape
        ax = axes[1, 2]
        self._draw_mutual_information_landscape(ax)
        
        plt.tight_layout()
        return fig
    
    def _draw_channel_capacity(self, ax):
        """Shannon's channel as a flow system"""
        # Source distribution (volume)
        n_sources = 8
        source_probs = np.random.dirichlet(np.ones(n_sources))
        source_y = np.linspace(0.2, 0.8, n_sources)
        
        # Channel (flow network)
        n_outputs = 4
        output_y = np.linspace(0.3, 0.7, n_outputs)
        
        # Transition probabilities (flow rates)
        # Create a channel that converges information
        for i, (p, y1) in enumerate(zip(source_probs, source_y)):
            # Information "flows" to outputs
            # Higher probability sources have thicker flows
            for j, y2 in enumerate(output_y):
                # Channel matrix element
                transition_prob = np.exp(-2 * abs(i/n_sources - j/n_outputs))
                transition_prob /= 4  # Normalize
                
                if transition_prob > 0.01:
                    ax.plot([0.2, 0.8], [y1, y2], 'blue', 
                           linewidth=p * transition_prob * 20, 
                           alpha=0.5)
        
        # Draw source nodes
        for p, y in zip(source_probs, source_y):
            size = p * 1000
            ax.scatter(0.2, y, s=size, c='green', alpha=0.7, edgecolor='darkgreen')
            ax.text(0.1, y, f'{p:.2f}', ha='right', va='center', fontsize=8)
        
        # Draw output nodes
        output_probs = np.zeros(n_outputs)
        for i, p_source in enumerate(source_probs):
            for j in range(n_outputs):
                transition_prob = np.exp(-2 * abs(i/n_sources - j/n_outputs)) / 4
                output_probs[j] += p_source * transition_prob
        
        for p, y in zip(output_probs, output_y):
            size = p * 1000
            ax.scatter(0.8, y, s=size, c='red', alpha=0.7, edgecolor='darkred')
            ax.text(0.9, y, f'{p:.2f}', ha='left', va='center', fontsize=8)
        
        # Calculate channel capacity
        H_input = entropy(source_probs, base=2)
        H_output = entropy(output_probs, base=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Information Channel as Flow System')
        ax.text(0.5, 0.1, f'H(input) = {H_input:.2f} bits → H(output) = {H_output:.2f} bits',
                ha='center', fontsize=10)
        ax.text(0.2, 0.05, 'Source\n(Volume)', ha='center', fontsize=9, weight='bold')
        ax.text(0.8, 0.05, 'Sink\n(Point)', ha='center', fontsize=9, weight='bold')
    
    def _draw_decision_tree(self, ax):
        """Decision tree following constructal branching"""
        # Information content at each node
        def draw_node(x, y, information, level=0, max_level=4):
            if level > max_level:
                return
            
            # Node size proportional to information content
            size = information * 100
            color = plt.cm.viridis(1 - information)
            ax.scatter(x, y, s=size, c=[color], edgecolor='black')
            
            if level < max_level:
                # Branching reduces uncertainty (information flow)
                # Optimal split roughly halves information
                left_info = information * 0.6  # Asymmetric like constructal
                right_info = information * 0.4
                
                # Branch lengths proportional to information gain
                dx = 0.3 * (0.8 ** level)
                dy = -0.2
                
                # Draw branches (information channels)
                ax.plot([x, x-dx], [y, y+dy], 'black', 
                       linewidth=left_info*5, alpha=0.7)
                ax.plot([x, x+dx], [y, y+dy], 'black', 
                       linewidth=right_info*5, alpha=0.7)
                
                # Information gain annotations
                ax.text(x-dx/2, y+dy/2, f'-{information-left_info:.2f}', 
                       fontsize=7, ha='center', color='red')
                ax.text(x+dx/2, y+dy/2, f'-{information-right_info:.2f}', 
                       fontsize=7, ha='center', color='red')
                
                # Recursive branching
                draw_node(x-dx, y+dy, left_info, level+1, max_level)
                draw_node(x+dx, y+dy, right_info, level+1, max_level)
        
        # Start with maximum uncertainty
        initial_information = 1.0
        draw_node(0.5, 0.9, initial_information)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Decision Tree: Information Flow')
        ax.text(0.5, 0.05, 'Information flows from root (high H) to leaves (low H)',
                ha='center', fontsize=9)
    
    def _draw_huffman_tree(self, ax):
        """Huffman coding as optimal flow structure"""
        # Symbol probabilities
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        probs = [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04]
        
        # Build Huffman tree (optimal for information flow)
        nodes = [(p, s) for p, s in zip(probs, symbols)]
        tree_levels = []
        
        y_positions = {}
        x_base = 0.9
        
        # Initial positions
        for i, (p, s) in enumerate(nodes):
            y_positions[s] = 0.1 + i * 0.1
            ax.scatter(x_base, y_positions[s], s=p*1000, c='lightblue')
            ax.text(x_base + 0.05, y_positions[s], f'{s}:{p:.2f}', 
                   fontsize=8, va='center')
        
        # Build tree
        current_nodes = nodes.copy()
        x_current = x_base
        
        while len(current_nodes) > 1:
            # Sort by probability
            current_nodes.sort(key=lambda x: x[0])
            
            # Combine two smallest (constructal merging)
            p1, s1 = current_nodes.pop(0)
            p2, s2 = current_nodes.pop(0)
            
            # New node
            p_new = p1 + p2
            s_new = f"({s1},{s2})"
            current_nodes.append((p_new, s_new))
            
            # Position for new node
            y_new = (y_positions[s1] + y_positions[s2]) / 2
            y_positions[s_new] = y_new
            x_new = x_current - 0.15
            
            # Draw merge (information channels)
            ax.plot([x_current, x_new], [y_positions[s1], y_new], 
                   'black', linewidth=p1*10, alpha=0.7)
            ax.plot([x_current, x_new], [y_positions[s2], y_new], 
                   'black', linewidth=p2*10, alpha=0.7)
            
            # Draw new node
            ax.scatter(x_new, y_new, s=p_new*1000, c='orange', alpha=0.7)
            
            x_current = x_new
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.set_title('Huffman Tree: Optimal Information Packing')
        ax.text(0.5, 0.95, 'Minimizes average path length (like constructal flow)',
                ha='center', fontsize=9)
    
    def _draw_neural_information_flow(self, ax):
        """Information flow in neural network"""
        # Create layers
        layers = [8, 4, 2, 1]  # Converging architecture
        layer_x = np.linspace(0.1, 0.9, len(layers))
        
        # Information content at each layer
        info_content = []
        positions = []
        
        for i, (n_neurons, x) in enumerate(zip(layers, layer_x)):
            layer_y = np.linspace(0.2, 0.8, n_neurons)
            layer_positions = [(x, y) for y in layer_y]
            positions.append(layer_positions)
            
            # Information decreases through layers (compression)
            layer_info = 1.0 * (0.6 ** i)  # Information reduction
            info_content.append([layer_info] * n_neurons)
            
            # Draw neurons
            for y in layer_y:
                size = layer_info * 200
                ax.scatter(x, y, s=size, c='purple', alpha=0.7, edgecolor='indigo')
        
        # Draw connections with width proportional to information flow
        for i in range(len(layers) - 1):
            for j, (x1, y1) in enumerate(positions[i]):
                info1 = info_content[i][j]
                for k, (x2, y2) in enumerate(positions[i+1]):
                    # Information flow between neurons
                    flow_rate = info1 / len(positions[i+1])
                    ax.plot([x1, x2], [y1, y2], 'gray', 
                           linewidth=flow_rate*10, alpha=0.3)
        
        # Annotate information content
        for i, (info_layer, x) in enumerate(zip(info_content, layer_x)):
            total_info = sum(info_layer)
            ax.text(x, 0.1, f'H={total_info:.2f}', ha='center', fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Neural Information Bottleneck')
        ax.text(0.5, 0.02, 'Information compresses from input to output',
                ha='center', fontsize=9)
    
    def _draw_categorization_flow(self, ax):
        """Categorization as information flow optimization"""
        # Following the Feldman mutual information paper
        
        # Create 2D feature space
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Two category distributions (Gaussian)
        cat_A = np.exp(-((X - 0.5)**2 + Y**2) / 0.5)
        cat_B = np.exp(-((X + 0.5)**2 + Y**2) / 0.5)
        
        # Total probability
        total = cat_A + cat_B
        p_A = cat_A / total
        p_B = cat_B / total
        
        # Mutual information at each point
        # MI = H(C) - H(C|x,y)
        H_C = 1.0  # Binary categories with equal priors
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        H_C_given_xy = -(p_A * np.log2(p_A + epsilon) + p_B * np.log2(p_B + epsilon))
        MI = H_C - H_C_given_xy
        
        # Plot mutual information landscape
        im = ax.contourf(X, Y, MI, levels=20, cmap='hot')
        plt.colorbar(im, ax=ax, label='Mutual Information (bits)')
        
        # Overlay optimal decision boundary
        ax.contour(X, Y, p_A - p_B, levels=[0], colors='white', linewidths=2)
        
        # Show information flow vectors
        # Gradient of MI shows direction of maximum information gain
        dy, dx = np.gradient(MI)
        skip = 10
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 dx[::skip, ::skip], dy[::skip, ::skip], 
                 alpha=0.5, color='white', scale=50)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title('Categorization: Information Flow Field')
        ax.text(0, -2.3, 'Features flow toward maximum mutual information',
                ha='center', fontsize=9)
    
    def _draw_mutual_information_landscape(self, ax):
        """3D visualization of MI as potential field"""
        # Create feature space
        angles = np.linspace(0, np.pi/2, 50)
        positions = np.linspace(0, 1, 50)
        Angles, Positions = np.meshgrid(angles, positions)
        
        # Calculate mutual information for each feature
        # Following Feldman's formulation
        MI = np.zeros_like(Angles)
        
        for i, angle in enumerate(angles):
            for j, pos in enumerate(positions):
                # Feature at boundary has max MI
                if pos == 0.5:  # At boundary
                    MI[j, i] = np.cos(angle)**2  # Max when aligned
                else:
                    # MI decreases away from boundary
                    distance_from_boundary = abs(pos - 0.5)
                    MI[j, i] = np.cos(angle)**2 * np.exp(-distance_from_boundary * 5)
        
        # Plot as heatmap with flow lines
        im = ax.imshow(MI, extent=[0, 90, 0, 1], aspect='auto', 
                      origin='lower', cmap='viridis')
        
        # Add flow lines showing optimal feature evolution
        # Features should flow toward high MI regions
        for start_angle in [10, 30, 50, 70]:
            trajectory_angles = [start_angle]
            trajectory_positions = [0.1]
            
            # Simulate feature evolution
            for _ in range(20):
                i = int(trajectory_angles[-1] / 90 * 49)
                j = int(trajectory_positions[-1] * 49)
                
                if i < 49 and j < 49:
                    # Gradient ascent on MI
                    if i < 48:
                        d_angle = MI[j, i+1] - MI[j, i]
                    else:
                        d_angle = 0
                    if j < 48:
                        d_pos = MI[j+1, i] - MI[j, i]
                    else:
                        d_pos = 0
                    
                    new_angle = trajectory_angles[-1] + d_angle * 50
                    new_pos = trajectory_positions[-1] + d_pos * 0.5
                    
                    trajectory_angles.append(np.clip(new_angle, 0, 90))
                    trajectory_positions.append(np.clip(new_pos, 0, 1))
            
            ax.plot(trajectory_angles, trajectory_positions, 'white', 
                   linewidth=2, alpha=0.7)
        
        plt.colorbar(im, ax=ax, label='Mutual Information')
        ax.set_xlabel('Feature Angle (degrees)')
        ax.set_ylabel('Position in Feature Space')
        ax.set_title('MI Landscape: Features Flow to High Information')
    
    def theoretical_connection(self):
        """Show the mathematical connection"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physical flow
        ax1.text(0.5, 0.9, 'CONSTRUCTAL LAW', ha='center', fontsize=16, weight='bold')
        ax1.text(0.5, 0.8, 'For a flow system to persist in time,\nit must evolve to provide easier access',
                ha='center', fontsize=12)
        ax1.text(0.1, 0.6, 'Minimize:', fontsize=12, weight='bold')
        ax1.text(0.1, 0.5, r'$R_{total} = \sum_i R_i = \sum_i \frac{L_i}{A_i}$', fontsize=14)
        ax1.text(0.1, 0.35, 'Where:', fontsize=10)
        ax1.text(0.1, 0.3, r'$R_i$ = resistance of branch i', fontsize=10)
        ax1.text(0.1, 0.25, r'$L_i$ = length of branch i', fontsize=10)
        ax1.text(0.1, 0.2, r'$A_i$ = cross-sectional area', fontsize=10)
        ax1.text(0.1, 0.1, 'Result: Tree-like structures', fontsize=11, style='italic')
        ax1.axis('off')
        
        # Information flow
        ax2.text(0.5, 0.9, 'INFORMATION THEORY', ha='center', fontsize=16, weight='bold')
        ax2.text(0.5, 0.8, 'For efficient communication,\nminimize uncertainty about messages',
                ha='center', fontsize=12)
        ax2.text(0.1, 0.6, 'Maximize:', fontsize=12, weight='bold')
        ax2.text(0.1, 0.5, r'$I(X;Y) = H(Y) - H(Y|X)$', fontsize=14)
        ax2.text(0.1, 0.35, 'Where:', fontsize=10)
        ax2.text(0.1, 0.3, r'$I(X;Y)$ = mutual information', fontsize=10)
        ax2.text(0.1, 0.25, r'$H(Y)$ = output entropy', fontsize=10)
        ax2.text(0.1, 0.2, r'$H(Y|X)$ = conditional entropy', fontsize=10)
        ax2.text(0.1, 0.1, 'Result: Tree-like codes (Huffman)', fontsize=11, style='italic')
        ax2.axis('off')
        
        fig.suptitle('The Deep Connection:\nBoth minimize "resistance" to flow (physical vs information)',
                    fontsize=14, weight='bold')
        
        return fig

# Create all visualizations
info_constructal = InformationConstructal()
fig1 = info_constructal.visualize_information_flow()
fig2 = info_constructal.theoretical_connection()
plt.show()

# Demonstrate the mathematical connection
print("\n" + "="*60)
print("MATHEMATICAL CONNECTION BETWEEN CONSTRUCTAL AND INFORMATION")
print("="*60)
print("\n1. PHYSICAL FLOW:")
print("   - Minimize resistance: R = L/A")
print("   - Optimal branching: d₀³ = d₁³ + d₂³")
print("   - Result: Power law distributions")

print("\n2. INFORMATION FLOW:")
print("   - Minimize uncertainty: H(X) = -Σ p(x)log(p(x))")
print("   - Optimal coding: L̄ = Σ p(i)l(i) ≥ H(X)")
print("   - Result: Power law distributions")

print("\n3. THE CONNECTION:")
print("   - Both involve VOLUME → POINT flow")
print("   - Both minimize a 'resistance' measure")
print("   - Both produce hierarchical trees")
print("   - Both follow power laws")

print("\n4. FELDMAN'S INSIGHT:")
print("   - Perceptual features evolve like flow channels")
print("   - They optimize for information transfer")
print("   - ΔThreshold ∝ Mutual Information")
print("   - Categories create information 'watersheds'")

print("\n5. UNIFIED VIEW:")
print("   - Physical systems: minimize energy dissipation")
print("   - Information systems: minimize uncertainty")
print("   - Both create channels for efficient transport")
print("   - The math is remarkably similar!")
