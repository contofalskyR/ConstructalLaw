import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product

class ConstructalFlowTypes:
    """
    Demonstrate different flow configurations in constructal theory
    """
    
    def visualize_flow_types(self):
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Volume-to-Point Flow (3D)
        ax1 = fig.add_subplot(231, projection='3d')
        self._show_volume_to_point(ax1)
        
        # 2. Area-to-Point Flow (2D)
        ax2 = fig.add_subplot(232)
        self._show_area_to_point(ax2)
        
        # 3. Line-to-Point Flow (1D)
        ax3 = fig.add_subplot(233)
        self._show_line_to_point(ax3)
        
        # 4. Real examples
        ax4 = fig.add_subplot(234)
        self._show_river_basin(ax4)
        
        ax5 = fig.add_subplot(235)
        self._show_lung_system(ax5)
        
        ax6 = fig.add_subplot(236)
        self._show_heat_sink(ax6)
        
        plt.tight_layout()
        return fig
    
    def _show_volume_to_point(self, ax):
        """
        Volume-to-point: Flow from entire 3D space to single outlet
        Example: Lung airways, vascular system
        """
        # Main trunk
        ax.plot([0, 0], [0, 0], [0, 2], 'b-', linewidth=4, label='Main channel')
        
        # Primary branches
        for zi in [0.5, 1.0, 1.5]:
            for xi, yi in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                ax.plot([0, xi], [0, yi], [zi, zi+0.3], 'b-', linewidth=2)
                
                # Secondary branches
                for dx, dy in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
                    if abs(xi + dx) < 1 and abs(yi + dy) < 1:
                        ax.plot([xi, xi+dx], [yi, yi+dy], 
                               [zi+0.3, zi+0.4], 'b-', linewidth=1, alpha=0.5)
        
        # Draw cube edges to show volume
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r, r, [0, 2]))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot(*zip(s, e), color="gray", alpha=0.3)
        
        # Labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('VOLUME-to-Point Flow\n(3D space → single point)\nExample: Lungs, blood vessels')
    
    def _show_area_to_point(self, ax):
        """
        Area-to-point: Flow from 2D surface to single point
        Example: River basins, heat sinks
        """
        # Create grid to show the area
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)
        
        # Color map showing "drainage" to center
        Z = np.sqrt(X**2 + Y**2)
        
        im = ax.contourf(X, Y, Z, levels=20, cmap='Blues_r', alpha=0.3)
        
        # Draw optimal flow channels (tree structure)
        # Main channel
        ax.plot([0, 0], [-2, 0], 'b-', linewidth=4)
        
        # Primary branches
        branch_angles = [np.pi/4, 3*np.pi/4]
        for angle in branch_angles:
            x_end = 1.5 * np.cos(angle)
            y_end = 1.5 * np.sin(angle)
            ax.plot([0, x_end], [0, y_end], 'b-', linewidth=3)
            
            # Secondary branches
            for sub_angle in [angle - np.pi/6, angle + np.pi/6]:
                x_start = 0.7 * np.cos(angle)
                y_start = 0.7 * np.sin(angle)
                x_end2 = x_start + 0.6 * np.cos(sub_angle)
                y_end2 = y_start + 0.6 * np.sin(sub_angle)
                ax.plot([x_start, x_end2], [y_start, y_end2], 'b-', linewidth=2)
        
        ax.scatter([0], [0], c='red', s=100, zorder=5, label='Collection point')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title('AREA-to-Point Flow\n(2D surface → single point)\nExample: River basins')
        ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, edgecolor='black', linewidth=2))
        ax.text(1.5, 1.5, 'AREA', fontsize=14, weight='bold', color='gray')
    
    def _show_line_to_point(self, ax):
        """
        Line-to-point: Flow from 1D line to point
        Simplest case
        """
        # Show a line with flow toward point
        x = np.linspace(-2, 2, 100)
        
        # Flow intensity along line
        flow_intensity = 2 - np.abs(x)
        
        ax.fill_between(x, 0, flow_intensity, alpha=0.3, color='blue', 
                       label='Flow collection')
        
        # Optimal collection channels
        for xi in [-1.5, -0.5, 0.5, 1.5]:
            ax.arrow(xi, 0, 0, 1.5 - 0.3*abs(xi), 
                    head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.7)
        
        # Collection point
        ax.scatter([0], [2], c='red', s=100, zorder=5)
        ax.plot([-2, 2], [0, 0], 'k-', linewidth=3, label='Source line')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title('LINE-to-Point Flow\n(1D line → single point)')
        ax.text(0, -0.3, 'LINE', fontsize=14, weight='bold', ha='center')
        ax.legend()
    
    def _show_river_basin(self, ax):
        """Real example: River basin (area-to-point)"""
        # Simulate a river network
        np.random.seed(42)
        
        # Main river
        x_main = np.array([0, 0.1, 0.15, 0.1, 0.2, 0.3])
        y_main = np.linspace(0, 1, 6)
        ax.plot(x_main, y_main, 'b-', linewidth=4)
        
        # Tributaries
        tributary_points = [
            (0.3, 0.2, 0.45), (0.5, 0.3, 0.55),
            (-0.2, 0.1, 0.35), (-0.4, 0.15, 0.4),
            (0.4, 0.5, 0.7), (-0.3, 0.6, 0.75)
        ]
        
        for x_start, y_start, y_join in tributary_points:
            # Find x position on main river at y_join
            x_join = np.interp(y_join, y_main, x_main)
            
            # Draw tributary
            x_trib = np.array([x_start, (x_start + x_join)/2, x_join])
            y_trib = np.array([y_start, (y_start + y_join)/2, y_join])
            ax.plot(x_trib, y_trib, 'b-', linewidth=2)
            
            # Sub-tributaries
            if abs(x_start) > 0.3:
                x_sub = x_start + np.sign(x_start) * 0.15
                y_sub = y_start - 0.08
                x_sub_join = x_trib[1]
                y_sub_join = y_trib[1]
                ax.plot([x_sub, x_sub_join], [y_sub, y_sub_join], 
                       'b-', linewidth=1, alpha=0.7)
        
        # Show drainage area
        circle = plt.Circle((0, 0.5), 0.7, fill=False, 
                          edgecolor='green', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(0, 1)
        ax.set_title('River Basin\nAREA of land → POINT (river mouth)')
        ax.text(0, -0.05, 'River Mouth (point)', ha='center', fontsize=10, weight='bold')
        ax.text(0.5, 0.8, 'Drainage\nArea', ha='center', color='green', fontsize=10)
    
    def _show_lung_system(self, ax):
        """Real example: Lungs (volume-to-point)"""
        # Bronchial tree structure
        levels = [
            {'y': 0, 'x': [0], 'width': 5},
            {'y': 0.2, 'x': [-0.2, 0.2], 'width': 3},
            {'y': 0.4, 'x': [-0.35, -0.1, 0.1, 0.35], 'width': 2},
            {'y': 0.6, 'x': [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45], 'width': 1.5},
            {'y': 0.8, 'x': np.linspace(-0.5, 0.5, 12), 'width': 1}
        ]
        
        # Draw connections between levels
        for i in range(len(levels)-1):
            current = levels[i]
            next_level = levels[i+1]
            
            for x1 in current['x']:
                # Find two nearest points in next level
                distances = [abs(x1 - x2) for x2 in next_level['x']]
                nearest_indices = np.argsort(distances)[:2]
                
                for idx in nearest_indices:
                    x2 = next_level['x'][idx]
                    ax.plot([x1, x2], [current['y'], next_level['y']], 
                           'r-', linewidth=current['width'], alpha=0.7)
        
        # Draw alveoli (air sacs) at the end
        for x in levels[-1]['x']:
            circle = plt.Circle((x, 0.9), 0.03, fill=True, 
                              facecolor='pink', edgecolor='red', alpha=0.5)
            ax.add_patch(circle)
        
        # Lung outline
        lung_x = np.array([-0.6, -0.65, -0.6, -0.4, 0, 0.4, 0.6, 0.65, 0.6])
        lung_y = np.array([0, 0.3, 0.6, 0.9, 1, 0.9, 0.6, 0.3, 0])
        ax.plot(lung_x, lung_y, 'k-', linewidth=2, alpha=0.3)
        ax.fill(lung_x, lung_y, 'pink', alpha=0.1)
        
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Lung System\nVOLUME of tissue → POINT (trachea)')
        ax.text(0, -0.05, 'Trachea (point)', ha='center', fontsize=10, weight='bold')
        ax.text(-0.5, 0.5, '3D Volume\nof lung tissue', fontsize=9, color='gray')
    
    def _show_heat_sink(self, ax):
        """Real example: Heat sink (area-to-point)"""
        # Base plate
        base = Rectangle((-1, 0), 2, 0.1, facecolor='gray', edgecolor='black')
        ax.add_patch(base)
        
        # Heat source (CPU)
        cpu = Rectangle((-0.2, -0.05), 0.4, 0.05, facecolor='red', edgecolor='darkred')
        ax.add_patch(cpu)
        ax.text(0, -0.025, 'CPU', ha='center', va='center', fontsize=8, weight='bold')
        
        # Fins (channels for heat flow)
        fin_positions = np.linspace(-0.8, 0.8, 9)
        for x in fin_positions:
            # Fin height proportional to distance from center (constructal)
            height = 0.6 * (1 - abs(x)/1.2)
            fin = Rectangle((x-0.03, 0.1), 0.06, height, 
                          facecolor='silver', edgecolor='black')
            ax.add_patch(fin)
            
            # Heat flow arrows
            ax.arrow(x, 0.05, 0, height*0.8, 
                    head_width=0.02, head_length=0.03,
                    fc='orange', ec='orange', alpha=0.5)
        
        # Air flow
        for y in np.linspace(0.2, 0.7, 4):
            ax.arrow(-1.2, y, 2.4, 0, head_width=0.03, head_length=0.05,
                    fc='lightblue', ec='lightblue', alpha=0.3)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.1, 0.8)
        ax.set_aspect('equal')
        ax.set_title('Heat Sink\nAREA of hot surface → POINT (CPU)')
        ax.text(1, 0.4, 'Air flow', color='lightblue', rotation=0)

# Now create and show the visualization
flow_demo = ConstructalFlowTypes()
fig = flow_demo.visualize_flow_types()
plt.show()
