
#%%
import numpy as np
import matplotlib.pyplot as plt

def apply_rope(x, position, dim):
    """
    Apply Rotary Positional Encoding to a vector x at given position
    x: input vector of shape (dim,)
    position: token position in sequence
    dim: embedding dimension
    """
    x_rope = x.copy()
    
    # Process pairs of dimensions
    for i in range(0, dim, 2):
        # Calculate rotation angle for this dimension pair
        theta = position / (10000 ** (2 * i / dim))
        
        # Get the pair of values
        x_i = x[i]
        x_i_plus_1 = x[i + 1]
        
        # Apply rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_rope[i] = x_i * cos_theta - x_i_plus_1 * sin_theta
        x_rope[i + 1] = x_i * sin_theta + x_i_plus_1 * cos_theta
    
    return x_rope

#%%
# Parameters
seq_length = 12
embed_dim = 8

print("=== RoPE Numerical Example ===")
print(f"Sequence length: {seq_length}")
print(f"Embedding dimension: {embed_dim}")
print()

# Example query vector (same for all positions before RoPE)
original_query = np.array([1.0, 0.5, -0.3, 0.8, 0.2, -0.6, 0.9, -0.1])
print("Original query vector (before RoPE):")
print(f"q = {original_query}")
print()

# Calculate rotation angles for each position and dimension pair
print("Rotation angles θ = position / 10000^(2i/d) for each dimension pair:")
print("Position |", end="")
for i in range(0, embed_dim, 2):
    print(f"  Pair({i},{i+1})  |", end="")
print()
print("-" * (10 + 13 * (embed_dim // 2)))

for pos in range(0, seq_length, 2):  # Show every other position to save space
    print(f"   {pos:2d}    |", end="")
    for i in range(0, embed_dim, 2):
        theta = pos / (10000 ** (2 * i / embed_dim))
        print(f"   {theta:8.5f}   |", end="")
    print()

print()

# Apply RoPE to queries at different positions
print("Query vectors after applying RoPE at different positions:")
print("Position | Rotated Query Vector")
print("-" * 60)

rope_queries = []
for pos in range(seq_length):
    q_rope = apply_rope(original_query, pos, embed_dim)
    rope_queries.append(q_rope)
    if pos % 2 == 0:  # Show every other position
        print(f"   {pos:2d}    | {q_rope}")

print()

# Show how dot products encode relative positions
print("Demonstrating relative position encoding through dot products:")
print("Computing q_i · q_j for different position pairs:")
print()

# Calculate dot products between queries at different positions
print("Pos i | Pos j | q_i · q_j  | Relative distance |i-j|")
print("-" * 50)

example_pairs = [(0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 6), (1, 5), (3, 7)]
for i, j in example_pairs:
    dot_product = np.dot(rope_queries[i], rope_queries[j])
    rel_distance = abs(i - j)
    print(f"  {i:2d}  |  {j:2d}   |  {dot_product:8.5f}  |       {rel_distance}")

print()

# Show that queries with same relative distance have similar dot products
print("Observing that same relative distances give similar dot products:")
print("Relative distance | Example pairs | Dot products")
print("-" * 50)

distances_to_check = [0, 2, 4]
for dist in distances_to_check:
    pairs_with_dist = []
    dot_products_for_dist = []
    
    for i in range(seq_length):
        j = i + dist
        if j < seq_length:
            pairs_with_dist.append((i, j))
            dot_products_for_dist.append(np.dot(rope_queries[i], rope_queries[j]))
    
    if pairs_with_dist:
        print(f"      {dist}       | {pairs_with_dist[:3]} | {dot_products_for_dist[:3]}")

print()

# Visualize the rotation effect
print("=== Visualization Data ===")
print("First two dimensions of queries at each position (for plotting):")
print("Position | Dim 0 | Dim 1 | Angle from origin")
print("-" * 45)

angles = []
for pos in range(seq_length):
    q = rope_queries[pos]
    angle = np.arctan2(q[1], q[0])
    angles.append(angle)
    print(f"   {pos:2d}    | {q[0]:5.2f} | {q[1]:5.2f} | {angle:8.5f} rad")

print(f"\nRotation per position in first dimension pair: {angles[1] - angles[0]:.5f} radians")
print(f"This equals position / 10000^(0/8) = 1 / 10000^0 = 1.0 radian per position")

# Show how the rotation matrix works for position 3
print(f"\n=== Detailed rotation example for position 3 ===")
pos = 3
print(f"Original query: {original_query}")
print(f"Applying RoPE at position {pos}:")

for i in range(0, embed_dim, 2):
    theta = pos / (10000 ** (2 * i / embed_dim))
    x_i = original_query[i]
    x_i_plus_1 = original_query[i + 1]
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    new_x_i = x_i * cos_theta - x_i_plus_1 * sin_theta
    new_x_i_plus_1 = x_i * sin_theta + x_i_plus_1 * cos_theta
    
    print(f"Dimension pair ({i},{i+1}): θ = {theta:.5f}")
    print(f"  [{x_i:.3f}, {x_i_plus_1:.3f}] → [{new_x_i:.3f}, {new_x_i_plus_1:.3f}]")
    print(f"  Rotation matrix: [[{cos_theta:.3f}, {-sin_theta:.3f}], [{sin_theta:.3f}, {cos_theta:.3f}]]")

final_q = apply_rope(original_query, pos, embed_dim)
print(f"Final rotated query at position {pos}: {final_q}")
# %%
