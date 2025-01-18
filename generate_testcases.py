import random
import numpy as np

seed = random.randint(0, 999999)

def generate_crs_matrix_to_file(size: int, num_connections: int, filename: str):
    if num_connections > size * size:
        raise ValueError("Number of connections cannot exceed total elements in the matrix.")

    # Set the random seed for reproducibility 
    random.seed(seed)
    print(f"Random seed for this run: {seed}")

    
    connections = set()

    # Randomly generate `num_connections` unique connections
    while len(connections) < num_connections:
        i = random.randint(1, size)  # 1 indexing to match the expected format
        j = random.randint(1, size)  # 1 indexing to match the expected format
        connections.add((i, j))

    # Write the matrix to the file
    with open(filename, "w") as file:
        # Write the first line: `x x n` where `x x` is the size of the matrix and `n` is the number of connections (decided on a square matrix for simplicity of writing generator)
        file.write(f"{size} {size} {num_connections}\n")

        # Write each connection as `row col val` (value = 1 as it won't affect the result, value is present just to match the format and make it a valid CRS matrix but it's not used in the algoritm that this is generating testcases/benchmarks for)
        for i, j in sorted(connections):  # Sort for easier debugging (doesn't affect the result)
            file.write(f"{i} {j} 1\n")
            
def abbreviate_number(num: int) -> str:
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num // 1000}k"
    else:
        return f"{num // 1_000_000}M"

size = int(input("Enter matrix size (the matrix is square so insert only 1 number): "))
num_connections = int(input("Enter number of connections (edges): "))
if num_connections < 0 or size < 0:
    raise ValueError("Size and number of connections must be positive integers.")
output_file = "crs_matrix_" + abbreviate_number(size) + "_" + str(seed) + ".txt"

generate_crs_matrix_to_file(size, num_connections, output_file)

print(f"CRS matrix written to {output_file}")
