import subprocess
import time

def run_program(command, matrix_file, source, repetitions):
    total_time = 0.0

    for i in range(repetitions):
        print(f"Run {i + 1}/{repetitions}...")
        
        # Start the timer, perf_counter() is used for more accurate timing as it is in nanoseconds (all subsecond timing are multiplied by 1000 to get milliseconds)
        start_time = time.perf_counter()

        try:
            subprocess.run([command, matrix_file, source], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running the program: {e}")
            continue

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        total_time += execution_time

        
        print(f"Execution time for run {i + 1}: {execution_time * 1000:.4f} ms")

    
    average_time = (total_time / repetitions) * 1000
    return average_time

if __name__ == "__main__":
    program = input("Enter the path to the program: ")
    matrix_file = input("Enter the path to the matrix file: ")
    source = input("Enter the source node: ")
    runs = int(input("Enter the number of runs: "))

    avg_time = run_program(program, matrix_file, source, runs)
    print(f"Average execution time over {runs} runs: {avg_time:.4f} ms")
