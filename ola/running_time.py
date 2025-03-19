# file modified from output of Claude 3.5 Haiku
import timeit
import random
import matplotlib.pyplot as plt
import numpy as np

from ola_encoding import (
    to_vector,
    to_tree,
)
from utils import (
    get_random_tree,
    get_random_vector,
)

def time_function(
    func, input_sizes, setup_func=None, 
    number_samples=10, number_timeit=10
):
    """
    Empirically test the running time of a function across different input sizes.
    
    Parameters:
    - func: The function to be timed
    - input_sizes: List of input sizes to test
    - setup_func: Optional function to generate inputs for the tested function
    - number: Number of times to run the function for each input size
    
    Returns:
    - List of average execution times for each input size
    """
    execution_times = []
    
    for size in input_sizes:
        # If no setup function is provided, use a default that generates a random list
        if setup_func is None:
            def default_setup():
                return list(range(size))
            setup_func = default_setup
        
        # Prepare the input
        input_data = [setup_func(size) for _ in range(number_samples)]
        
        # Time the function using timeit
        time = timeit.timeit(
            lambda: [func(input) for input in input_data], 
            number=number_timeit
        )
        
        # Calculate average time per execution
        avg_time = time / (number_timeit * number_samples)
        execution_times.append(avg_time)
        print(f"finished execution, size {size}")
    
    return execution_times

def visualize_performance(
    input_sizes, execution_times, 
    function_name="Function", output="temp.pdf"
):
    """
    Create a visualization of the function's performance.
    
    Parameters:
    - input_sizes: List of input sizes tested
    - execution_times: Corresponding list of execution times
    """
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, execution_times, marker='o')
    plt.title(f'{function_name} Performance vs Input Size')
    plt.xlabel('Input Size')
    plt.ylabel('Average Execution Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Fit a line to log-log plot to estimate complexity
    log_sizes = np.log(input_sizes)
    log_times = np.log(execution_times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    
    plt.text(0.05, 0.95, f'Estimated complexity: O(n^{slope:.2f})', 
             transform=plt.gca().transAxes, 
             verticalalignment='top')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(output)


# Test the function
def main(func, setup_func, func_name):
    # Define input sizes to test (logarithmic scale)
    input_sizes = [10, 20, 50, 100, 200, 500, 1000]
    # [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    # Time the function
    execution_times = time_function(
        func=func, 
        input_sizes=input_sizes, 
        setup_func=setup_func,
        number_timeit=100,
    )
    
    # Print results
    print("Input Sizes:", input_sizes)
    print("Execution Times:", execution_times)
    
    # Visualize performance
    visualize_performance(input_sizes, execution_times, function_name=func_name)

if __name__ == "__main__":
    main(to_vector, get_random_tree, "OLA Encoding")
    # main(to_tree, get_random_vector, "OLA Decoding")