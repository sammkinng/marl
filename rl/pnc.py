import itertools

# --- Configuration ---
START_VALUE = -5.0
END_VALUE = 5.0
STEP_SIZE = 0.1
NUM_VARIABLES = 4
# We use 1 for rounding precision since the step size is 0.1 (one decimal place)
PRECISION = len(str(STEP_SIZE).split('.')[-1])

def generate_float_range(start, end, step):
    """
    Generates a list of floating-point numbers in the specified range and step.
    Rounds to the required precision to avoid floating-point errors.
    """
    values = []
    current = start
    # We add a small tolerance (step / 2) to the end condition to ensure the
    # final value is included despite potential floating point inaccuracies.
    while current <= end + (step / 2):
        values.append(round(current, PRECISION))
        current += step
    return values

def generate_all_permutations(possible_values, num_variables):
    """
    Generates the Cartesian product (permutations with repetition) for the
    given list of values and number of variables.
    Returns a generator object for memory efficiency.
    """
    # itertools.product is perfect for generating the Cartesian product,
    # which covers all permutations with repetition.
    return itertools.product(possible_values, repeat=num_variables)

