"""
Mock numpy for testing cognitive grammar without external dependencies
"""

import math
import random

class MockArray:
    """Mock numpy array"""
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
            self.shape = shape or (len(data),)
        elif isinstance(data, (int, float)):
            self.data = [data]
            self.shape = (1,)
        else:
            self.data = [0.0]
            self.shape = (1,)
        
        self.size = len(self.data)
        self.ndim = len(self.shape)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        return MockArray(self.data[key])
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.data[key] = value
    
    def __len__(self):
        return len(self.data)
    
    def flatten(self):
        return MockArray(self.data)
    
    def copy(self):
        return MockArray(self.data.copy(), self.shape)
    
    def tolist(self):
        return self.data.copy()
    
    def reshape(self, new_shape):
        # Simple reshape - just change shape metadata
        return MockArray(self.data, new_shape)

def zeros(shape, dtype=None):
    """Mock np.zeros"""
    if isinstance(shape, int):
        return MockArray([0.0] * shape)
    elif isinstance(shape, tuple):
        total_size = 1
        for dim in shape:
            total_size *= dim
        return MockArray([0.0] * total_size, shape)
    return MockArray([0.0])

def ones(shape, dtype=None):
    """Mock np.ones"""
    if isinstance(shape, int):
        return MockArray([1.0] * shape)
    elif isinstance(shape, tuple):
        total_size = 1
        for dim in shape:
            total_size *= dim
        return MockArray([1.0] * total_size, shape)
    return MockArray([1.0])

def array(data, dtype=None):
    """Mock np.array"""
    return MockArray(data)

def random_normal(loc=0, scale=1, size=None):
    """Mock np.random.normal"""
    if size is None:
        return random.gauss(loc, scale)
    elif isinstance(size, int):
        return MockArray([random.gauss(loc, scale) for _ in range(size)])
    elif isinstance(size, tuple):
        total_size = 1
        for dim in size:
            total_size *= dim
        return MockArray([random.gauss(loc, scale) for _ in range(total_size)], size)

def mean(arr, axis=None, keepdims=False):
    """Mock np.mean"""
    if hasattr(arr, 'data'):
        return sum(arr.data) / len(arr.data)
    return sum(arr) / len(arr)

def max(arr, axis=None):
    """Mock np.max"""
    if hasattr(arr, 'data'):
        return max(arr.data) if arr.data else 0
    return max(arr) if arr else 0

def min(arr, axis=None):
    """Mock np.min"""
    if hasattr(arr, 'data'):
        return min(arr.data) if arr.data else 0
    return min(arr) if arr else 0

def sum(arr, axis=None, keepdims=False):
    """Mock np.sum"""
    if hasattr(arr, 'data'):
        return sum(arr.data)
    return sum(arr)

def dot(a, b):
    """Mock np.dot"""
    if hasattr(a, 'data') and hasattr(b, 'data'):
        result = 0
        for i in range(min(len(a.data), len(b.data))):
            result += a.data[i] * b.data[i]
        return result
    return 0

def outer(a, b):
    """Mock np.outer"""
    if hasattr(a, 'data') and hasattr(b, 'data'):
        result = []
        for av in a.data:
            row = [av * bv for bv in b.data]
            result.extend(row)
        return MockArray(result, (len(a.data), len(b.data)))
    return MockArray([0])

def stack(arrays, axis=0):
    """Mock np.stack"""
    if not arrays:
        return MockArray([])
    
    all_data = []
    for arr in arrays:
        if hasattr(arr, 'data'):
            all_data.extend(arr.data)
        else:
            all_data.extend(arr)
    
    return MockArray(all_data)

def exp(arr):
    """Mock np.exp"""
    if hasattr(arr, 'data'):
        return MockArray([math.exp(min(x, 700)) for x in arr.data])  # Prevent overflow
    return math.exp(min(arr, 700))

def tanh(arr):
    """Mock np.tanh"""
    if hasattr(arr, 'data'):
        return MockArray([math.tanh(x) for x in arr.data])
    return math.tanh(arr)

def log(arr):
    """Mock np.log"""
    if hasattr(arr, 'data'):
        return MockArray([math.log(max(x, 1e-8)) for x in arr.data])  # Prevent log(0)
    return math.log(max(arr, 1e-8))

def clip(arr, a_min, a_max):
    """Mock np.clip"""
    if hasattr(arr, 'data'):
        return MockArray([max(a_min, min(x, a_max)) for x in arr.data])
    return max(a_min, min(arr, a_max))

def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Mock np.allclose"""
    if hasattr(a, 'data') and hasattr(b, 'data'):
        if len(a.data) != len(b.data):
            return False
        for av, bv in zip(a.data, b.data):
            if abs(av - bv) > atol + rtol * abs(bv):
                return False
        return True
    return abs(a - b) <= atol + rtol * abs(b)

def all(arr):
    """Mock np.all"""
    if hasattr(arr, 'data'):
        return all(bool(x) for x in arr.data)
    return bool(arr)

def isfinite(arr):
    """Mock np.isfinite"""
    if hasattr(arr, 'data'):
        return MockArray([math.isfinite(x) for x in arr.data])
    return math.isfinite(arr)

def linalg_norm(arr):
    """Mock np.linalg.norm"""
    if hasattr(arr, 'data'):
        return math.sqrt(sum(x*x for x in arr.data))
    return abs(arr)

def prod(arr):
    """Mock np.prod"""
    if hasattr(arr, 'data'):
        result = 1
        for x in arr.data:
            result *= x
        return result
    return arr

def polyfit(x, y, deg):
    """Mock np.polyfit - simplified linear fit"""
    if len(x) < 2 or len(y) < 2:
        return [0.0, 0.0]
    
    # Simple linear regression for degree 1
    if deg == 1:
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return [0.0, sum_y / n]
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        return [slope, intercept]
    
    return [0.0] * (deg + 1)

# Mock numpy module structure
class MockLinalg:
    norm = linalg_norm

class MockRandom:
    normal = random_normal
    
    @staticmethod
    def random():
        return random.random()
    
    @staticmethod
    def choice(arr):
        return random.choice(arr)
    
    @staticmethod
    def binomial(n, p, size=None):
        if size is None:
            return 1 if random.random() < p else 0
        elif isinstance(size, tuple):
            total_size = 1
            for dim in size:
                total_size *= dim
            data = [1 if random.random() < p else 0 for _ in range(total_size)]
            return MockArray(data, size)
        else:
            return MockArray([1 if random.random() < p else 0 for _ in range(size)])
    
    @staticmethod
    def uniform(low, high, size=None):
        if size is None:
            return random.uniform(low, high)
        elif isinstance(size, int):
            return MockArray([random.uniform(low, high) for _ in range(size)])
        return MockArray([random.uniform(low, high)])

# Export mock numpy interface
float32 = float
ndarray = MockArray
linalg = MockLinalg()
random = MockRandom()