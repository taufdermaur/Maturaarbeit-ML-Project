def elementwise_multiplication(vec_a, vec_b):
    """Multipliziert zwei Vektoren elementweise."""
    return [a * b for a, b in zip(vec_a, vec_b)]

def elementwise_addition(vec_a, vec_b):
    """Addiert zwei Vektoren elementweise."""
    return [a + b for a, b in zip(vec_a, vec_b)]

def vector_sum(vec_a):
    """Berechnet die Summe aller Elemente in einem Vektor."""
    return sum(vec_a)

def vector_average(vec_a):
    """Berechnet den Durchschnitt der Elemente in einem Vektor."""
    return sum(vec_a) / len(vec_a) if vec_a else 0  # Verhindert Division durch Null

def scalar_product(vec_a, vec_b):
    """Berechnet das Skalarprodukt zweier Vektoren."""
    return vector_sum(elementwise_multiplication(vec_a, vec_b))


v1 = [1, 2, 3]
v2 = [4, 5, 6]

print(elementwise_multiplication(v1, v2))  
print(elementwise_addition(v1, v2))       
print(vector_sum(v1))                     
print(vector_average(v1))   
print(scalar_product(v1, v2))               

