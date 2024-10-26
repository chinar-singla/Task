def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    return fib_sequence[:n]

# Input: Enter the number of terms
n = int(input("Enter the number of terms in the Fibonacci sequence: "))
print("Fibonacci sequence up to", n, "terms:")
print(fibonacci(n))
