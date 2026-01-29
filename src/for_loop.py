def print_numbers(numbers):
    for i in range(numbers):
        idx = i + 1
        print(f'readable index: {idx}')
        print(f'actual index: {i}')

def main():
    print_numbers(10)

if __name__ == '__main__':
    main()