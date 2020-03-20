import enum
import itertools


def create_fruit_labels(fruits=("apple", "banana", "mix")):
    FruitLabel = enum.Enum('FruitLabel', zip(fruits, itertools.count()))
    return FruitLabel

if __name__ == '__main__':
    fruits = ("apple", "banana", "mix")
    FruitLabel = create_fruit_labels(fruits=fruits)
    for fruit in FruitLabel:
        print("Fruit name: {}\t\tFruit enum value: {}".format(fruit.name, fruit.value))

    # example of converting enum value to name
    print(FruitLabel(0).name)
