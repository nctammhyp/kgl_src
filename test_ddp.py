import os

RANK = int(os.environ["RANK"])

def say_hello(name):
    print(f"Hello {name}")

if __name__ == "__main__":
    names = [
        "Barry",
        "Alice",
        "Barbara",
        "Tom",
    ]
    say_hello(names[RANK])
