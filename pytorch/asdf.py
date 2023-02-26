def mygen():
    for i in range(5):
        yield i


a = mygen()

b = []

print(list(a))
print(list(a))