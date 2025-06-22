# This code prints a lower triangle pattern using "*".
rows = int(input("Enter the number of rows: "))
for i in range(1, rows + 1):
    print("* " * i)

# Output:
# Enter the number of rows: 4
# * 
# * *
# * * *
# * * * *
