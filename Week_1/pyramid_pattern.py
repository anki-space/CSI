# This code prints a pyramid pattern using "*".
rows = int(input("Enter the number of rows: "))
for i in range(1, rows + 1):
    spaces = "  " * (rows - i)
    stars = "* " * (2 * i - 1)
    print(spaces + stars)

#Output:

# Enter the number of rows: 5
#         * 
#       * * *
#     * * * * *
#   * * * * * * *
# * * * * * * * * *
