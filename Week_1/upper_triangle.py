# This code prints a upper triangle pattern using "*".
rows= int(input("Enter the number of rows: "))
for i in range(rows, 0, -1):
    spaces = "  " * (rows - i)
    stars = "* " * i
    print(spaces + stars)


# Output:
# Enter the number of rows: 5
# * * * * * 
#   * * * *
#     * * *
#       * *
#         *