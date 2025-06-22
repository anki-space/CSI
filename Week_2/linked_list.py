class Node:
    def __init__(self, data):
        self.data = data  # Node's data
        self.next = None  # Pointer to the next node

class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the list to None

    def add_node(self, data):
        """Add a node to the end of the linked list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node  # If the list is empty, make the new node the head
        else:
            current = self.head
            while current.next:  # Traverse to the last node
                current = current.next
            current.next = new_node  # Add the new node at the end

    def print_list(self):
        """Print the entire linked list."""
        if self.head is None:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")  # End of the list

    def delete_node(self, n):
        """Delete the nth node (1-based index) from the linked list."""
        if self.head is None:
            raise IndexError("Cannot delete from an empty list.")
        
        # If deleting the first node
        if n == 1:
            self.head = self.head.next
            return
        
        current = self.head
        count = 1
        
        while current and current.next:
            if count == n - 1:
                if current.next:
                    current.next = current.next.next
                    return
            current = current.next
            count += 1
        
        # If n is out of range
        raise IndexError(f"Index {n} is out of range.")
        
# Testing the LinkedList implementation
if __name__ == "__main__":
    # Creating the linked list and adding some nodes
    ll = LinkedList()
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    ll.add_node(50)
    
    print("Original List:")
    ll.print_list()  
    
    # Deleting the 3rd node (index starts from 1)
    print("\nDeleting the 3rd node:")
    ll.delete_node(3)
    ll.print_list()  # Expected output: 10 -> 20 -> 40 -> 50 -> None
    
    # Trying to delete a node that doesn't exist (out of range)
    try:
        print("\nTrying to delete the 10th node (out of range):")
        ll.delete_node(10)
    except IndexError as e:
        print(e)  # Expected output: Index 10 is out of range.
    
    # Deleting the first node
    print("\nDeleting the 1st node:")
    ll.delete_node(1)
    ll.print_list()  # Expected output: 20 -> 40 -> 50 -> None
    
    # Deleting the last node
    print("\nDeleting the last node:")
    ll.delete_node(3)
    ll.print_list()  # Expected output: 20 -> 40 -> None

# output:
# Original List:
# 10 -> 20 -> 30 -> 40 -> 50 -> None

# Deleting the 3rd node:
# 10 -> 20 -> 40 -> 50 -> None

# Trying to delete the 10th node (out of range):
# Index 10 is out of range.

# Deleting the 1st node:
# 20 -> 40 -> 50 -> None

# Deleting the last node:
# 20 -> 40 -> None