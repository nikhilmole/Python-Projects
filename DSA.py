#///////////////////////////////////////////////////////////
#
#   Project name : Data Structure Library
#
#///////////////////////////////////////////////////////////


#######################################################
##
##  Code of Singly Linked List
##
#######################################################

from typing import Generic, TypeVar, Optional

T = TypeVar('T')

class nodeSL(Generic[T]):
    def __init__(self, inum: T):
        self.data = inum
        self.next: Optional['nodeSL[T]'] = None

class SinglyLinkedList(Generic[T]):
    def __init__(self):
        self.First: Optional[nodeSL[T]] = None
        self.count = 0

    def Display(self):
        temp = self.First
        if self.First is None:
            print("Linked list is empty")
            return
        while temp is not None:
            print(f"|{temp.data}|->", end="")
            temp = temp.next
        print("None")

    def InsertFirst(self, Value: T):
        new_node = nodeSL(Value)
        if self.First is None:
            self.First = new_node
        else:
            new_node.next = self.First
            self.First = new_node
        self.count += 1

    def InsertLast(self, Value: T):
        new_node = nodeSL(Value)
        if self.First is None:
            self.First = new_node
        else:
            temp = self.First
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node
        self.count += 1

    def CountNodes(self) -> int:
        return self.count

    def DeleteFirst(self):
        if self.First is None:
            print("Unable to delete node because linked list is empty")
            return
        if self.First.next is None:
            self.First = None
        else:
            self.First = self.First.next
        self.count -= 1

    def DeleteLast(self):
        if self.First is None:
            print("Unable to delete node because linked list is empty")
            return
        if self.First.next is None:
            self.First = None
        else:
            temp = self.First
            while temp.next.next is not None:
                temp = temp.next
            temp.next = None
        self.count -= 1

    def InsertAtPos(self, Value: T, Pos: int):
        if Pos < 1 or Pos > self.count + 1:
            print("Invalid position")
            return
        if Pos == 1:
            self.InsertFirst(Value)
        elif Pos == self.count + 1:
            self.InsertLast(Value)
        else:
            new_node = nodeSL(Value)
            temp = self.First
            for _ in range(1, Pos - 1):
                temp = temp.next
            new_node.next = temp.next
            temp.next = new_node
            self.count += 1

    def DeleteAtPos(self, Pos: int):
        if Pos < 1 or Pos > self.count:
            print("Invalid position")
            return
        if Pos == 1:
            self.DeleteFirst()
        elif Pos == self.count:
            self.DeleteLast()
        else:
            temp = self.First
            for _ in range(1, Pos - 1):
                temp = temp.next
            temp.next = temp.next.next
            self.count -= 1
##############################################################
##
##  Code of Doubly Linked List
##
##############################################################
class nodeDL(Generic[T]):
    def __init__(self, num: T):
        self.data: T = num
        self.next: Optional['nodeDL[T]'] = None
        self.prev: Optional['nodeDL[T]'] = None

class DoublyLL(Generic[T]):
    def __init__(self):
        self.First: Optional[nodeDL[T]] = None
        self.Count: int = 0
    
    def Display(self):
        if self.First is None:
            print("Linked list is empty")
            return

        temp = self.First
        while temp is not None:
            print(f"|{temp.data}|<=>", end="")
            temp = temp.next
        print("None")

    def CountNodes(self) -> int:
        return self.Count
    
    def InsertFirst(self, Value: T):
        new_node = nodeDL(Value)
        if self.First is None:
            self.First = new_node
        else:
            new_node.next = self.First
            self.First.prev = new_node
            self.First = new_node
        self.Count += 1
    
    def InsertLast(self, Value: T):
        new_node = nodeDL(Value)
        if self.First is None:
            self.First = new_node
        else:
            temp = self.First
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node
            new_node.prev = temp
        self.Count += 1

    def DeleteFirst(self):
        if self.First is None:
            print("Unable to delete node because Linked list is empty")
            return
        
        if self.First.next is None:
            self.First = None
        else:
            self.First = self.First.next
            self.First.prev = None
        self.Count -= 1 

    def DeleteLast(self):
        if self.First is None:
            print("Linked list is empty")
            return
         
        if self.First.next is None:
            self.First = None
        else:
            temp = self.First
            while temp.next.next is not None:
                temp = temp.next
            temp.next = None
        self.Count -= 1
    
    def InsertAtPose(self, Value: T, Pose: int):
        if Pose < 1 or Pose > self.Count + 1:
            print("Invalid Position")
            return
        if Pose == 1:
            self.InsertFirst(Value)
        elif Pose == self.Count + 1:
            self.InsertLast(Value)
        else:
            new_node = nodeDL(Value)
            temp = self.First
            for _ in range(1, Pose - 1):
                temp = temp.next
            new_node.next = temp.next
            temp.next.prev = new_node
            temp.next = new_node
            new_node.prev = temp
            self.Count += 1
            
    def DeleteAtPose(self, Pose: int):
        if Pose < 1 or Pose > self.Count:
            print("Unable to delete node because invalid position")
            return 
        
        if Pose == 1:
            self.DeleteFirst()
        elif Pose == self.Count:
            self.DeleteLast()
        else:
            temp = self.First
            for _ in range(1, Pose - 1):
                temp = temp.next
            temp.next = temp.next.next
            if temp.next is not None:
                temp.next.prev = temp
            self.Count -= 1 
#########################################################
##
##  Code of Singly Circular 
##
#########################################################
class nodeSCL(Generic[T]):
    def __init__(self, num: T):
        self.next: Optional['nodeSCL[T]'] = None
        self.data: T = num

class SinglyCircular(Generic[T]):
    def __init__(self):
        self.First: Optional[nodeSCL[T]] = None
        self.Last: Optional[nodeSCL[T]] = None
        self.Count: int = 0
    
    def Display(self):
        if (self.First is None) and (self.Last is None):
            print("LL is empty")
            return
        temp = self.First

        while True:
            print(f"| {temp.data} |->", end=" ")
            temp = temp.next
            if temp == self.First:
                break
        print("None")
    
    def CountNode(self) -> int:
        return self.Count
    
    def InsertFirst(self, Value: T):
        new_node = nodeSCL(Value)
        if (self.First is None) and (self.Last is None):
            self.First = new_node
            self.Last = new_node
        else:
            new_node.next = self.First
            self.First = new_node
        self.Last.next = self.First 

        self.Count += 1

    def InsertLast(self, Value: T):
        new_node = nodeSCL(Value)
        if (self.First is None) and (self.Last is None):
            self.First = new_node
            self.Last = new_node
        else:
            self.Last.next = new_node
            self.Last = new_node
        self.Last.next = self.First 

        self.Count += 1

    def DeleteFirst(self):
        if (self.First is None) and (self.Last is None):
            print("Linked list is empty")
            return
        
        if self.First == self.Last:
            self.First = None
            self.Last = None
        else:
            self.First = self.First.next
            self.Last.next = self.First

        self.Count -= 1

    def DeleteLast(self):
        if (self.First is None) and (self.Last is None):
            print("Linked list is empty")
            return
        
        if self.First == self.Last:
            self.First = None
            self.Last = None
        else:
            temp = self.First
            while temp.next != self.Last:
                temp = temp.next
            
            self.Last = temp
            self.Last.next = self.First

        self.Count -= 1

    def InsertAtPose(self, Value: T, Pose: int):
        if Pose < 1 or Pose > self.Count + 1:
            print("Invalid option")
            return
        
        if Pose == 1:
            self.InsertFirst(Value)
        elif Pose == self.Count + 1:
            self.InsertLast(Value)
        else:
            temp = self.First
            new_node = nodeSCL(Value)

            for _ in range(1, Pose - 1):
                temp = temp.next
            
            new_node.next = temp.next
            temp.next = new_node

            self.Count += 1

    def DeleteAtPose(self, Pose: int):
        if Pose < 1 or Pose > self.Count:
            print("Invalid option")
            return
            
        if Pose == 1:
            self.DeleteFirst()
        elif Pose == self.Count:
            self.DeleteLast()
        else:
            temp = self.First
            
            for _ in range(1, Pose - 1):
                temp = temp.next
            
            temp.next = temp.next.next      
                 
            self.Count -= 1

#########################################################
##
##  Code of Doubly Circular 
##
#########################################################
class nodeDCL(Generic[T]):
    def __init__(self, num: T):
        self.data: T = num
        self.next: Optional['nodeDCL[T]'] = None
        self.prev: Optional['nodeDCL[T]'] = None

class DoublyCircular(Generic[T]):
    def __init__(self):
        self.Count: int = 0
        self.First: Optional[nodeDCL[T]] = None
        self.Last: Optional[nodeDCL[T]] = None

    def CountNodes(self) -> int:
        return self.Count

    def Display(self):
        if self.First is None and self.Last is None:
            print("Linked list is empty")
            return
        
        temp = self.First
        while True:
            print(f"| {temp.data} |<=>", end=" ")
            temp = temp.next
            if temp == self.First:
                break
        print("None")

    def InsertFirst(self, Value: T):
        new_node = nodeDCL(Value)

        if self.First is None and self.Last is None:
            self.First = new_node
            self.Last = new_node
        else:
            new_node.next = self.First
            self.First.prev = new_node
            self.First = new_node

        self.Last.next = self.First
        self.First.prev = self.Last

        self.Count += 1

    def InsertLast(self, Value: T):
        new_node = nodeDCL(Value)

        if self.First is None and self.Last is None:
            self.First = new_node
            self.Last = new_node
        else:
            self.Last.next = new_node
            new_node.prev = self.Last
            self.Last = new_node

        self.Last.next = self.First
        self.First.prev = self.Last

        self.Count += 1

    def DeleteFirst(self):
        if self.First is None and self.Last is None:
            print("Linked list is empty")
            return

        if self.First == self.Last:
            self.First = None
            self.Last = None
        else:
            self.First = self.First.next
            self.Last.next = self.First
            self.First.prev = self.Last

        self.Count -= 1

    def DeleteLast(self):
        if self.First is None and self.Last is None:
            print("Linked list is empty")
            return

        if self.First == self.Last:
            self.First = None
            self.Last = None
        else:
            self.Last = self.Last.prev
            self.Last.next = self.First
            self.First.prev = self.Last

        self.Count -= 1

    def InsertAtPose(self, Value: T, Pose: int):
        if Pose < 1 or Pose > self.Count + 1:
            print("Invalid Position")
            return

        if Pose == 1:
            self.InsertFirst(Value)
        elif Pose == self.Count + 1:
            self.InsertLast(Value)
        else:
            new_node = nodeDCL(Value)
            temp = self.First
            for _ in range(1, Pose - 1):
                temp = temp.next
            
            new_node.next = temp.next
            temp.next.prev = new_node
            new_node.prev = temp
            temp.next = new_node

            self.Count += 1

    def DeleteAtPose(self, Pose: int):
        if Pose < 1 or Pose > self.Count:
            print("Invalid Position")
            return

        if Pose == 1:
            self.DeleteFirst()
        elif Pose == self.Count:
            self.DeleteLast()
        else:
            temp = self.First
            for _ in range(1, Pose - 1):
                temp = temp.next

            temp.next = temp.next.next
            if temp.next:
                temp.next.prev = temp

            self.Count -= 1
#######################################################################
##
##  Code of Binary Search Tree
##
#######################################################################
class NodeBST(Generic[T]):
    def __init__(self, data: T):
        self.data: T = data
        self.lchild: Optional['NodeBST[T]'] = None
        self.rchild: Optional['NodeBST[T]'] = None

class BinaryTree(Generic[T]):
    def __init__(self):
        self.root: Optional[NodeBST[T]] = None
        self.leaf_count: int = 0
        self.parent_count: int = 0
        self.node_count: int = 0

    def insert(self, data: T):
        new_node = NodeBST(data)
        if self.root is None:
            self.root = new_node
        else:
            temp = self.root
            while True:
                if temp.data == data:
                    print("Unable to insert node as element is already present")
                    break
                elif data > temp.data:
                    if temp.rchild is None:
                        temp.rchild = new_node
                        break
                    temp = temp.rchild
                else:
                    if temp.lchild is None:
                        temp.lchild = new_node
                        break
                    temp = temp.lchild

    def inorder(self, root: Optional[NodeBST[T]]):
        if root is not None:
            self.inorder(root.lchild)
            print(root.data)
            self.inorder(root.rchild)

    def preorder(self, root: Optional[NodeBST[T]]):
        if root is not None:
            print(root.data)
            self.preorder(root.lchild)
            self.preorder(root.rchild)

    def postorder(self, root: Optional[NodeBST[T]]):
        if root is not None:
            self.postorder(root.lchild)
            self.postorder(root.rchild)
            print(root.data)

    def search(self, root: Optional[NodeBST[T]], data: T) -> bool:
        while root is not None:
            if root.data == data:
                return True
            elif data > root.data:
                root = root.rchild
            else:
                root = root.lchild
        return False

    def count_leaf(self, root: Optional[NodeBST[T]]) -> int:
        if root is not None:
            if root.lchild is None and root.rchild is None:
                self.leaf_count += 1
            self.count_leaf(root.lchild)
            self.count_leaf(root.rchild)
        return self.leaf_count

    def count_parent(self, root: Optional[NodeBST[T]]) -> int:
        if root is not None:
            if root.lchild is not None or root.rchild is not None:
                self.parent_count += 1
            self.count_parent(root.lchild)
            self.count_parent(root.rchild)
        return self.parent_count

    def count_all(self, root: Optional[NodeBST[T]]) -> int:
        if root is not None:
            self.node_count += 1
            self.count_all(root.lchild)
            self.count_all(root.rchild)
        return self.node_count
    
########################################################################
##
##  Code of Searching Sorting
##
########################################################################
class ArrayX(Generic[T]):
    def __init__(self, Value: int):
        self.Arr: List[T] = [None] * Value
        self.Size: int = Value

    def Accept(self):
        print("Enter the elements: ")
        for i in range(self.Size):
            element = input()  # Accepting input as string
            self.Arr[i] = self.convert(element)

    def convert(self, element: str) -> T:
        """Convert the input element to the appropriate type."""
        # Default conversion is to string
        return element  # Modify this method as needed for other types

    def Display(self):
        for i in range(self.Size):
            print(self.Arr[i], end=" ")
        print()

    def LinearSearch(self, iNo: T) -> bool:
        for i in range(self.Size):
            if self.Arr[i] == iNo:
                return True
        return False
    
    def BidirectionalSearch(self, iNo: T) -> bool:
        start = 0
        end = self.Size - 1

        while start <= end:
            if self.Arr[start] == iNo or self.Arr[end] == iNo:
                return True
            start += 1
            end -= 1

        return False
    
    def BinarySearch(self, iNo: T) -> bool:
        start = 0
        end = self.Size - 1

        while start <= end:
            mid = start + ((end - start) // 2)
            if self.Arr[mid] == iNo:
                return True
            elif iNo < self.Arr[mid]:
                end = mid - 1
            else:
                start = mid + 1

        return False
    
    def BubbleSort(self):
        for i in range(self.Size - 1):
            for j in range(self.Size - 1 - i):
                if self.Arr[j] > self.Arr[j + 1]:
                    self.Arr[j], self.Arr[j + 1] = self.Arr[j + 1], self.Arr[j]
            print(f"\nArray after pass {i + 1}:")
            self.Display()

    def BubbleSortEfficient(self):
        for i in range(self.Size - 1):
            swapped = False
            for j in range(self.Size - 1 - i):
                if self.Arr[j] > self.Arr[j + 1]:
                    self.Arr[j], self.Arr[j + 1] = self.Arr[j + 1], self.Arr[j]
                    swapped = True
            if not swapped:
                break
            print(f"\nArray after pass {i + 1}:")
            self.Display()

    def SelectionSort(self):
        for i in range(self.Size - 1):
            min_index = i
            for j in range(i + 1, self.Size):
                if self.Arr[j] < self.Arr[min_index]:
                    min_index = j
            self.Arr[i], self.Arr[min_index] = self.Arr[min_index], self.Arr[i]

    def InsertionSort(self):
        for i in range(1, self.Size):
            selected = self.Arr[i]
            j = i - 1
            while j >= 0 and self.Arr[j] > selected:
                self.Arr[j + 1] = self.Arr[j]
                j -= 1
            self.Arr[j + 1] = selected
#########################################################################
##
##  Code of Stack 
##
#########################################################################
class NodeS(Generic[T]):
    def __init__(self, num: T):
        self.next: Optional[NodeS[T]] = None
        self.data: T = num

class Stack(Generic[T]):
    def __init__(self):
        self.First: Optional[NodeS[T]] = None
        self.Count: int = 0

    def Push(self, Value: T):
        new_node = NodeS(Value)

        if self.First is None:
            self.First = new_node
        else:
            new_node.next = self.First
            self.First = new_node

        self.Count += 1

    def Pop(self) -> T:
        if self.First is None:
            print("Stack is empty")
            return None 
        
        Value = self.First.data
        self.First = self.First.next

        self.Count -= 1
        return Value

    def Display(self):
        temp = self.First

        while temp is not None:
            print(temp.data)
            temp = temp.next

    def Count(self) -> int:
        return self.Count
    
#########################################################################
##
##  Code of Queue
##
#########################################################################
class NodeQ(Generic[T]):
    def __init__(self, Value: T):
        self.data: T = Value
        self.next: Optional[NodeQ[T]] = None

class Queue(Generic[T]):
    def __init__(self):
        self.First: Optional[NodeQ[T]] = None
        self.Last: Optional[NodeQ[T]] = None
        self.Count: int = 0

    def Display(self):
        temp = self.First

        while temp is not None:
            print(f"{temp.data}  ", end=" ")
            temp = temp.next

        print()

    def Count(self) -> int:
        return self.Count
    
    def Enqueue(self, Value: T):
        new_node = NodeQ(Value)
        
        if self.First is None:
            self.First = new_node
            self.Last = new_node
        else:
            self.Last.next = new_node
            self.Last = new_node
        
        self.Count += 1

    def Dequeue(self) -> T:
        if self.First is None:
            print("Queue is empty")
            return None 
        
        Value = self.First.data
        self.First = self.First.next

        if self.First is None:
            self.Last = None

        self.Count -= 1
        return Value
#########################################################################
##
##  Code of Matrices
##
#########################################################################
class Matrix(Generic[T]):
    def __init__(self, rows: int, cols: int):
        self.Rows = rows
        self.Cols = cols
        self.Arr: List[List[T]] = [[0 for _ in range(self.Cols)] for _ in range(self.Rows)]

    def Accept(self):
        print("Please enter elements")
        for i in range(self.Rows):
            for j in range(self.Cols):
                self.Arr[i][j] = self._input_element()

    def _input_element(self) -> T:
        while True:
            try:
                value = input()
                if isinstance(self.Arr[0][0], int):
                    return int(value)  
                elif isinstance(self.Arr[0][0], float):
                    return float(value)
                else:
                    raise ValueError("Unsupported type")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def Display(self):
        for i in range(self.Rows):
            for j in range(self.Cols):
                print(f"{self.Arr[i][j]} ", end=" ")
            print()

    def Summation(self) -> T:
        total = type(self.Arr[0][0])(0)  
        for i in range(self.Rows):
            for j in range(self.Cols):
                total += self.Arr[i][j]
        return total

    def Maximum(self) -> T:
        max_value = self.Arr[0][0]
        for i in range(self.Rows):
            for j in range(self.Cols):
                if max_value < self.Arr[i][j]:
                    max_value = self.Arr[i][j]
        return max_value

    def Minimum(self) -> T:
        min_value = self.Arr[0][0]
        for i in range(self.Rows):
            for j in range(self.Cols):
                if min_value > self.Arr[i][j]:
                    min_value = self.Arr[i][j]
        return min_value

    def RowSum(self):
        for i in range(self.Rows):
            row_sum = type(self.Arr[0][0])(0)
            for j in range(self.Cols):
                row_sum += self.Arr[i][j]
            print(f"Summation of row {i} is: {row_sum}")

    def ColSum(self):
        for j in range(self.Cols):
            col_sum = type(self.Arr[0][0])(0)
            for i in range(self.Rows):
                col_sum += self.Arr[i][j]
            print(f"Summation of column {j} is: {col_sum}")

    def DiagonalSum(self) -> T:
        if self.Rows != self.Cols:
            print("Unable to perform addition because matrix is not square")
            return -1
        
        diagonal_sum = type(self.Arr[0][0])(0)
        for i in range(self.Rows):
            diagonal_sum += self.Arr[i][i]
        return diagonal_sum

    def SumEven(self) -> T:
        even_sum = type(self.Arr[0][0])(0)
        for i in range(self.Rows):
            for j in range(self.Cols):
                if self.Arr[i][j] % 2 == 0:
                    even_sum += self.Arr[i][j]
        return even_sum

    def SumOdd(self) -> T:
        odd_sum = type(self.Arr[0][0])(0)
        for i in range(self.Rows):
            for j in range(self.Cols):
                if self.Arr[i][j] % 2 != 0:
                    odd_sum += self.Arr[i][j]
        return odd_sum    

    def UpdateMatrix(self):
        for i in range(self.Rows):
            for j in range(self.Cols):
                if self.Arr[i][j] % 5 == 0:
                    self.Arr[i][j] = type(self.Arr[0][0])(0)

    def UpdateMatrixEven(self):
        for i in range(self.Rows):
            for j in range(self.Cols):
                if self.Arr[i][j] % 2 == 0:
                    self.Arr[i][j] += 1 

    def AdditionDigit(self):
        for i in range(self.Rows):
            for j in range(self.Cols):
                number = self.Arr[i][j]
                digit_sum = 0
                while number > 0:
                    digit_sum += number % 10
                    number //= 10
                self.Arr[i][j] = digit_sum

    def SwapRow(self):
        for i in range(0, self.Rows - 1, 2):
            for j in range(self.Cols):
                self.Arr[i][j], self.Arr[i + 1][j] = self.Arr[i + 1][j], self.Arr[i][j]

    def ReverseRow(self):
        for i in range(self.Rows):
            start = 0
            end = self.Cols - 1
            while start < end:
                self.Arr[i][start], self.Arr[i][end] = self.Arr[i][end], self.Arr[i][start]
                start += 1
                end -= 1

    def ReverseColumn(self):
        for j in range(self.Cols):
            start = 0
            end = self.Rows - 1
            while start < end:
                self.Arr[start][j], self.Arr[end][j] = self.Arr[end][j], self.Arr[start][j]
                start += 1
                end -= 1

    def Transpose(self):
        for i in range(self.Rows):
            for j in range(i + 1, self.Cols):
                self.Arr[i][j], self.Arr[j][i] = self.Arr[j][i], self.Arr[i][j]
                
def main():
    pass

if __name__ == "__main__":
    main()
