"utils"
import heapq

class MaxHeap:
    "store (index, value)"
    def __init__(self):
        self._data = []

    def push(self, index: int, value: float):
        "heap push"
        # priority is negative of the value
        heapq.heappush(self._data, (-value, index, value))

    def pop(self):
        "heap pop"
        if not self._data:
            raise IndexError("pop from empty heap")
        _, index, value = heapq.heappop(self._data)
        return index, value

    def peek(self):
        "heap peek"
        if not self._data:
            raise IndexError("peek from empty heap")
        _, index, value = self._data[0]
        return index, value

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        "check if heap is empty"
        return len(self._data) == 0

if __name__ == "__main__":
    heap = MaxHeap()
    heap.push(1, 10.0)
    heap.push(2, 20.0)
    heap.push(3, 15.0)
    while not heap.is_empty():
        index_, value_ = heap.pop()
        print(f"Index: {index_}, Value: {value_}")
