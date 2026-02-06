import heapq


class IndexedHeap:
    def __init__(self, *, max_heap: bool = False):
        self.heap = []              # list of [priority, key]
        self.position = {}          # key -> index in heap
        self.max_heap = max_heap

    # ---------- comparison helper ----------

    def _higher_priority(self, a, b):
        """Return True if priority a should be above b in heap"""
        return a > b if self.max_heap else a < b

    # ---------- internal helpers ----------

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.position[self.heap[i][1]] = i
        self.position[self.heap[j][1]] = j

    def _siftdown(self, idx):
        heap = self.heap
        while idx > 0:
            parent = (idx - 1) // 2
            if self._higher_priority(heap[idx][0], heap[parent][0]):
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _siftup(self, idx):
        heap = self.heap
        n = len(heap)
        while True:
            left = 2 * idx + 1
            right = left + 1
            best = idx

            if left < n and self._higher_priority(heap[left][0], heap[best][0]):
                best = left
            if right < n and self._higher_priority(heap[right][0], heap[best][0]):
                best = right

            if best == idx:
                break

            self._swap(idx, best)
            idx = best

    # ---------- public API ----------

    def push(self, key, priority):
        if key in self.position:
            raise KeyError("Key already exists")

        idx = len(self.heap)
        self.heap.append([priority, key])
        self.position[key] = idx
        self._siftdown(idx)

    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")

        top_priority, top_key = self.heap[0]
        last = self.heap.pop()
        del self.position[top_key]

        if self.heap:
            self.heap[0] = last
            self.position[last[1]] = 0
            self._siftup(0)

        return top_key, top_priority

    def set_priority(self, key, priority):
        """Set absolute priority (recommended for non-numeric priorities)"""
        if key not in self.position:
            raise KeyError("Key not found")

        idx = self.position[key]
        old_priority = self.heap[idx][0]
        self.heap[idx][0] = priority

        if self._higher_priority(priority, old_priority):
            self._siftdown(idx)
        else:
            self._siftup(idx)

    def get(self, key):
        return self.heap[self.position[key]][0]

    def __contains__(self, key):
        return key in self.position

    def __len__(self):
        return len(self.heap)