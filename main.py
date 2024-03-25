import pydoc
import numpy as np
import pandas as pd
from functools import wraps
import copy
import timeit

executed_times = dict()

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = f(*args, **kwargs)
        ellapsed_time = timeit.default_timer() - start_time
        executed_times[f'{f.__name__}_{np.random.randint(1000, 10000)}'] = ellapsed_time
        return result
    return wrapper


def RSHash(s):
    b = 378551
    a = 63689
    hash = 0
    for char in s:
        hash = hash * a + ord(char)
        a = a * b
    return hash


class BinaryTree:
    class Node:
        def __init__(self, key: int, data: list):
            self.key = key
            self.data = data
            self.left = None
            self.right = None

    def insert(self, node: 'Node' or None, key: int, data: list):
        if node is None:
            return self.Node(key, data)
        if key < node.key:
            node.left = self.insert(node.left, key, data)
        elif key > node.key:
            node.right = self.insert(node.right, key, data)
        return node

    def search(self, root: 'Node', key: int) -> 'Node' or None:
        if root is None or root.key == key:
            return root
        if root.key < key:
            return self.search(root.right, key)
        return self.search(root.left, key)

    @timing
    def search_wrapper(self, root: 'Node', key: int):
        return self.search(root, key)


class RedBlackTree:
    class Node:
        def __init__(self, item, data = None):
            self.item = item
            self.data = data
            self.parent = None  # parent node
            self.left = None  # left node
            self.right = None  # right node
            self.color = 1  # 1=red , 0 = black

    def __init__(self):
        self.TNULL = self.Node(0)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL

    # Preorder
    def pre_order_helper(self, node):
        if node != self.TNULL:
            self.pre_order_helper(node.left)
            self.pre_order_helper(node.right)

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    # Balance the tree after insertion
    def fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right

                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    def preorder(self):
        self.pre_order_helper(self.root)

    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    def successor(self, x):
        if x.right != self.TNULL:
            return self.minimum(x.right)

        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(self, x):
        if (x.left != self.TNULL):
            return self.maximum(x.left)

        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key, data):
        node = self.Node(key)
        node.data = data
        node.parent = None
        node.item = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if node.item < x.item:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y == None:
            self.root = node
        elif node.item < y.item:
            y.left = node
        else:
            y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self.fix_insert(node)

    def get_root(self):
        return self.root

    @timing
    def search(self, root, key) -> 'Node' or None:
        if root is None or root.item == key:
            return root
        if root.item < key:
            return self.search(root.right, key)
        return self.search(root.left, key)


class HashTable:
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    def set_val(self, key, val):
        hashed_key = RSHash(key) % self.size
        bucket = self.hash_table[hashed_key]
        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:
                bucket[index] = (key, val)
                return
        bucket.append([key, val])

    @timing
    def get_val(self, key):
        hashed_key = RSHash(key) % self.size
        bucket = self.hash_table[hashed_key]
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:
                found_key = True
                if found_key:
                    return record_val
                break
        else:
            return "No record found"

    def delete_val(self, key):
        hashed_key = RSHash(key) % self.size
        bucket = self.hash_table[hashed_key]
        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:
                found_key = True
                if found_key:
                    bucket.pop(index)
                break
        return

    def get_collisions(self):
        return [len(bk) for bk in self.hash_table if len(bk) > 1]

    def __str__(self):
        return "".join(str(item) for item in self.hash_table)



if __name__ == '__main__':
    # prepare_data()
    for set_number in range(7):
        current_data = pd.read_csv(f'data/dataset_{set_number}')
        data_as_array = np.array(
            [current_data['Номер рейса'],
             current_data['Название авиакомпании'],
             current_data['Дата прилета'],
             current_data['Время  прилета по расписанию'],
             current_data['Число пассажиров на борту']]
        )
        dataset = [list(data_as_array[:, ind]) for ind in range(0, current_data.shape[0])]

        bt = BinaryTree()
        bst = RedBlackTree()
        hash_table = HashTable(700000)
        build_in_hashtable = dict()
        root = None
        root = bt.insert(root, RSHash(dataset[0][1]), copy.copy(dataset[0]))
        bst.insert(RSHash(dataset[0][1]), copy.copy(dataset[0]))
        # hash_table.set_val(dataset[0])

        for record in dataset[1:]:
            bt.insert(root, RSHash(record[3]), copy.copy(record))
            record[0] += 1
            bst.insert(RSHash(record[3]), copy.copy(record))
            hash_table.set_val(record[3], record)
            build_in_hashtable[record[3]] = record

        key = np.random.choice(data_as_array[3])
        try:
            print(bt.search_wrapper(root, RSHash(key)).data)
        except AttributeError:
            print('Ключа нет')

        key = np.random.choice(data_as_array[3])
        try:
            print(bst.search(bst.root, RSHash(key)).data)
        except AttributeError:
            print('Ключа нет')

        key = np.random.choice(data_as_array[3])
        print(hash_table.get_val(key))
        print(build_in_hashtable[key])

        print(sum([val - 1 for val in  hash_table.get_collisions()]))

    print(executed_times)
    pydoc.writedoc("main")
