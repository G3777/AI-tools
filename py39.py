"""
9 9 乘法表
"""

# for i in range(1, 10):
#     for j in range(1, i + 1):
#         print(f"{j} * {i} = {i * j}", end="\t")
#     print()

# row =1
# while row <= 9:
#     col = 1
#     while col <= row:
#         print(f"{col} * {row} = {row * col}", end="\t")
#         col += 1
#     print()
#     row += 1
#     pass


# letters = 'abcdabcdabcdabcefg'

# # 创建一个字典来存储字符及其出现的次数
# char_count = {}

# # 遍历字符串中的每个字符
# for char in letters:
#     if char in char_count:
#         # 如果字符已经在字典中，增加其计数
#         char_count[char] += 1
#     else:
#         # 如果字符不在字典中，将其添加到字典并设置计数为1
#         char_count[char] = 1

# # 打印结果
# for char, count in char_count.items():
#     print(f"{char}:{count}", end=" ")


# # a. 计算列表长度并输出
# li = ['ethan', 'zoran', "iim"]
# length = len(li)
# print("列表长度:", length)

# # b. 列表中追加元素“lucy”，并输出添加后的列表
# li.append("lucy")
# print("添加后的列表:", li)

# # c. 请在列表的第1个位置插入元素“Tony"，并输出添加后的列表
# li.insert(0, "Tony")
# print("插入后的列表:", li)

# # d. 请修改列表第2个位置的元素为“Kelly"，并输出修改后的列表
# li[1] = "Kelly"
# print("修改后的列表:", li)

# # e. 请删除列表中的元素“ethan”，并输出修改后的列表
# if "ethan" in li:
#     li.remove("ethan")
# print("删除后的列表:", li)

# # f. 请删除列表中的第2个元素，并输出删除元素后的列表
# del li[1]
# print("删除元素后的列表:", li)



# 3. 给定一个列表 nums
# nums = [10, 20, 30, 50, 70, 20]

# for idx, num in enumerate(nums):
#   dict[num] = dict.get(num, []) + [idx]
#   pass
# print(dict)

# # 查询20首次出现的索引位置
# first_index = -1
# for index, value in enumerate(nums):
#     if value == 20:
#         first_index = index
#         break  # 找到后退出循环

# print("20首次出现的索引位置:", first_index)

# # 查询20出现的所有位置
# all_indices = []
# for index, value in enumerate(nums):
#     if value == 20:
#         all_indices.append(index)

# print("20出现的所有位置:", all_indices)


# names = [["张飞", "刘备", "关羽"], ["曹操", "典韦", "司马懿"]]
# result = []
# for sublist in names:
#     for item in sublist:
#         result.append(item)
# print(result)


# 定义students列表
# students = [
#     {'name': 'Tom', 'sex': '女', 'tel': '15300022839', 'age': 19, 'score': 92},
#     {'name': 'Jerry', 'sex': '男', 'tel': '15300022838', 'age': 20, 'score': 40},
#     {'name': 'Andy', 'sex': '女', 'tel': '15300022837', 'age': 18, 'score': 85},
#     {'name': 'Jack', 'sex': '男', 'tel': '15300022428', 'age': 19, 'score': 65},
#     {'name': 'Rose', 'sex': '女', 'tel': '15300022653', 'age': 17, 'score': 59},
#     {'name': 'Bob', 'sex': '男', 'tel': '15300022867', 'age': 18, 'score': 78}
# ]

# # 遍历所有的姓名
# print("所有学生姓名：")
# for student in students:
#     print(student['name'], end='| ')
# print()

# # 统计不及格学生的个数
# fail_students = [student for student in students if student['score'] < 60]
# print(f"不及格学生个数：{len(fail_students)}")
# print()

# # 打印所有男生的信息
# male_students = [student for student in students if student['sex'] == '男']
# for male_student in male_students:
#     print(male_student, end=' ')
# print()

# # 求平均分数
# total_score = sum(student['score'] for student in students)
# average_score = total_score / len(students)
# print(f"平均分数：{average_score:.2f}")
# print()


# # 输入每天卖出多少碗面
# bowls_sold_per_day = int(input("请输入每天卖出多少碗面: "))

# # 输入每碗面多少块
# price_per_bowl = float(input("请输入每碗面多少块: "))

# # 输入今年共营业多少天
# days_of_operation = int(input("请输入今年共营业多少天: "))

# # 计算一年的总销售额
# total_sales = bowls_sold_per_day * price_per_bowl * days_of_operation

# # 输出年销售额
# print(f"一年的总销售额是: {total_sales}块")


## 冒泡排序
# nums = [10, 12, 8, 11, 6, 2]
# def sort1(nums):
#     for j in range(len(nums)-1):
#             for i in range(len(nums)-1-j):
#                 if nums[i] > nums[i+1]:
#                     nums[i], nums[i+1] = nums[i+1], nums[i]
#     print(nums)
# sort1(nums)



"""
人体关键点的描述
1)人有姓名、年龄。根据下面的图形还有很多的关键点landmarks。每一个关键点的描述使用 x，y去描述要求:
创建关键点(创建3个即可)，给person人添加关键点之后。遍历每一个关键点
class Person:
name
age
landmarks
"""
# class Person:
#     def __init__(self, name, age, landmarks=None):
#         self.name = name
#         self.age = age
#         if landmarks is None:
#             self.landmarks = []
#         else:
#             self.landmarks = landmarks

#     def add_landmark(self, x, y):
#         self.landmarks.append((x, y))

#     def review_landmarks(self):
#         for landmark in self.landmarks:
#             print(landmark)

# # 创建一个Person对象，不提供landmarks参数
# person1 = Person("Alice", 30)
# person1.add_landmark("head", 100)
# person1.review_landmarks()

# # 创建一个Person对象，并提供landmarks参数
# person2 = Person("Bob", 25, [("left_hand", 20), ("right_hand", 40)])
# person2.review_landmarks()

"""
人体关键点的描述
1)人有姓名、年龄。根据下面的图形还有很多的关键点landmarks。每一个关键点的描述使用 x，y去描述要求:
创建关键点(创建3个即可)，给person人添加关键点之后。遍历每一个关键点
plus
需要区分出 左右手关键点
plus +
手：拇指、食指、中指、无名指、小拇指
人有两只手 每只手有5个手指头，每个手指有多个关键点
"""

"""
class Mark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Finger:
    def __init__(self, type):
        self.landmarks = []
        self.type = type

    def add_landmark(self, mark):
        self.landmarks.append(mark)

class Hand:
    def __init__(self, type):
        self.landmarks = []
        self.type = type

    def add_landmark(self, mark):
        self.landmarks.append(mark)

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        # 关键点
        self.landmarks = []
        self.hands = []

    def add_hand(self, hand_type):
        new_hand = Hand(hand_type)
        self.hands.append(new_hand)

    def add_landmark(self, mark):
        self.landmarks.append(mark)

    def get_left_hand_landmarks(self):
        for hand in self.hands:
            if hand.type == "left":
                return hand.landmarks
        return []

    def get_right_hand_landmarks(self):
        for hand in self.hands:
            if hand.type == "right":
                return hand.landmarks
        return []

# 示例使用
person = Person("Alice", 30)

# 添加关键点
mark1 = Mark(10, 20)
mark2 = Mark(30, 40)
mark3 = Mark(50, 60)

person.add_landmark(mark1)
person.add_landmark(mark2)
person.add_landmark(mark3)

# 添加左右手
person.add_hand("left")
person.add_hand("right")

# 添加左手关键点
left_hand_mark1 = Mark(70, 80)
left_hand_mark2 = Mark(90, 100)
person.hands[0].add_landmark(left_hand_mark1)
person.hands[0].add_landmark(left_hand_mark2)

# 添加右手关键点
right_hand_mark1 = Mark(110, 120)
right_hand_mark2 = Mark(130, 140)
person.hands[1].add_landmark(right_hand_mark1)
person.hands[1].add_landmark(right_hand_mark2)

# 添加左手指头关键点
left_hand_big_mark1 = Mark(70,75)
left_hand_big_mark2 = Mark(75,80)
person.hands[0].add_landmark(left_hand_big_mark1)
person.hands[0].add_landmark(left_hand_big_mark2)

# 添加右手指头关键点
right_hand_big_mark1 = Mark(110,115)
right_hand_big_mark2 = Mark(115,120)
person.hands[1].add_landmark(right_hand_big_mark1)
person.hands[1].add_landmark(right_hand_big_mark2)


# 遍历关键点
for mark in person.landmarks:
    print(f"Landmark: ({mark.x}, {mark.y})")

# 获取并打印左手关键点
left_hand_landmarks = person.get_left_hand_landmarks()
print("Left Hand Landmarks:")
for mark in left_hand_landmarks:
    print(f"  Left Hand Landmark: ({mark.x}, {mark.y})")

# 获取并打印右手关键点
right_hand_landmarks = person.get_right_hand_landmarks()
print("Right Hand Landmarks:")
for mark in right_hand_landmarks:
    print(f"  Right Hand Landmark: ({mark.x}, {mark.y})")
"""

"""
class Mark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Hand:
    def __init__(self, type):
        # 存放关键点
        self.landmarks = []
        self.type = type

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        # 存放关键点
        self.landmarks = []
        self.hands = []

person = Person("松韵", 18)
left_mark1 = Mark(20, 100)
left_mark2 = Mark(50, 200)
left_mark3 = Mark(70, 300)
left_hand = Hand("left")
left_hand.landmarks.append(left_mark1)
left_hand.landmarks.append(left_mark2)
left_hand.landmarks.append(left_mark3)
person.hands.append(left_hand)

right_mark1 = Mark(100, 100)
right_mark2 = Mark(150, 200)
right_mark3 = Mark(170, 300)
right_hand = Hand("right")
right_hand.landmarks.append(right_mark1)
right_hand.landmarks.append(right_mark2)
right_hand.landmarks.append(right_mark3)
person.hands.append(right_hand)

for hand in person.hands:
    for mark in hand.landmarks:
        print(hand.type, mark.x, mark.y)
"""


"""
学生信息(做)
1) 封装Student类，包含属性name、age、tel、score、sex，包含方法getScore(打印name、
score)、 getStudent（打印个人的全部信息）。
2) 使用list列对象，存储5个学生对象，迭代所有学生信息。
3）打印不及格学生信息以及统计不及格学生数量
"""
# class Student():
#     def __init__(self, name, age, tel, score, sex):
#         self.name = name
#         self.age = age
#         self.tel = tel
#         self.score = score
#         self.sex = sex

#     def getStudent(self):
#         print(f"姓名：{self.name}, 年龄：{self.age}, 电话：{self.tel}, 分数：{self.score}, 性别：{self.sex}")

# names = ["alice", "bob", "charlie", "david", "eve"]
# ages = [18, 20, 19, 21, 17]
# tel = [13800138000, 13900139000, 14700147000, 15600156000, 15700157000]
# score = [85, 92, 78, 65, 55]
# sex = ["男", "女", "男", "男", "女"]

# student = []
# count = 0

# for i in range(len(names)):
#     if len(ages) > i and len(tel) > i and len(score) > i:
#         student.append(Student(names[i], ages[i], tel[i], score[i], sex[i]))
#         print(f"姓名：{names[i]}, 年龄：{ages[i]}, 电话：{tel[i]}, 分数：{score[i]}, 性别：{sex[i]}")
#         if score[i] < 60:
#             print(f"{names[i]}不及格")
#             count += 1
#     else:
#         print("Error: Input lists have different lengths")

# print(f"不及格学生数量：{count}")


# 给定列表 nums = [10, 20, 30, 50, 20], 定义一个函数找出给定元素的所有位置
# def find_all_positions(nums, target):
#     positions = []
#     for i in range(len(nums)):
#         if nums[i] == target:
#             positions.append(i)
#     return positions

# nums = [10, 20, 30, 50, 20]
# target = 20
# positions = find_all_positions(nums, target)
# print(f"找到目标元素{target}，位于{positions}")

# 使用for循环实现99乘法表
# for i in range(1, 10):
#     for j in range(1, i + 1):
#         print(f"{j}*{i}={i*j}", end="\t")
#     print()

# 定义一个函数, 求出1 + 2！+ 3！+ 4！+...+20！的结果
# def factorial_and_sum(n):
#     total = 0
#     for i in range(1, n + 1):
#         factorial = 1
#         for j in range(1, i + 1):
#             factorial *= j
#         total += factorial
#     return total

# # 20
# n = 20
# result = factorial_and_sum(n)
# print(f"1 + 2! + 3! + ... + {n}! 的和是：{result}")



## 求前k个高频元素
nums = [2,3,1,2,3,3,4,5, 5, 6, 7, 8, 9]
# from collections import Counter

# # 统计每个数字出现的次数
# counter = Counter(nums)

# # 获取前k个高频元素
# k = 3
# most_common_elements = counter.most_common(k)
# print(most_common_elements)  # 输出：[(3, 4), (2, 3), (5, 2)]

# # 打印每个数字出现的次数
# for num, count in counter.items():
#     print(f"{num}: {count}")
"""
dic = {}
for num in nums:
    dic[num] = dic.get(num, 0) + 1
print(dic)
"""


## python 文件操作

# file = open('t1', 'w', encoding='utf-8')
"""
with open自动释放文件
with open('t1', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(lines)
"""



"""
判断是文件还是目录
os.path.isfile / os.path.isdir(r'')
路径拼接
base_path = r''
os.path.join(base_path, 'added_path')

读取当前目录下的文件名
os.listdir()
./folder - 同级目录
.. - 上级目录

创建目录
os.mkdir('') - 创建一层目录
os.makedirs('', exist_ok = True) - 一次创建连续的目录

json格式-不局限于python，java，c++等
语法
key:value
大括号表示对象
元素之间使用逗号隔开
中括号表示数组

关键点描述
{
    "name": "str",
    "age": int,
    "landmarks": [
    {"x":int, "y":int},
    {...}
    ]
}


json文件操作
将python对象写入到json文件中
python对象
列表、元组、字典、自定义的对象

将python对象转换为json字符串
import json
json.dumps(dic)

将json转换为python格式
json.loads(json_file)

---------------------------------

YAML文件操作
yaml文件的语法和json类似，但更简洁
使用第三方库PyYAML进行解析和生成yaml文件
yml 可注释，作配置文件
yaml.safe_dump(data, file)
yaml.safe_load(file)

-----------------------------------

自定义线程类
继承Thread类
重写run方法
创建实例并调用start()方法启动线程
线程池：ThreadPoolExecutor(max_workers=5) - 创建一个包含5个线程的线程池
线程同步：Lock、Condition、Event等

from threading import Thread
class MyThread(Thread):
    def __init__(self, name, delay):
       super().__init__()   #对父类不造成影响

       def run(self):
           threeading.current_thread().name = name
           print(f"Thread {name} starting")
           time.sleep(delay)
           print(f"Thread {name} finishing")

# 创建线程实例并启动
thread1 = MyThread("Thread-1", 2)
thread1.start()

# 多个线程共享数据，如果存在写操作就会出现线程安全问题，需要加锁保护数据
import threading
lock = threading.Lock()
def critical_section():
    global shared_data
    lock.acquire()
    try:
        # 临界区代码
        print("Critical section accessed by", threading.current_thread().name)
        shared_data += 1
        print("New shared data:", shared_data)
        print("Thread", threading.current_thread().name, "finished")
        lock.release()
    except Exception as e:
        lock.release()
        print("Error:", e)

shared_data = 0
thread1 = Thread(target=critical_section)
thread2 = Thread(target=critical_section)
thread1.start()
thread2.start()

# 使用线程池进行多任务处理
from concurrent.futures import ThreadPoolExecutor
def task(x):
    return x * x

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(task, [1, 2, 3, 4, 5]))
print(results)

# 使用条件变量实现线程间通信
import threading
condition = threading.Condition()
def producer():
    with condition:
        print("Producer is producing")
        condition.notify()

def consumer():
    with condition:
        condition.wait()
        print("Consumer is consuming")

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

# 使用事件实现线程间通信
import threading
event = threading.Event()
def producer():
    with event:
        print("Producer is producing")
        event.set()

def consumer():
    with event:
        event.wait()
        print("Consumer is consuming")

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

# 使用信号量实现线程间通信
import threading
semaphore = threading.Semaphore(3)
def task():
    with semaphore:
        print(f"Thread {threading.current_thread().name} is running")
        time.sleep(1)

threads = []
for i in range(5):
    thread = threading.Thread(target=task)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# 使用队列实现线程间通信
import queue
q = queue.Queue()
def producer():
    for i in range(10):
        q.put(i)
        print(f"Producer put {i}")

def consumer():
    while not q.empty():
        item = q.get()
        print(f"Consumer got {item}")
        time.sleep(1)

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)












-----------------------------------

多进程编程
使用multiprocessing模块
Process类创建进程对象
target参数指定要执行的目标函数
args参数传递给目标函数的参数元组
start()方法启动进程
join()方法等待进程结束
进程间通信：Queue、Pipe、SharedMemory等

-----------------------------------

异步编程
使用asyncio库
定义协程函数（async def）
使用await关键字调用其他协程或阻塞操作
事件循环：EventLoop.run_until_complete(task)
并发执行多个任务：asyncio.gather(*tasks)

-----------------------------------

网络编程
socket模块进行TCP/IP通信
创建套接字对象：socket.socket(socket.AF_INET, socket.SOCK_STREAM)
绑定地址和端口：sock.bind(('localhost', 12345))
监听连接请求：sock.listen(5)
接受客户端连接：conn, addr = sock.accept()
发送数据：conn.sendall(b'Hello, client!')
接收数据：data = conn.recv(1024)
关闭连接：conn.close()

-----------------------------------

文件操作
打开文件：open(file_path, mode='r', encoding='utf-8')

"""

# # 导入MNIST数据集，分为test和train两类，每类包含多个文件夹及文件
# import os
# base_path = r'C:\Users\ktgr3\Desktop\processing\training'
# project_mnist_path = os.path.join(base_path, r'python_training_files\MNIST\raw')

# # 检查是否已经存在数据集目录
# if not os.path.isdir(project_mnist_path):
#     print("MNIST raw data directory does not exist.")
# else:
#     list_project_mnist_path = os.listdir(project_mnist_path)
#     print(f"We have these folders right now: {list_project_mnist_path}\n")

#     # 拾取train和test文件夹
#     train_folder_path = os.path.join(project_mnist_path, 'train')
#     test_folder_path = os.path.join(project_mnist_path, 'test')

#     print(train_folder_path)
#     print(test_folder_path)

#     train_files = []
#     test_files = []

#     if not os.path.isdir(train_folder_path):
#         print(f"Train folder does not exist for {train_foler_path}. Skipping...")
#     else:
#         # 处理train目录
#         for train_sub_folder in os.listdir(train_folder_path):
#             train_file_path = os.path.join(train_folder_path, train_sub_folder)
#             print(f"Processing {train_sub_folder} folder in {train_folder_path}: {train_file_path}")
#             for train_file in os.listdir(train_file_path):
#                 print(f"Processing file: {train_file}")
#                 with open(os.path.join(train_file_path, train_file), 'rb') as f:
#                     train_data = f.read()
#                     print(f"Train data read from {train_file}.")
#                     train_files.append((train_file_path, train_file))
#                     print(f"Train data appended to list.")

#     if not os.path.isdir(test_folder_path):
#         print(f"Test folder does not exist for {test_folder_path}. Skipping...")
#     else:
#         # 处理test目录
#         for test_sub_folder in os.listdir(test_folder_path):
#             test_file_path = os.path.join(test_folder_path, test_sub_folder)
#             print(f"Processing {test_sub_folder} folder in {test_folder_path}: {test_file_path}")
#             for test_file in os.listdir(test_file_path):
#                 print(f"Processing file: {test_file}")
#                 with open(os.path.join(test_file_path, test_file), 'rb') as f:
#                     test_data = f.read()
#                     print(f"Test data read from {test_file}.")
#                     test_files.append((test_file_path, test_file))
#                     print(f"Test data appended to list.")

# # 打印处理后的train和test文件列表
# print(f"Train files:{train_files}")
# print(f"Test files:{test_files}")

# print('Done.')


## 导入MNIST数据集，分为test和train两类，每类包含多个文件夹及文件
# import os
# base_path = r'C:\Users\ktgr3\Desktop\processing\training'
# project_mnist_path = os.path.join(base_path, r'python_training_files\MNIST\raw')


# def load_data(base_path, project_path):
#     train_folder_path = os.path.join(base_path, project_path, r'train')
#     test_folder_path = os.path.join(base_path, project_path, r'test')
#     #检查路径
#     print(f"Processing train folder: {train_folder_path}")
#     print(f"Processing test folder: {test_folder_path}")

#     train_files = []
#     test_files = []

#     if not os.path.isdir(train_folder_path):
#         print(f"Train folder does not exist for {train_folder_path}. Skipping...")
#     else:
#         # 处理train目录
#         for train_sub_folder in os.listdir(train_folder_path):
#             train_file_path = os.path.join(train_folder_path, train_sub_folder)
#             # print(f"Processing {train_sub_folder} folder in {train_folder_path}: {train_file_path}")
#             for train_file in os.listdir(train_file_path):
#                 # print(f"Processing file: {train_file}")
#                 with open(os.path.join(train_file_path, train_file), 'rb') as f:
#                     train_data = f.read()
#                     # print(f"Train data read from {train_file}.")
#                     train_files.append((train_file_path, train_file))
#                     # print(f"Train data appended to list.")

#     if not os.path.isdir(test_folder_path):
#         print(f"Test folder does not exist for {test_folder_path}. Skipping...")
#     else:
#         # 处理test目录
#         for test_sub_folder in os.listdir(test_folder_path):
#             test_file_path = os.path.join(test_folder_path, test_sub_folder)
#             # print(f"Processing {test_sub_folder} folder in {test_folder_path}: {test_file_path}")
#             for test_file in os.listdir(test_file_path):
#                 # print(f"Processing file: {test_file}")
#                 with open(os.path.join(test_file_path, test_file), 'rb') as f:
#                     test_data = f.read()
#                     # print(f"Test data read from {test_file}.")
#                     test_files.append((test_file_path, test_file))
#                     # print(f"Test data appended to list.")

#     # 打印处理后的train和test文件列表
#     print(f"Train files: {train_files}")
#     print(f"Test files: {test_files}")

#     return train_files, test_files

# train_files, test_files = load_data(base_path, project_mnist_path)
# print('Done.')



# # 读取label_train.txt，用\t分割
# import os
# base_path = r'C:\Users\ktgr3\Desktop\processing\training'
# train_file_path = os.path.join(base_path, r'python_training_files\label_train.txt')

# with open(train_file_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# # 将每一行按\t分割成一个列表
# for line in lines:
#     print(line.strip().split('\t'))


# import numpy as np

# a = np.array([
# [1, 4, 2, 5],
# [5, 6, 7, 8],
# [9, 10, 12, 13]
# ])
# # (3, 4)
# c = np.array([
# [8, 7, 255, 6],
# [5, 255, 255, 255],
# [3, 5, 255, 255]
# ])
# """
# 最后得到的数组：
# [[ 1 4 255 5]
# [ 5 255 255 255]
# [ 9 10 255 255]]
# """

# mask_np = np.where( 255==c )
# print(f"c数组中值为255的位置:\n{mask_np}")

# a[mask_np] = c[mask_np]
# print(f"掩码处理后的数组a:\n{a}")


# # arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) 转为一维数组
# arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# # arr_1d = np.reshape(arr_3d.size) # not working
# arr_1d = np.ndarray.flatten(arr_3d)
# print(arr_1d)


#使用代码完成下面的二维数组，边界值为1，其余值为0
"""
[[1. 1. 1. 1. 1.]
[1. 0. 0. 0. 1.]
[1. 0. 0. 0. 1.]
[1. 0. 0. 0. 1.]
[1. 1. 1. 1. 1.]]
"""
# dim = 6
# matrix = np.zeros((dim, dim), dtype=float)
# for i in range(matrix.shape[0]):
#     matrix[i, 0] = 1
#     matrix[i, -1] = 1
# for j in range(matrix.shape[1]):
#     matrix[0, j] = 1
#     matrix[-1, j] = 1
# print(matrix)


# 计算两维度的分数
# item = np.array([
# [3,5,8],
# [4,6,5],
# [3,8,3],
# [2,6,9]
# ])

# sum1 = item.sum(axis=0)
# print(f"0轴总分：{sum1}")
# sum2 = item.sum(axis=1)
# print(f"1轴总分：{sum2}")

## 求 target_vector 和 vector_sy、vector_qq、vector_lm、vector_mgt 下列最相近的两个向量
# 求模数
# 目标向量
# target_vector = np.array([1, 2])
# # 示例向量列表
# names = ['sy', 'qq', 'lm', 'mgt']
# vector_sy = np.array([4, 6])
# vector_qq = np.array([1, 2])
# vector_lm = np.array([10, 11])
# vector_mgt = np.array([1, 3])

# norm_sy = np.linalg.norm(vector_sy)
# norm_qq = np.linalg.norm(vector_qq)
# norm_lm = np.linalg.norm(vector_lm)
# norm_mgt = np.linalg.norm(vector_mgt)

# print(f"sy向量的模数：{norm_sy}")
# print(f"qq向量的模数：{norm_qq}")
# print(f"lm向量的模数：{norm_lm}")
# print(f"mgt向量的模数：{norm_mgt}")

# # 计算目标向量与每个示例向量之间的距离
# distances = np.array([
#     np.linalg.norm(target_vector - vector_sy),
#     np.linalg.norm(target_vector - vector_qq),
#     np.linalg.norm(target_vector - vector_lm),
#     np.linalg.norm(target_vector - vector_mgt)
# ])

# # 找到最接近目标向量的两个向量
# closest_indices = distances.argsort()[:2]
# closest_vectors = [names[i] for i in closest_indices]

# print(f"最接近目标向量 {target_vector} 的两个向量是：{closest_vectors}")


# from PIL import Image
# # 打开图像并获取尺寸
# img = Image.open('./image_01.jpg')
# width, height = img.size

# # 计算子图块的宽度和高度
# block_width = width // 3
# block_height = height // 3

# # 创建一个新图像以存储合并后的结果
# merged_img = Image.new('RGB', (width * 2, height))

# # 逐行逐列处理图像块
# for i in range(3):
#     for j in range(3):
#         # 计算子图块的起始和结束坐标
#         start_x = j * block_width
#         end_x = (j + 1) * block_width
#         start_y = i * block_height
#         end_y = (i + 1) * block_height

#         # 当前裁剪的子图块
#         block_img = img.crop((start_x, start_y, end_x, end_y))

#         # 将子图块复制到合并图像的相应位置
#         merged_img.paste(block_img, (start_x, start_y))
#         merged_img.paste(block_img, (width + start_x, start_y))
# # 显示
# merged_img.show()



# import numpy as np
# data = np.array(
# [
# [[0.50788623, 0.27810092, 0.33357751, 0.44273496, 0.79644001],
# [0.75848508, 0.26722897, 0.39062434, 0.43147366, 0.41888956]],
# [[0.79044801, 0.27298049, 0.26465866, 0.41627939, 0.41240781],
# [0.50502996, 0.10372226, 0.23264812, 0.40855196, 0.45969932]],
# [[0.71268181, 0.45235896, 0.57299559, 0.49017589, 0.69338482],
# [0.16953653, 0.17006457, 0.12208736, 0.42710763, 0.41410585]],
# [[0.78925542, 0.25889081, 0.36485464, 0.59822239, 0.40035307],
# [0.13446042, 0.13492031, 0.23874074, 0.45157296, 0.42610657]
# ]
# ])
# # 需求1
# """
# [[0.75848508 0.26722897 0.39062434 0.43147366 0.41888956]
# [0.79044801 0.27298049 0.26465866 0.41627939 0.41240781]
# [0.71268181 0.45235896 0.57299559 0.49017589 0.69338482]
# [0.78925542 0.25889081 0.36485464 0.59822239 0.40035307]]
# """
# idx = data[:, :, 0] > 0.7
# print(idx)
# filter_boxes = data[idx]
# # print(filter_boxes)
# # 需求2
# """
# [[0.26722897 0.39062434 0.43147366 0.41888956]
# [0.27298049 0.26465866 0.41627939 0.41240781]
# [0.45235896 0.57299559 0.49017589 0.69338482]
# [0.25889081 0.36485464 0.59822239 0.40035307]]
# """
# boxes = filter_boxes[:, 1:]
# # print(boxes)
# """
# 需求3:再对每个元素*100得到以下的内容
# """
# boxes = (boxes * 100).astype(np.int32)

# print(boxes)
# """
# 需求4:
# 求box:[46, 57, 49, 68]和下列矩形框相交面积，并找出相交面积最大的矩形框。
# [x1, y1, x2, y2]
# boxes:
# [[26 39 43 41]
# [27 26 41 41]
# [45 57 49 69]
# [25 36 59 40]]
# """
# def inter_areas(box, boxes):
# box_area = (box[2] - box[0]) * (box[3] - box[1])
# box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
# l_x = np.maximum(box[0], boxes[:, 0])
# l_y = np.maximum(box[1], boxes[:, 1])
# r_x = np.minimum(box[2], boxes[:, 2])
# r_y = np.minimum(box[3], boxes[:, 3])
# w = np.maximum(0, r_x-l_x)
# h = np.maximum(0, r_y-l_y)
# inter_areas = w * h
# # iou交并比
# iou_val = inter_areas/(box_areas+box_area-inter_areas)
# return iou_val
# if __name__ == '__main__':
# box = np.array([46, 57, 49, 68])
# # boxes = np.array([[26, 39, 43, 41],
# # [27 ,26 ,41 ,41],
# # [45 ,57 ,49 ,69],
# # [25 ,36 ,59 ,40]])
# areas = inter_areas(box, boxes)
# print(areas)