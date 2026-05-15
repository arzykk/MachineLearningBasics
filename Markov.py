import numpy as np
import time
import matplotlib.pyplot as plt

# state matrix
t1 = ['P', 'St', 'Sc']
# matrix of repetition
wyst = np.array([[1, 2, 1], [0, 1, 0], [0, 0, 1]])
# oponent probability distribution - could be any
prob_op = np.array([1, 0, 0])
print(wyst)
print(wyst[0])
print(wyst[0] / sum(wyst[0]))
print(np.random.choice(t1, p=wyst[0] / sum(wyst[0])))

kasa = []
stankasy = 0

n = 300
state = 'P'

for i in range(n):
    if state == 'P':
        # prediction of oponent's movement
        pred = np.random.choice(t1, p=wyst[0] / sum(wyst[0]))
        print(pred)
        op_akc = np.random.choice(t1, p=prob_op)
        print(op_akc)

    if state == 'St':
        pred = np.random.choice(t1, p=wyst[1] / sum(wyst[1]))
        print(pred)
        op_akc = np.random.choice(t1, p=prob_op)
        print(op_akc)

    if state == 'Sc':
        pred = np.random.choice(t1, p=wyst[2] / sum(wyst[2]))
        op_akc = np.random.choice(t1, p=prob_op)
        print(op_akc)
        print(pred)

    # map our prediction to our action
    if pred == 'St':
        our_move = 'P'
    if pred == 'P':
        our_move = 'Sc'
    if pred == 'Sc':
        our_move = 'St'
    # real movement of oponent
    # update score acording to op_akc and our_move
    if our_move == 'P' and op_akc == 'St':
        stankasy += 1
    if our_move == 'St' and op_akc == 'Sc':
        stankasy += 1
    if our_move == 'Sc' and op_akc == 'P':
        stankasy += 1
    if our_move == 'St' and op_akc == 'P':
        stankasy -= 1
    if our_move == 'P' and op_akc == 'Sc':
        stankasy -= 1
    if our_move == 'Sc' and op_akc == 'St':
        stankasy -= 1
        # update the matrix wyst based on real oponent movement - this is what we learn
    wyst[t1.index(our_move), t1.index(op_akc)] += 1
    # go to state op_akc

    kasa.append(stankasy)

    state = op_akc

    print("\n")
   # time.sleep(0.1)

print(wyst)
plt.plot(kasa)
plt.show()
print(len(kasa))