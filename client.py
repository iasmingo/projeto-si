import connection as cn


socket = cn.connect(111)

while True:
    state, reward = cn.get_state_reward(socket, "jump")
    print(state, reward)