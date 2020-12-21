import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import copy
import keras
class QNetwork():

    def __init__(self,lr,fle=None):


        if fle!=None:
            self.model = keras.models.load_model(fle)

        else:
            optim = keras.optimizers.Adam(
            lr=lr)

            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(3,(3,3),activation='relu',input_shape = (27,27,3)))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(32,activation='relu'))
            model.add(keras.layers.Dense(1))
            model.compile(loss='mse', optimizer = optim)
            self.model = model

    def padding(self,states):

        for state in states:
            for channel in state:
                if len(channel)==11:
                    for i in range(5):
                        channel.insert(0,[0]*21)
                    for i in range(5):
                        channel.append([0]*21)

    def reshaping(self,states):

        new_states = []
        for state in states:

            nb = []
            #first row
            fr = [0]*6
            sr = [0]*5
            for i in range(6,21):
                if (i-6)%4==0:
                    fr.append(state[0][0][i])
                    state[0][0][i] = 0
                else:
                    fr.append(0)
            for i in range(5,21):
                if (i-5)%2==0:
                    sr.append(state[0][0][i])
                    state[0][0][i] = 0
                else:
                    sr.append(0)
            nb.append(fr)
            nb.append(sr)
            nb.append(state[0][0].copy())
            rule = [(5,2),(6,4),(3,2),(4,4),(1,2),(2,4),(1,2),(0,4),(3,2),(2,4)]
            for i in range(1,len(state[0])):
                if i % 2 == 1:
                    nb.append(state[0][i].copy())
                else:

                    v_bgn, v_skip = rule.pop()
                    r_bgn, r_skip = rule.pop()

                    vs = [0]*v_bgn
                    rs = [0]*r_bgn
                    for j in range(v_bgn, 21):
                        if (j-v_bgn)%4 == 0:
                            vs.append(state[0][i][j])
                            state[0][i][j] = 0
                        else:
                            vs.append(0)

                    for j in range(r_bgn, 21):
                        if (j-r_bgn)%2 == 0:
                            rs.append(state[0][i][j])
                            state[0][i][j] = 0
                        else:
                            rs.append(0)

                    nb.append(state[0][i].copy())
                    nb.append(rs)
                    nb.append(vs)

            nb.insert(0,[0]*21)
            nb.insert(0,[0]*21)
            nb.append([0]*21)
            nb.append([0]*21)


            for row in nb:
                for i in range(3):
                    row.insert(0,0)
                    row.append(0)

            road_board = copy.deepcopy(nb)

            for i in range(len(road_board)):

                if i % 2 ==0:
                    nb[i] = [0]*27
                else:
                    road_board[i] = [0]*27

            card_layer = copy.deepcopy(state[1])

            for i in range(8):
                card_layer.insert(0,[0]*21)
                card_layer.append([0]*21)

            for row in card_layer:
                for i in range(3):
                    row.insert(0,0)
                    row.append(0)

            for i in range(27):
                for j in range(27):

                    card_layer[i][j]/=19

            new_states.append([nb,road_board,card_layer])

        return new_states


    def train(self,states,targets):
        #self.padding(states)
        states = self.reshaping(states)

        states = np.array([np.array([np.array([np.array(row) for row in channel]) for channel in state]) for state in states])
        states = np.transpose(states,(0,2,3,1))
        #states = (states-np.mean(states))/np.std(states)
        #states = (states-np.min(states))/np.ptp(states)
        return self.model.fit(states,np.array(targets),verbose = 0)

    def evaluate(self,next_states):
        #self.padding(next_states)
        next_states = self.reshaping(next_states)
        states = np.array([np.array([np.array([np.array(row) for row in channel]) for channel in state]) for state in next_states])
        #print(states[0][0])
        states = np.transpose(states,(0,2,3,1))
        #states = (states-np.mean(states))/np.std(states)
        #states = (states-np.min(states))/np.ptp(states)
        return self.model.predict(states)


