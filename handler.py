from game import Game
from player import Player
from robot import Robot
import random
import copy
import functools
import numpy as np
#from visualize import Visualize


def getState(game):
   # homes,cities,roads = game.availableMoves()
    board =  [[0]*21 for i in range(11)]

    for i in range(len(game.board.vertices)):
        for q in range(len(game.board.vertices[0])):
            if game.board.vertices[i][q]:
                if game.board.vertices[i][q].owner == game.current_player:
                    board[i*2][q*2] = 1 + int(game.board.vertices[i][q].city == True)
                elif game.board.vertices[i][q].owner:
                    board[i*2][q*2] = -1 - int(game.board.vertices[i][q].city == True)


            if ((i,q),(i,q+1)) in game.board.edges:
                if game.board.edges[((i,q),(i,q+1))].owner == game.current_player:
                    board[i*2][q*2+1] = 1
                elif game.board.edges[((i,q),(i,q+1))].owner:
                    board[i*2][q*2+1] = -1

            if ((i,q),(i+1,q)) in game.board.edges:
                if game.board.edges[((i,q),(i+1,q))].owner == game.current_player:
                    board[i*2+1][q*2] = .75
                elif game.board.edges[((i,q),(i+1,q))].owner:
                    board[i*2+1][q*2] = -.75

    for spot in game.board.spots.values():
        if spot.blocked:
            xs = []
            ys = []

            for x,y in spot.vertices:
                xs.append(x*2)
                ys.append(y*2)
            board[sum(xs)//len(xs)][sum(ys)//len(ys)] = .5

    state = [board]

    other_features = []
    for key in game.current_player.hand:
        other_features.append([game.current_player.hand[key]]*21)

    #development cards
    other_features.append([game.current_player.knight]*21)
    other_features.append([game.current_player.monopoly]*21)
    other_features.append([game.current_player.road_builder]*21)
    other_features.append([game.current_player.point_cards]*21)
    other_features.append([game.current_player.year_of_plenty]*21)

    #other auxilary features
    other_features.append([game.current_player.played_knights]*21)
    state.append(other_features)
    return state

def switch(game,state,r1,r2,v1,v2):
    hand = game.current_player.hand
    key_index = {key:i for key,i in zip(hand.keys(),list(range(5)))}
    c_state = copy.deepcopy(state)
    c_state[1][key_index[r1]] = [hand[r1] - v1]*21
    c_state[1][key_index[r2]] = [hand[r2] + 1]*21
    return c_state


def moveRobber(game,state,knight = False):

    spots = []
    new_states = []

    t_state = copy.deepcopy(state)
    for i in range(len(state)):
        for j in range(len(state[0])):

            if t_state[i][j] == .5:
                t_state[i][j] = 0

    for spot in game.board.spots.values():
        if not spot.blocked:
            xs = []
            ys = []
            for x,y in spot.vertices:
                xs.append(x*2)
                ys.append(y*2)
            spots.append(spot)
            tmp = copy.deepcopy(t_state)
            tmp[0][int(np.mean(xs))][int(np.mean(ys))] = .5
            new_states.append(tmp)

    return [['MR',(spot,knight),b] for spot,b in zip(spots,new_states)]


def yearOfPlenty(game,state):

    permutations = [[2,0,0,0,0],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2],
                    [1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],
                    [0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1],
                    [0,0,1,0,1],[0,0,1,0,1],
                    [0,0,0,1,1]]

    actions = []
    for permutation in permutations:

        c_state = copy.deepcopy(state)
        for val,i in zip(permutation,range(5)):
            c_state[1][i] = [state[1][i][0]+val]*21

        actions.append(('YP',permutation,c_state))

    return actions


def monopoly(game,state):

    actions = []

    for resource,i in zip(game.current_player.hand.keys(),range(5)):

        c_state = copy.deepcopy(state)
        count = 0
        for player in game.players:

            count += player.hand[resource]

        c_state[1][i] = [count]*21

        actions.append(('M',resource,c_state))

    return actions


def processActions(game,state):

    game.availableMoves()
    if game.sevenBlock:
        game.sevenBlock = False
        return moveRobber(game,state)

    actions = [['E',[],copy.deepcopy(state)]]
    new_states = [copy.deepcopy(state)]
    moves = game.moves
    shtetles = moves['settlements']

    for i in range(len(shtetles)):
        for q in range(len(shtetles[0])):
            if shtetles[i][q]:
                c_state = copy.deepcopy(state)
                # board level
                c_state[0][i*2][q*2] = 1
                actions.append(['S',(i,q),c_state])
                #new_states.append(c_state)

    cities = [[]]
    if 'cities' in moves:
        cities = moves['cities']

    for i in range(len(cities)):
        for q in range(len(cities[0])):
            if cities[i][q]:
                c_state = copy.deepcopy(state)
                # board level
                c_state[0][i*2][q*2] = 2
                actions.append(['C',(i,q),c_state])


    roads = moves['roads']
    road_actions = []
    for v1,v2 in roads:

        c_state = copy.deepcopy(state)
        #downwards road
        if v2[0] - v1[0] == 1:
            #board level
            c_state[0][v1[0]*2+1][v1[1]*2] = .75

        #sideways road
        else:
            c_state[0][v1[0]*2][v1[1]*2+1] = .75

        road_actions.append(['R',(v1,v2),c_state])

        if game.road_building:
            return road_actions

        actions += road_actions

    #trades
    hand = game.current_player.hand
    for resource in hand.keys():

        if game.current_player.ports[resource] and hand[resource] >= 2:
            for el in hand.keys():
                if el != resource:
                    c_state = switch(game,state,resource,el,2,1)
                    actions.append(['T',(resource,el,2,1),c_state])

        elif hand[resource] >= 3 and  game.current_player.ports['3:1']:
            for el in hand:
                if el != resource:
                    c_state = switch(game,state,resource,el,3,1)
                    actions.append(['T',(resource,el,3,1),c_state])

        elif hand[resource] >= 4:
            for el in hand:
                if el!= resource:
                    c_state = switch(game,state,resource,el,4,1)
                    actions.append(['T',(resource,el,4,1),c_state])


    #dev cards, create all five possibilities
    if 'dev_card' in game.moves and game.moves['dev_card']:
        dev_states = []
        for i in range(5,10):
            c_state = copy.deepcopy(state)
            c_state[1][i] = [c_state[1][i][0]+1]*21
            actions.append(('D',(),c_state))

    if 'knight' in game.moves and game.moves.knight:
        for rob in moveRobber(game,state,True):
            actions.append(rob)

    if 'road_builder' in game.moves and game.moves['road_builder']:

        actions.append(('RB',(),copy.deepcopy(state)))

    if 'year_of_plenty' in game.moves and game.moves['year_of_plenty']:

        for action in yearOfPlenty(game,state):
            actions.append(action)

    if 'monopoly' in game.moves and game.moves['monopoly']:

        for action in monopoly(game,state):
            actions.append(action)

    return actions

def takeCard(game,spec):

    opponents = set()

    for a,b in spec.vertices:
        if game.board.vertices[a][b].owner:
            opponents.add(game.board.vertices[a][b].owner)

    if game.current_player in opponents:
        opponents.remove(game.current_player)

    if len(opponents) > 0:

        takeFrom = list(opponents)[random.randint(0,len(opponents)-1)]

        types = list(takeFrom.hand.keys())
        random.shuffle(types)

        for el in types:
            if takeFrom.hand[el] > 0:
                takeFrom.hand[el] -= 1
                game.current_player.hand[el] += 1
                break


#this is where the network will be
def selectAction(game,actions,Qprincipal,epsilon=0):
    won = False
    action = 'E'
    spec = []
    state = actions[0][2]
    #print(actions)
    if len(actions) > 1:
        if random.random() < epsilon:
            i = random.randint(int(game.round<2),len(actions)-1)
        else:
            states = [el[2] for el in actions]
            if game.round < 2:
                states.pop(0)
            Qvalues = Qprincipal.evaluate(states)
            i = np.argmax(Qvalues) + (game.round<2)
        action,spec,state = actions[i]
    if action == 'S':
        game.buySettlement(game.current_player.name,spec)
    elif action == 'C':
        game.buyCity(game.current_player.name,spec)
    elif action == 'R':
        if game.road_building:
            game.road_build_count -= 1
            if game.road_build_count == 0:
                game.road_bulding = False

        game.buyRoad(game.current_player.name,spec)
    elif action == 'T':
        game.current_player.hand[spec[0]] -= spec[2]
        game.current_player.hand[spec[1]] += spec[3]
    elif action == 'D':
        game.buyDev()
    elif action == 'MR':
        takeCard(game,spec[0])
        for el in game.board.spots.values():
            el.blocked = False

        spec[0].blocked = True
        if spec[1]:
            game.playedKnight()
            game.current_player.knight -= 1
    elif action == 'RB':
        game.road_building = True
        game.road_build_count = 2
        game.current_player.hand['brick'] += 2
        game.current_player.hand['wood'] += 2
        game.current_player.road_builder -= 1
    elif action == 'YP':
        for key,val in zip(game.current_player.hand,spec):
            game.current_player.hand[key] += val
        game.current_player.year_of_plenty -= 1
    elif action == 'M':
        count = 0
        for player in game.players:
            count += player.hand[spec]
            player.hand[spec] = 0

        game.current_player.hand[spec] = count
        game.current_player.monopoly -= 1

    else:
        won = game.current_player.points >= 10
        game.turn += 1
        game.round = game.turn//3
        if game.round > 1:
            game.rollDice()
    game.availableMoves()

    return action,spec,state,won

@functools.lru_cache(maxsize= 50)
def permutations(to_drop,r,hand):

    if to_drop == 0:
        return [hand]
    if r == len(hand):
        return []

    h = list(hand)

    ans = []
    for i in range(min(hand[r],to_drop)+1):

        nh= h.copy()
        nh[r] = nh[r] - i
        for el in permutations(to_drop-i,r+1,tuple(nh)):
            ans.append(el)

    return ans

def dropCards(game):

    for player in game.players:
        total = sum(player.hand.values())
        if total > 7:
            to_drop = total//2

            perms = permutations(to_drop,0,tuple(player.hand.values()))
            randHand = perms[random.randint(0,len(perms)-1)]
            for key,i in zip(player.hand,range(5)):
                player.hand[key] = randHand[i]



def threeBot(max_length,rewards,Qprincipal):

    bots = [Robot() for i in range(3)]
    game = Game(player_names = [bot.name for bot in bots])
    for i in range(max_length):
        if game.dieRoll == 7:
            dropCards(game)
        state = getState(game)
        actions = processActions(game,state)
        if selectAction(game,actions,Qprincipal,epsilon):
            print(i)
            break
    print('end')

