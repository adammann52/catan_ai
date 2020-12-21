from handler import *
from cnnPlaidML import *
from replayBuffer import ReplayBuffer
from visualizeLearning import *
import csv

def softUpdate(principal,target, alpha):

    weights = principal.model.get_weights()
    target_weights = target.model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = weights[i] * alpha + target_weights[i]*(1-alpha)
    target.model.set_weights(target_weights)

if __name__ == '__main__':
    rewards = {'E':0,'S':1.2,'C':1.2,'R':.3,'T':0,'D':.4,'MR':.6,'RB':.6,'YP':.6,'M':.6}
    episodes = 301
    lr = .001
    gamma = .93
    alpha = .001
    epsilon = 1
    tau = 2500
    wait = 3000
    batch_size = 32
    maxLengthGame = 450
    Qprincipal = QNetwork(lr)
    Qtarget = QNetwork(lr)

    Qtarget.model.set_weights(Qprincipal.model.get_weights())

    rBuffer = ReplayBuffer(5000)
    count = 0
    results = []

    print(' Episode   |   Score   |    Loss  |  Rounds')
    for ep in range(episodes):
        loss = 0
        bots = [Robot() for i in range(3)]
        game = Game(player_names = [bot.name for bot in bots])
        epsilon = max(epsilon*.995,.1)
        winnings = 20000
        for i in range(maxLengthGame):
            maxLengthGame -= 1
            winnings -= 20
            count += 1
            if game.dieRoll == 7:
                dropCards(game)
            state = getState(game)
            actions = processActions(game,state)
            action, spec, state, win = selectAction(game,actions,Qprincipal,epsilon)

            #get replay tuple
            n_state = getState(game)
            n_actions = []
            acts = []
            for el in processActions(game,state):
                n_actions.append(el[2])
                acts.append(el[0])
            if action == 'E':
                game.playerUpdate()

            if action == 'S' or action == 'C':
                r = game.board.vertices[spec[0]][spec[1]].score
            else:
                r = rewards[action]
            rBuffer.append((state,r + win*winnings,n_actions,acts))
            rBuffer.pop()

            if count > wait:
                states = []
                targets = []
                for s_0, r, s_1,a in rBuffer.sample(batch_size):

                    states.append(s_0)
                    q_t = np.max(Qtarget.evaluate(s_1))
                    targets.append(r+gamma*q_t)

                loss += Qprincipal.train(states,targets).history['loss'][0]


                softUpdate(Qprincipal,Qtarget,alpha)

        bots = [Robot() for i in range(3)]
        game = Game(player_names = [bot.name for bot in bots])

        for i in range(maxLengthGame):

            if game.dieRoll == 7:
                dropCards(game)
            state = getState(game)
            actions = processActions(game,state)
            tup = selectAction(game,actions,Qprincipal,0)
            if tup[0] == 'E':
                game.playerUpdate()
            if tup[3]:
                break
        for player in game.players:
            player.updateScore()
        scores = [player.points for player in game.players]

        scores += [i]
        scores += [game.round]

        results.append(scores)


        print('    '+str(ep)+'    |'+ '  '+str(sum(scores[:3])/3)[:3]+','+str(max(scores[:3]))+'  |'+'  '+str(loss/500)+'  | '+str(scores[4]))
        if ep%20==0:
            Qprincipal.model.save('models/cnn_dl'+str(ep))
            #Visualize(ai=True,game=None,network = Qprincipal)

    with open('results_dl.csv', 'w') as f:

        write = csv.writer(f)

        write.writerows(results)
