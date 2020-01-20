package com.fossgalaxy.games.fireworks.ai.rvdm;

import java.util.*;

import com.fossgalaxy.games.fireworks.state.*;
import com.fossgalaxy.games.fireworks.state.actions.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.ai.rule.logic.DeckUtils;
import com.fossgalaxy.games.fireworks.annotations.AgentBuilderStatic;
import com.fossgalaxy.games.fireworks.annotations.AgentConstructor;
import com.fossgalaxy.games.fireworks.utils.DebugUtils;

/**
 * Created by WebPigeon on 09/08/2016.
 */
public class MCTS implements Agent {
    public static final int DEFAULT_ITERATIONS = 50_000;
    public static final int DEFAULT_ROLLOUT_DEPTH = 18;
    public static final int DEFAULT_TREE_DEPTH_MUL = 1;
    public static final int NO_LIMIT = 100;
    protected static final boolean OLD_UCT_BEHAVIOUR = false;

    protected final int roundLength;
    protected final int rolloutDepth;
    protected final int treeDepthMul;
    protected final Random random;
    protected final Logger logger = LoggerFactory.getLogger(MCTS.class);

    private final boolean calcTree = false;

    protected MultiLayerNetwork model;
    protected NormalizerStandardize normalizer;

    /**
     * Create a default MCTS implementation.
     * <p>
     * This creates an MCTS agent that has a default roll-out length of 50_000 iterations, a depth of 18 and a tree
     * multiplier of 1.
     */
    public MCTS() {
        this(DEFAULT_ITERATIONS, DEFAULT_ROLLOUT_DEPTH, DEFAULT_TREE_DEPTH_MUL);
    }

    public MCTS(int roundLength) {
        this(roundLength, DEFAULT_ROLLOUT_DEPTH, DEFAULT_TREE_DEPTH_MUL);
    }

    @AgentConstructor("mcts")
    public MCTS(int roundLength, int rolloutDepth, int treeDepthMul) {
        this.roundLength = roundLength;
        this.rolloutDepth = rolloutDepth;
        this.treeDepthMul = treeDepthMul;
        this.random = new Random();
        System.out.println("Modified MCTS ctor");
    }

    @AgentBuilderStatic("mctsND")
    public static MCTS buildMCTSND() {
        return new MCTS(MCTS.DEFAULT_ITERATIONS, MCTS.NO_LIMIT, MCTS.NO_LIMIT);
    }

    @Override
    public Action doMove(int agentID, GameState state) {
        long finishTime = System.currentTimeMillis() + 1000;
        MCTSNode root = new MCTSNode(
                (agentID + state.getPlayerCount() - 1) % state.getPlayerCount(),
                null,
                Utils.generateAllActions(agentID, state.getPlayerCount())
        );

        Map<Integer, List<Card>> possibleCards = DeckUtils.bindCard(agentID, state.getHand(agentID), state.getDeck().toList());
        List<Integer> bindOrder = DeckUtils.bindOrder(possibleCards);


        if (logger.isTraceEnabled()) {
            logger.trace("Possible bindings: ");
            possibleCards.forEach((slot, cards) -> logger.trace("\t {} {}", slot, DebugUtils.getHistStr(DebugUtils.histogram(cards))));

            // Guaranteed cards
            logger.trace("Guaranteed Cards");

            possibleCards.entrySet().stream()
                    .filter(x -> x.getValue().size() == 1)
                    .forEach(this::printCard);

            logger.trace("We know the value of these");
            possibleCards.entrySet().stream()
                    .filter(x -> x.getValue().stream().allMatch(y -> y.value.equals(x.getValue().get(0).value)))
                    .forEach(this::printCard);

            DebugUtils.printTable(logger, state);
        }

//        for (int round = 0; round < roundLength; round++) {
        while(System.currentTimeMillis() < finishTime){
            //find a leaf node
            GameState currentState = state.getCopy();
            IterationObject iterationObject = new IterationObject(agentID);

            Map<Integer, Card> myHandCards = DeckUtils.bindCards(bindOrder, possibleCards);

            Deck deck = currentState.getDeck();
            Hand myHand = currentState.getHand(agentID);
            for (int slot = 0; slot < myHand.getSize(); slot++) {
                Card hand = myHandCards.get(slot);
                myHand.bindCard(slot, hand);
                deck.remove(hand);
            }
            deck.shuffle();

            MCTSNode current = select(root, currentState, iterationObject);

            int score = estimateScore(current, currentState);

            //int score = rollout(currentState, current);

            current.backup(score);
            if(calcTree){
                System.err.println(root.printD3());
            }
        }

        if (logger.isInfoEnabled()) {
            for (MCTSNode level1 : root.getChildren()) {
                logger.info("rollout {} moves: max: {}, min: {}, avg: {}, N: {} ", level1.getAction(), level1.rolloutMoves.getMax(), level1.rolloutMoves.getMin(), level1.rolloutMoves.getMean(), level1.rolloutMoves.getN());
                logger.info("rollout {} scores: max: {}, min: {}, avg: {}, N: {} ", level1.getAction(), level1.rolloutScores.getMax(), level1.rolloutScores.getMin(), level1.rolloutScores.getMean(), level1.rolloutScores.getN());
            }
        }

        if (logger.isTraceEnabled()) {
            logger.trace("next player's moves considerations: ");
            for (MCTSNode level1 : root.getChildren()) {
                logger.trace("{}'s children", level1.getAction());
                level1.printChildren();
            }
        }

        Action chosenOne = root.getBestNode().getAction();
        if (logger.isTraceEnabled()) {
            logger.trace("Move Chosen by {} was {}", agentID, chosenOne);
            root.printChildren();
        }
        return chosenOne;
    }

    private int estimateScore(MCTSNode node, GameState state)
    {
        int currentScore = state.getScore();
        int info_tokens = state.getInfomation();
        int lives = state.getLives();
        int deck = state.getDeck().getCardsLeft();
        int moves_left = state.getMovesLeft();

        HashMap<Integer, Integer> counts = GetCardCount(state);
        int fives = counts.get(5);
        int fours = counts.get(4);
        int threes = counts.get(3);
        int twos = counts.get(2);
        int num_players = state.getPlayerCount();

        Action action = node.getAction();
        int discard = action instanceof DiscardCard ? 1 : 0;
        int play = action instanceof PlayCard ? 1 : 0;
        int tellC = action instanceof TellColour ? 1 : 0;
        int tellV = action instanceof TellValue ? 1 : 0;

        double[][] raw_input = {{currentScore, info_tokens, lives, deck, moves_left, fives, fours, threes, twos,
                            num_players, discard, play, tellC, tellV}};
        final INDArray input = Nd4j.create(raw_input);

        this.normalizer.transform(input);
        INDArray out = model.output(input, false);
        double d_out = out.getDouble(0);

        return (int)Math.round(d_out);
    }

    private HashMap<Integer, Integer> GetCardCount(GameState state)
    {
        HashMap<Integer, Integer> result = new HashMap<>();
        result.put(2, 0);
        result.put(3, 0);
        result.put(4, 0);
        result.put(5, 0);

        for(CardColour colour : CardColour.values())
        {
            int val = state.getTableValue(colour);
            for(int i=val ; i>1 ; i--)
            {
                result.put(i, result.get(i) + 1);
            }
        }

        return result;
    }

    protected MCTSNode select(MCTSNode root, GameState state, IterationObject iterationObject) {
        MCTSNode current = root;
        int treeDepth = calculateTreeDepthLimit(state);
        boolean expandedNode = false;
        
        while (!state.isGameOver() && current.getDepth() < treeDepth && !expandedNode) {
            MCTSNode next;
            if (current.fullyExpanded(state)) {
                next = current.getUCTNode(state);
            } else {
                next = expand(current, state);
                expandedNode = true;
            }
            
            if (next == null) {
                //XXX if all follow on states explored so far are null, we are now a leaf node
            	//ok to early return here - we will have applied current last time round the loop!
                return current;
            }
            current = next;

            int agent = current.getAgent();
            int lives = state.getLives();
            int score = state.getScore();

            Action action = current.getAction();
            if (action != null) {
                action.apply(agent, state);
            }

            if (iterationObject.isMyGo(agent)) {
                if (state.getLives() < lives) {
                    iterationObject.incrementLivesLostMyGo();
                }
                if (state.getScore() > score) {
                    iterationObject.incrementPointsGainedMyGo();
                }
            }
        }
        return current;
    }

    protected int calculateTreeDepthLimit(GameState state){
        return (state.getPlayerCount() * treeDepthMul) + 1;
    }

    /**
     * Select a new action for the expansion node.
     *
     * @param state   the game state to travel from
     * @param agentID the AgentID to use for action selection
     * @param node    the Node to use for expansion
     * @return the next action to be added to the tree from this state.
     */
    protected Action selectActionForExpand(GameState state, MCTSNode node, int agentID) {
        Collection<Action> legalActions = node.getLegalMoves(state, agentID);
        if (legalActions.isEmpty()) {
            return null;
        }

        Iterator<Action> actionItr = legalActions.iterator();

        int selected = random.nextInt(legalActions.size());
        Action curr = actionItr.next();
        for (int i = 0; i < selected; i++) {
            curr = actionItr.next();
        }

        return curr;
    }

    protected MCTSNode expand(MCTSNode parent, GameState state) {
        int nextAgentID = (parent.getAgent() + 1) % state.getPlayerCount();
        Action action = selectActionForExpand(state, parent, nextAgentID);
        // It is possible it wasn't allowed
        if (action == null) {
            return parent;
        }
        if (parent.containsChild(action)) {
            // return the correct node instead
            return parent.getChild(action);
        }
        //XXX we may expand a node which we already visited? :S
        MCTSNode child = new MCTSNode(parent, nextAgentID, action, Utils.generateAllActions(nextAgentID, state.getPlayerCount()));
        parent.addChild(child);
        return child;
    }

    protected Action selectActionForRollout(GameState state, int playerID) {
        Collection<Action> legalActions = Utils.generateActions(playerID, state);

        List<Action> listAction = new ArrayList<>(legalActions);
        Collections.shuffle(listAction);

        return listAction.get(0);
    }

    protected int rollout(GameState state, MCTSNode current) {
        int playerID = (current.getAgent() + 1) % state.getPlayerCount();
        int moves = 0;

        while (!state.isGameOver() && moves < rolloutDepth) {
            Action action = selectActionForRollout(state, playerID);
            action.apply(playerID, state);
            playerID = (playerID + 1) % state.getPlayerCount();
            moves++;
        }

        current.backupRollout(moves, state.getScore());
        return state.getScore();
    }

    @Override
    public String toString() {
        return "MCTS";
    }

    private void printCard(Map.Entry<Integer, List<Card>> entry) {
        logger.trace("{} : {}", entry.getKey(), entry.getValue());
    }

}
