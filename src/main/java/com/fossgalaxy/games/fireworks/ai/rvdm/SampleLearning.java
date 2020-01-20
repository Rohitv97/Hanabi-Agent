package com.fossgalaxy.games.fireworks.ai.rvdm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.iggi.Utils;
import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.HistoryEntry;
import com.fossgalaxy.games.fireworks.state.actions.Action;

/**
 * A sample agent for the learning track.
 * 
 * This agent demonstrates how to get the player IDs for the paired agents.
 * 
 * 
 * You can see more agents online at:
 * https://git.fossgalaxy.com/iggi/hanabi/tree/master/src/main/java/com/fossgalaxy/games/fireworks/ai
 */
public class SampleLearning implements Agent {

	private Map<String, Map<Action, Integer>> actionHistory;

	private Random random;
	private int myID;
	private String[] currentPlayers;
	
	public SampleLearning() {
		this.actionHistory = new HashMap<>();
	}
	
	@Override
	public void receiveID(int agentID, String[] names) {
		this.myID = agentID;
		this.currentPlayers = names;
		this.random = new Random();
	}



	@Override
	public Action doMove(int agentID, GameState state) {
		// this is where you make decisions on your turn.
		updateHistogram(state);
		
		// calculate possible moves
        List<Action> possibleMoves = new ArrayList<>(Utils.generateActions(agentID, state));

        //choose a random item from that list and return it
        int moveToMake = random.nextInt(possibleMoves.size());
        return possibleMoves.get(moveToMake);
	}
	
	/**
	 * A sample history measure, keeping track of how many times a player has made a given move.
	 * 
	 * For the 'learning' track, this will work across games. You'd probably want to do something more
	 * complex, but this is just provided as an example :).
	 * 
	 * @param state the current game state
	 */
	private void updateHistogram(GameState state) {
		// calculate history for previous moves
		List<HistoryEntry> lastMoves = getLastMoves(state);
		for (HistoryEntry entry : lastMoves) {
			
			String playerName = currentPlayers[entry.playerID];
			
			//grab this histogram for this player
			Map<Action, Integer> histogram = actionHistory.get(playerName);
			if (histogram == null) {
				histogram = new HashMap<>();
				actionHistory.put(playerName, histogram);
			}
			
			// update the histogram
			int previousValue = histogram.getOrDefault(entry.action, 0);
			histogram.put(entry.action, previousValue+1);
		}
	}
	
	/**
	 * Method to get all moves made since this agent's last go.
	 * 
	 * @param state the current game state
	 * @return the moves made since the agent's last go.
	 */
	private List<HistoryEntry> getLastMoves(GameState state) {
		List<HistoryEntry> allHistory = state.getActionHistory();
		int historySize = allHistory.size();
		
		List<HistoryEntry> movesSinceLastGo = new ArrayList<>();
		
		//we want the last n moves, if possible
		int movesToGet = Math.min(state.getPlayerCount(), historySize);
		
		for (int i = 0; i < movesToGet; i++) {
			int moveIndex = (historySize - 1);
			
			//negative game events are events that don't belong to any player, eg. game setup
			HistoryEntry entry = allHistory.get(moveIndex);
			if (entry.playerID < 0) {
				continue;
			}
			
			movesSinceLastGo.add(allHistory.get(moveIndex));
		}
		
		return movesSinceLastGo;
	}

}
